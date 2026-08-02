function [W, Z, mu, b, M_p, status, S_p] = solve_p3_sca_t(prm, W_prev, b_prev, eta_rank, eta_b, b_fixed)
%SOLVE_P3_SCA_T  Solve (P3-SCA-t) via CVX/SDPT3 or MOSEK
%
%   Optional: b_fixed (M x P) forces b to be constant (used for final re-solve).
%   Z is the aggregate sensing covariance sum_p S_p(:,:,p), retained for
%   compatibility; the seventh output is the target-specific covariance tensor.

if nargin < 6 || isempty(b_fixed)
    b_fixed = [];
end
partial_b_fixing = isempty(b_fixed) && isfield(prm, 'b_fixed_mask') && ...
    any(prm.b_fixed_mask(:));
if partial_b_fixing
    assert(isequal(size(prm.b_fixed_mask), [prm.M, prm.P]) && ...
        isequal(size(prm.b_fixed_values), [prm.M, prm.P]), ...
        'Partial b-fixing masks must have size M-by-P.');
end

K = prm.K; P = prm.P; N = prm.N; M = prm.M; Nt = prm.N / prm.M;
N_theta = prm.N_theta;
E = build_E_m(M, Nt);
if isfield(prm, 'sensing_min_power') && ~isempty(prm.sensing_min_power)
    sensing_min_power = prm.sensing_min_power;
else
    sensing_min_power = 0;  % legacy authorization-only model
end

% Eigenvectors for DC rank-1 linearization
U = cell(K,1);
for k = 1:K
    [V, D] = eig(W_prev{k}, 'vector');
    [~, idx] = max(D);
    u = V(:, idx);
    U{k} = - (u * u');   % negative outer product, Hermitian
end

% Binary DC linearization coefficient: c_mp = 1 - 2 b_prev
c_mp = 1 - 2 * b_prev;
if isfield(prm, 'rho_b_prox') && ~isempty(prm.rho_b_prox)
    rho_b_prox = prm.rho_b_prox;
else
    rho_b_prox = 0;
end

if isfield(prm, 'cvx_quiet') && ~prm.cvx_quiet
    cvx_begin
else
    cvx_begin quiet
end
    if isfield(prm, 'solver') && strcmpi(prm.solver, 'mosek')
        cvx_solver mosek
        if isfield(prm, 'mosek_tol_rel_gap')
            cvx_solver_settings('MSK_DPAR_INTPNT_TOL_REL_GAP', prm.mosek_tol_rel_gap);
        end
        if isfield(prm, 'mosek_tol_pfeas')
            cvx_solver_settings('MSK_DPAR_INTPNT_TOL_PFEAS', prm.mosek_tol_pfeas);
        end
        if isfield(prm, 'mosek_max_time') && isfinite(prm.mosek_max_time)
            cvx_solver_settings('MSK_DPAR_OPTIMIZER_MAX_TIME', prm.mosek_max_time);
        end
    else
        cvx_solver SDPT3
    end
    cvx_precision default
    variable W_cvx(N,N,K) hermitian
    variable S_p_cvx(N,N,P) hermitian
    variable mu_cvx(K) nonnegative
    variable M_p_cvx(N_theta, N_theta, P) hermitian
    if isempty(b_fixed)
        variable b_cvx(M,P) nonnegative
    else
        b_cvx = b_fixed;
    end

    % ---------- constraints ----------
    S_total = sum(S_p_cvx, 3);
    R_X = sum(W_cvx, 3) + S_total;

    % (P3-C1) SINR
    for k = 1:K
        hk = prm.H(:, k);
        if prm.use_s_procedure
            Ak = (1 / prm.gamma_k(k)) * W_cvx(:,:,k);
            for j = setdiff(1:K, k)
                Ak = Ak - W_cvx(:,:,j);
            end
            if ~isfield(prm, 'sensing_waveform_cancelled_at_ue') || ...
                    ~prm.sensing_waveform_cancelled_at_ue
                Ak = Ak - S_total;
            end
            top_left = Ak + mu_cvx(k) * eye(N);
            top_right = Ak * hk;
            bot_left = hk' * Ak;
            hk_norm2 = real(hk' * hk);  % ||hat h_k||^2
            bot_right = real(hk' * Ak * hk) - prm.sigma_c2 - mu_cvx(k) * prm.eps_h^2 * hk_norm2;
            [top_left, top_right; bot_left, bot_right] == hermitian_semidefinite(N+1);
        else
            sig = real(hk' * W_cvx(:,:,k) * hk);
            interf = 0;
            for j = setdiff(1:K,k)
                interf = interf + real(hk' * W_cvx(:,:,j) * hk);
            end
            if ~isfield(prm, 'sensing_waveform_cancelled_at_ue') || ...
                    ~prm.sensing_waveform_cancelled_at_ue
                interf = interf + real(hk' * S_total * hk);
            end
            prm.gamma_k(k) * (interf + prm.sigma_c2) <= sig;
        end
    end

    % (P3-C2) sensing SINR (can be disabled for the communication-only baseline)
    if ~isfield(prm, 'enable_sensing_sinr') || prm.enable_sensing_sinr
        for p = 1:P
            gp = prm.G(:, p);
            real(gp' * S_p_cvx(:,:,p) * gp) >= prm.gamma_PoD(p) * prm.sigma_s2;
        end
    end

    % (C3)(C4) Exact trace-of-inverse PCRB Schur LMI.
    if ~isfield(prm, 'enable_pcrb') || prm.enable_pcrb
        for p = 1:P
            Dp = prm.D(:,:,p);
            % Dedicated, known sensing waveform for target p. Communication
            % covariances do not receive PCRB credit in this architecture.
            J_p = 2 * real(Dp' * S_p_cvx(:,:,p) * Dp) / prm.sigma_s2;
            if N_theta == 1
                inv_pos(J_p) <= M_p_cvx(1,1,p);
            else
                [M_p_cvx(:,:,p), eye(N_theta); ...
                 eye(N_theta),              J_p      ] == hermitian_semidefinite(2 * N_theta);
            end
            real(trace(M_p_cvx(:,:,p))) <= prm.Gamma_track(p);
        end
    end

    % (P3-C5) total per-AP transmit-hardware power ceiling.
    for m = 1:M
        real(trace(E{m} * R_X)) <= prm.Pmax;
    end

    % (P3-C5b) AP-target dedicated sensing participation. A selected AP
    % must radiate at least sensing_min_power toward its assigned target;
    % communication transmission remains globally cooperative and ungated.
    for p = 1:P
        for m = 1:M
            real(trace(E{m} * S_p_cvx(:,:,p))) >= sensing_min_power * b_cvx(m,p);
            real(trace(E{m} * S_p_cvx(:,:,p))) <= prm.Pmax * b_cvx(m,p);
        end
    end

    % (P3-C6) service count: only active targets must be served by exactly N_req APs
    if isempty(b_fixed)
        for p = 1:P
            if isfield(prm, 'active_targets') && ~ismember(p, prm.active_targets)
                continue;  % inactive target: no service constraint
            end
            sum(b_cvx(:, p)) == prm.N_req;
        end
    end

    % (P3-C7)(C8) PSD handled by hermitian declaration + semidefinite
    for k = 1:K
        W_cvx(:,:,k) == hermitian_semidefinite(N);
    end
    for p = 1:P
        S_p_cvx(:,:,p) == hermitian_semidefinite(N);
    end
    % Optional ablation baseline: independently generated AP sensing signals.
    % The proposed architecture leaves these cross-AP covariance blocks free
    % and can therefore exploit coherent cooperative sensing beamforming.
    if isfield(prm, 'sensing_covariance_structure') && ...
            strcmpi(prm.sensing_covariance_structure, 'block_diagonal')
        for p = 1:P
            for m1 = 1:M
                rows1 = (m1-1)*Nt + (1:Nt);
                for m2 = 1:M
                    if m1 == m2, continue; end
                    rows2 = (m2-1)*Nt + (1:Nt);
                    S_p_cvx(rows1, rows2, p) == 0;
                end
            end
        end
    end

    if isempty(b_fixed)
        b_cvx <= 1;
        if partial_b_fixing
            b_cvx(prm.b_fixed_mask) == prm.b_fixed_values(prm.b_fixed_mask);
        end
    end

    % ---------- objective ----------

    main_obj = 0;
    for k = 1:K
        main_obj = main_obj + (1 + eta_rank) * real(trace(W_cvx(:,:,k)));
    end
    for p = 1:P
        main_obj = main_obj + real(trace(S_p_cvx(:,:,p)));
    end

    rank_pen = 0;
    for k = 1:K
        rank_pen = rank_pen + real(trace(U{k} * W_cvx(:,:,k)));
    end

    bin_pen = sum(sum(c_mp .* b_cvx)) + sum(b_prev(:).^2);
    % Proximal DC term: keeps the association update inside a controlled
    % trust region while the binary penalty is continued.
prox_b = 0.5 * rho_b_prox * sum(sum_square_abs(b_cvx - b_prev));

    minimize( main_obj + eta_rank * rank_pen + eta_b * bin_pen + prox_b )
cvx_end

status = cvx_status;
if ~contains(cvx_status, 'Solved')
    fprintf('CVX status: %s\n', cvx_status);
end

% Extract outputs
W = cell(K,1);
for k = 1:K
    W{k} = full(W_cvx(:,:,k));
end
S_p = full(S_p_cvx);
Z = sum(S_p, 3);
mu = full(mu_cvx);
b = full(b_cvx);
    M_p = full(M_p_cvx);
    if N_theta == 1
        % Preserve consistent 3-D shape for downstream callers
        M_p = reshape(M_p, 1, 1, P);
    end

end

function E = build_E_m(M, Nt)
%BUILD_E_M  Local helper kept for solve_p3_sca_t internal use.
N = M * Nt;
E = cell(M, 1);
for m = 1:M
    Em = zeros(N, N);
    Em((m-1)*Nt + 1 : m*Nt, (m-1)*Nt + 1 : m*Nt) = eye(Nt);
    E{m} = Em;
end
end
