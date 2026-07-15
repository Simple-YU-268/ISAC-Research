function [W, Z, mu, b, M_p, status] = solve_p3_sca_t(prm, W_prev, b_prev, eta_rank, eta_b, b_fixed)
%SOLVE_P3_SCA_T  Solve (P3-SCA-t) via CVX/SDPT3 or MOSEK
%
%   Optional: b_fixed (M x P) forces b to be constant (used for final re-solve).

if nargin < 6 || isempty(b_fixed)
    b_fixed = [];
end

K = prm.K; P = prm.P; N = prm.N; M = prm.M; Nt = prm.N / prm.M;
N_theta = prm.N_theta;
E = build_E_m(M, Nt);

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

cvx_begin quiet
    if isfield(prm, 'solver') && strcmpi(prm.solver, 'mosek')
        cvx_solver mosek
    else
        cvx_solver SDPT3
    end
    cvx_precision default
    variable W_cvx(N,N,K) hermitian
    variable Z_cvx(N,N) hermitian
    variable mu_cvx(K) nonnegative
    variable M_p_cvx(P) nonnegative
    if isempty(b_fixed)
        variable b_cvx(M,P) nonnegative
    else
        b_cvx = b_fixed;
    end

    % ---------- constraints ----------
    R_X = sum(W_cvx, 3) + Z_cvx;

    % (P3-C1) SINR
    for k = 1:K
        hk = prm.H(:, k);
        if prm.use_s_procedure
            Ak = (1 / prm.gamma_k(k)) * W_cvx(:,:,k) - sum(W_cvx(:,:,setdiff(1:K,k)), 3);
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
            prm.gamma_k(k) * (interf + prm.sigma_c2) <= sig;
        end
    end

    % (P3-C2) sensing SINR
    for p = 1:P
        gp = prm.G(:, p);
        real(gp' * Z_cvx * gp) >= prm.gamma_PoD(p) * prm.sigma_s2;
    end

    % (C3)(C4) PCRB: multi-dimensional Schur LMI (N_theta >= 2) or scalar inv_pos (N_theta=1)
    for p = 1:P
        Dp = prm.D(:,:,p);
        if prm.N_theta == 1
            J_p = real(Dp' * R_X * Dp) / prm.sigma_s2;
            inv_pos(J_p) <= M_p_cvx(p);
        else
            J_p = real(Dp' * R_X * Dp) / prm.sigma_s2;
            [M_p_cvx(p) * eye(N_theta), eye(N_theta); ...
             eye(N_theta),              J_p      ] == hermitian_semidefinite(2 * N_theta);
        end
        M_p_cvx(p) * prm.N_theta <= prm.Gamma_track(p);
    end

    % (P3-C5a) per-AP power with AP-target gate
    for m = 1:M
        real(trace(E{m} * R_X)) <= prm.Pmax * sum(b_cvx(m, :));
    end
    % (P3-C5b) hard ceiling
    for m = 1:M
        real(trace(E{m} * R_X)) <= prm.Pmax;
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
    Z_cvx == hermitian_semidefinite(N);

    if isempty(b_fixed)
        b_cvx <= 1;
    end

    % ---------- objective ----------

    main_obj = 0;
    for k = 1:K
        main_obj = main_obj + (1 + eta_rank) * real(trace(W_cvx(:,:,k)));
    end
    main_obj = main_obj + real(trace(Z_cvx));

    rank_pen = 0;
    for k = 1:K
        rank_pen = rank_pen + real(trace(U{k} * W_cvx(:,:,k)));
    end

    bin_pen = sum(sum(c_mp .* b_cvx)) + sum(b_prev(:).^2);

    minimize( main_obj + eta_rank * rank_pen + eta_b * bin_pen )
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
Z = full(Z_cvx);
mu = full(mu_cvx);
b = full(b_cvx);
M_p = full(M_p_cvx);

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
