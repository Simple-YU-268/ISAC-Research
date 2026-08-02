function diagnosis = diagnose_fixed_b_slack(prm, b_fixed)
%DIAGNOSE_FIXED_B_SLACK  Minimum normalized violation for a binary topology.
%   Hardware power and AP-target Big-M constraints remain hard.  Only PCRB,
%   sensing-SINR, and communication-SINR requirements receive nonnegative,
%   dimensionless slacks.  This routine is diagnostic: it never certifies a
%   physical solution and is used only to rank infeasible fixed-b candidates.

K = prm.K; P = prm.P; N = prm.N; M = prm.M; Nt = N / M;
N_theta = prm.N_theta;
assert(isequal(size(b_fixed), [M, P]), 'b_fixed must be M-by-P.');
assert(all(b_fixed(:) == 0 | b_fixed(:) == 1), 'b_fixed must be binary.');
E = build_E_m(M, Nt);

cvx_begin quiet
    if isfield(prm, 'solver') && strcmpi(prm.solver, 'mosek')
        cvx_solver mosek
        if isfield(prm, 'recovery_mosek_max_time')
            cvx_solver_settings('MSK_DPAR_OPTIMIZER_MAX_TIME', ...
                prm.recovery_mosek_max_time);
        end
    else
        cvx_solver SDPT3
    end
    cvx_precision default
    variable W(N,N,K) hermitian
    variable S(N,N,P) hermitian
    variable mu(K) nonnegative
    variable M_p(N_theta,N_theta,P) hermitian
    variable xi_pcrb(P) nonnegative
    variable xi_sens(P) nonnegative
    variable xi_sinr(K) nonnegative

    S_total = sum(S, 3);
    R_X = sum(W, 3) + S_total;
    for k = 1:K
        hk = prm.H(:,k);
        if prm.use_s_procedure
            Ak = W(:,:,k) / prm.gamma_k(k);
            for j = setdiff(1:K, k)
                Ak = Ak - W(:,:,j);
            end
            if ~isfield(prm, 'sensing_waveform_cancelled_at_ue') || ...
                    ~prm.sensing_waveform_cancelled_at_ue
                Ak = Ak - S_total;
            end
            top_left = Ak + mu(k) * eye(N);
            top_right = Ak * hk;
            bot_left = hk' * Ak;
            bot_right = real(hk' * Ak * hk) - prm.sigma_c2 - ...
                mu(k) * prm.eps_h^2 * real(hk' * hk) + ...
                xi_sinr(k) * prm.sigma_c2;
            [top_left, top_right; bot_left, bot_right] == ...
                hermitian_semidefinite(N+1);
        else
            sig = real(hk' * W(:,:,k) * hk);
            interf = 0;
            for j = setdiff(1:K,k)
                interf = interf + real(hk' * W(:,:,j) * hk);
            end
            if ~isfield(prm, 'sensing_waveform_cancelled_at_ue') || ...
                    ~prm.sensing_waveform_cancelled_at_ue
                interf = interf + real(hk' * S_total * hk);
            end
            prm.gamma_k(k) * (interf + prm.sigma_c2) <= sig + ...
                xi_sinr(k) * prm.sigma_c2;
        end
    end

    for p = 1:P
        gp = prm.G(:,p);
        real(gp' * S(:,:,p) * gp) + xi_sens(p) * ...
            prm.gamma_PoD(p) * prm.sigma_s2 >= prm.gamma_PoD(p) * prm.sigma_s2;
        Dp = prm.D(:,:,p);
        Jp = 2 * real(Dp' * S(:,:,p) * Dp) / prm.sigma_s2;
        if N_theta == 1
            inv_pos(Jp) <= M_p(1,1,p);
        else
            [M_p(:,:,p), eye(N_theta); eye(N_theta), Jp] == ...
                hermitian_semidefinite(2 * N_theta);
        end
        real(trace(M_p(:,:,p))) <= prm.Gamma_track(p) * (1 + xi_pcrb(p));
    end

    for m = 1:M
        real(trace(E{m} * R_X)) <= prm.Pmax;
        for p = 1:P
            real(trace(E{m} * S(:,:,p))) <= prm.Pmax * b_fixed(m,p);
        end
    end
    for k = 1:K, W(:,:,k) == hermitian_semidefinite(N); end
    for p = 1:P, S(:,:,p) == hermitian_semidefinite(N); end

    minimize(sum(xi_pcrb) + sum(xi_sens) + sum(xi_sinr) + ...
        1e-4 * real(trace(R_X)) / (M * prm.Pmax))
cvx_end

diagnosis.status = cvx_status;
diagnosis.total_slack = NaN;
diagnosis.pcrb_slack = NaN(P,1);
diagnosis.sensing_sinr_slack = NaN(P,1);
diagnosis.communication_sinr_slack = NaN(K,1);
if contains(cvx_status, 'Solved')
    diagnosis.pcrb_slack = full(xi_pcrb);
    diagnosis.sensing_sinr_slack = full(xi_sens);
    diagnosis.communication_sinr_slack = full(xi_sinr);
    diagnosis.total_slack = sum(diagnosis.pcrb_slack) + ...
        sum(diagnosis.sensing_sinr_slack) + sum(diagnosis.communication_sinr_slack);
end
end
