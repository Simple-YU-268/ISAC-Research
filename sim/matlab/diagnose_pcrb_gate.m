function diagnose_pcrb_gate()
%DIAGNOSE_PCRB_GATE  Two checks for the PCRB feasibility question.
%   1) Under the nearest-neighbor activation gate, solve a PCRB-only SDP to
%      get the true minimum trace(M_p) per target, and compare with the
%      auto-calibrated Gamma_track.
%   2) Cross-check that solve_p3_sca_t really enforces the Schur LMI: feed it
%      an undersized Gamma and independently verify the returned (M_p, R).

prm = generate_scenario(8,4,4,2,2,20,'auto','AreaSize',400,'N_req',2, ...
    'eps_h',0.05,'seed',1,'noise_snr_target',1e4);
prm.solver = 'mosek';
N = prm.N; Nt = prm.Nt; M = prm.M; P = prm.P; N_theta = prm.N_theta;
fprintf('Corrected Gamma_auto: '); fprintf('%.2f ', prm.Gamma_track); fprintf('\n');

% Nearest-neighbor assignment and activation gate
b = zeros(M, P);
for p = 1:P
    d = sqrt(sum((prm.AP_pos - prm.Target_pos(p,:)).^2, 2));
    [~, idx] = sort(d);
    b(idx(1:prm.N_req), p) = 1;
end
act = sum(b, 2) > 0;
fprintf('Active APs (nearest-neighbor): %d/%d\n', nnz(act), M);

E = build_E_m(M, Nt);

% --- 1) PCRB-only feasibility SDP under the gate --------------------------
% Test A: free trace (feasibility only). Test B: with the undersized Gamma
% caps applied, to decide whether the old Gamma is attainable under ANY
% directional covariance respecting this gate.
for cap_test = 0:1
    Gamma_cap = [45.76; 604.44];  % undersized Gamma from the flawed calibration
    cvx_begin quiet
        cvx_solver mosek
        variable R(N, N) hermitian
        variable Mp(N_theta, N_theta, P) hermitian
        minimize (0)
        subject to
            R == hermitian_semidefinite(N);
            for m = 1:M
                real(trace(E{m} * R)) <= prm.Pmax * double(act(m));
            end
            for p = 1:P
                Dp = prm.D(:,:,p);
                Jp = 2 * real(Dp' * R * Dp) / prm.sigma_s2;
                [Mp(:,:,p), eye(N_theta); eye(N_theta), Jp] == hermitian_semidefinite(2*N_theta);
                if cap_test
                    real(trace(Mp(:,:,p))) <= Gamma_cap(p);
                end
            end
    cvx_end
    if cap_test == 0
        fprintf('Test A (free trace) status: %s\n', cvx_status);
    else
        fprintf('Test B (undersized Gamma caps) status: %s\n', cvx_status);
    end
    if contains(cvx_status, 'Solved')
        for p = 1:P
            Dp = prm.D(:,:,p);
            Jp = 2 * real(Dp' * R * Dp) / prm.sigma_s2;
            fprintf('  Target %d (gated): trace(Mp)=%.2f | trace(inv(J))=%.2f | Gamma_auto=%.2f\n', ...
                p, real(trace(Mp(:,:,p))), trace(inv(Jp)), prm.Gamma_track(p));
        end
    end
end

% --- 2) Cross-check Schur enforcement in solve_p3_sca_t -------------------
prm_bad = prm;
prm_bad.Gamma_track = [45.76; 604.44];  % undersized (from the 4x-overpowered calibration)
K = prm.K;
W0 = cell(K,1);
for k = 1:K
    W0{k} = eye(N) * (prm.Pmax / prm.K / Nt);
end
b0 = ones(M, P) * (prm.N_req / M);
[~, ~, ~, ~, Mp2, status, S_p2] = solve_p3_sca_t(prm_bad, W0, b0, 0, 0);
fprintf('\nWarm-start with undersized Gamma: status=%s\n', status);
if contains(status, 'Solved')
    % Cross-check against the constraint the solver actually enforces:
    % J_p is built from the target-specific S_p only (no PCRB credit for W).
    for p = 1:P
        Dp = prm.D(:,:,p);
        Jp = 2 * real(Dp' * S_p2(:,:,p) * Dp) / prm.sigma_s2;
        S = [Mp2(:,:,p), eye(N_theta); eye(N_theta), Jp];
        S = (S + S') / 2;
        fprintf('Target %d returned: Schur mineig=%.3e | trace(Mp)=%.2f | cap=%.2f\n', ...
            p, min(eig(S)), real(trace(Mp2(:,:,p))), prm_bad.Gamma_track(p));
    end
end
end
