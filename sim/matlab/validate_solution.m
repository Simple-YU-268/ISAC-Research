function [max_violation, viol_report] = validate_solution(prm, W, Z, mu, b, M_p, tol)
%VALIDATE_SOLUTION  Check feasibility of a candidate (W, Z, mu, b, M_p).
%
%   Inputs:
%     prm  : parameter struct
%     W    : {K x 1} cell of N x N Hermitian PSD
%     Z    : N x N Hermitian PSD
%     mu   : (K x 1) S-Procedure multipliers (or empty if unused)
%     b    : (M x P) assignment matrix
%     M_p  : (P x 1) PCRB auxiliaries
%     tol  : tolerance for feasibility (default 1e-6)
%
%   Outputs:
%     max_violation : maximum absolute/relative constraint violation
%     viol_report   : struct with per-constraint violations

if nargin < 7, tol = 1e-6; end

K = prm.K; P = prm.P; N = prm.N; M = prm.M; Nt = prm.N / prm.M;
N_theta = prm.N_theta;
E = build_E_m(M, Nt);
R = zeros(N, N);
for k = 1:K
    R = R + W{k};
end
R = R + Z;

viol_report = struct();
max_violation = 0;

% (C7) W_k PSD and (C8) Z PSD
for k = 1:K
    d = eig(W{k}, 'vector');
    viol = max(0, -min(d));
    viol_report.W_psd(k) = viol;
    max_violation = max(max_violation, viol);
end
dz = eig(Z, 'vector');
viol_report.Z_psd = max(0, -min(dz));
max_violation = max(max_violation, viol_report.Z_psd);

% (C1) SINR
for k = 1:K
    hk = prm.H(:, k);
    sig = real(hk' * W{k} * hk);
    interf = 0;
    for j = setdiff(1:K, k)
        interf = interf + real(hk' * W{j} * hk);
    end
    lhs = prm.gamma_k(k) * (interf + prm.sigma_c2);
    viol = max(0, lhs - sig);
    viol_report.sinr(k) = viol;
    max_violation = max(max_violation, viol);
end

% (C2) Sensing SINR
for p = 1:P
    gp = prm.G(:, p);
    lhs = prm.gamma_PoD(p) * prm.sigma_s2;
    val = real(gp' * Z * gp);
    viol = max(0, lhs - val);
    viol_report.sensing_sinr(p) = viol;
    max_violation = max(max_violation, viol);
end

% (C3)(C4) PCRB: multi-dimensional Schur LMI (N_theta >= 2) or scalar inv_pos (N_theta=1)
for p = 1:P
    Dp = prm.D(:,:,p);
    if prm.N_theta == 1
        Jp = real(Dp' * R * Dp) / prm.sigma_s2;
        viol = max(0, 1/Jp - M_p(p));
    else
        Jp = real(Dp' * R * Dp) / prm.sigma_s2;
        Schur = [M_p(p) * eye(N_theta), eye(N_theta); eye(N_theta), Jp];
        d = eig(Schur, 'vector');
        viol = max(0, -min(d));
    end
    viol_report.pcrb_lower(p) = viol;
    max_violation = max(max_violation, viol);
    viol = max(0, M_p(p) * prm.N_theta - prm.Gamma_track(p));
    viol_report.pcrb_upper(p) = viol;
    max_violation = max(max_violation, viol);
end

% (C5'a)(C5'b) per-AP power
for m = 1:M
    pwr = real(trace(E{m} * R));
    viol = max(0, pwr - prm.Pmax * sum(b(m, :)));
    viol_report.power_gate(m) = viol;
    max_violation = max(max_violation, viol);
    viol = max(0, pwr - prm.Pmax);
    viol_report.power_hard(m) = viol;
    max_violation = max(max_violation, viol);
end

% (C6) service count
for p = 1:P
    viol = abs(sum(b(:, p)) - prm.N_req);
    viol_report.service_count(p) = viol;
    max_violation = max(max_violation, viol);
end

% (C10) box
viol_report.b_box = max([0, max(-b(:)), max(b(:) - 1)]);
max_violation = max(max_violation, viol_report.b_box);

end
