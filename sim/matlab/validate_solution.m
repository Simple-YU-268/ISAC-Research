function [max_violation, viol_report] = validate_solution(prm, W, Z, mu, b, M_p, tol)
%VALIDATE_SOLUTION  Check feasibility of a candidate (W, Z, mu, b, M_p).
%
%   Inputs:
%     prm  : parameter struct
%     W    : {K x 1} cell of N x N Hermitian PSD
%     Z    : N x N Hermitian PSD
%     mu   : (K x 1) S-Procedure multipliers (or empty if unused)
%     b    : (M x P) assignment matrix
%     M_p  : (N_theta x N_theta x P) or (1 x 1 x P) PCRB auxiliary matrices
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
max_violation = -inf;

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
    if prm.use_s_procedure
        Ak = (1 / prm.gamma_k(k)) * W{k};
        for j = setdiff(1:K, k)
            Ak = Ak - W{j};
        end
        top_left = Ak + mu(k) * eye(N);
        top_right = Ak * hk;
        bot_left = hk' * Ak;
        hk_norm2 = real(hk' * hk);
        bot_right = real(hk' * Ak * hk) - prm.sigma_c2 - mu(k) * prm.eps_h^2 * hk_norm2;
        L = [top_left, top_right; bot_left, bot_right];
        d = eig(L, 'vector');
        viol = max(0, -min(real(d)));
    else
        sig = real(hk' * W{k} * hk);
        interf = 0;
        for j = setdiff(1:K, k)
            interf = interf + real(hk' * W{j} * hk);
        end
        lhs = prm.gamma_k(k) * (interf + prm.sigma_c2);
        viol = max(0, lhs - sig);
    end
    viol_report.sinr(k) = viol;
    max_violation = max(max_violation, viol);
end

% (C2) Sensing SINR
if ~isfield(prm, 'enable_sensing_sinr') || prm.enable_sensing_sinr
    for p = 1:P
        gp = prm.G(:, p);
        lhs = prm.gamma_PoD(p) * prm.sigma_s2;
        val = real(gp' * Z * gp);
        viol = max(0, lhs - val);
        viol_report.sensing_sinr(p) = viol;
        max_violation = max(max_violation, viol);
    end
end

% (C3)(C4) Exact trace-of-inverse PCRB Schur LMI.
if ~isfield(prm, 'enable_pcrb') || prm.enable_pcrb
    for p = 1:P
        Dp = prm.D(:,:,p);
        Jp = 2 * real(Dp' * R * Dp) / prm.sigma_s2;
        if prm.N_theta == 1
            viol = max(0, 1/Jp - M_p(1,1,p));
        else
            Schur = [M_p(:,:,p), eye(N_theta); eye(N_theta), Jp];
            d = eig(Schur, 'vector');
            viol = max(0, -min(real(d)));
        end
        viol_report.pcrb_lower(p) = viol;
        max_violation = max(max_violation, viol);
        viol = max(0, real(trace(M_p(:,:,p))) - prm.Gamma_track(p));
        viol_report.pcrb_upper(p) = viol;
        max_violation = max(max_violation, viol);
    end
end

% (C5) per-AP power ceiling (uniform)
for m = 1:M
    pwr = real(trace(E{m} * R));
    viol = max(0, pwr - prm.Pmax);
    viol_report.power(m) = viol;
    max_violation = max(max_violation, viol);
end

% (C6) service count: only active targets are constrained
for p = 1:P
    if isfield(prm, 'active_targets') && ~ismember(p, prm.active_targets)
        viol = 0;
    else
        viol = abs(sum(b(:, p)) - prm.N_req);
    end
    viol_report.service_count(p) = viol;
    max_violation = max(max_violation, viol);
end

% (C10) box
viol_report.b_box = max([0, max(-b(:)), max(b(:) - 1)]);
max_violation = max(max_violation, viol_report.b_box);
viol_report.tolerance = tol;
viol_report.is_feasible = max_violation <= tol;

end
