function [max_violation, viol_report] = validate_solution(prm, W, S_p, mu, b, M_p, tol)
%VALIDATE_SOLUTION  Check feasibility of a candidate (W, S_p, mu, b, M_p).
%
%   Inputs:
%     prm  : parameter struct
%     W    : {K x 1} cell of N x N Hermitian PSD
%     S_p  : N x N x P Hermitian PSD target-specific sensing covariances
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
R = R + sum(S_p, 3);
S_total = sum(S_p, 3);

viol_report = struct();
max_violation = -inf;

% (C7) W_k PSD and (C8) target-specific sensing covariance PSD constraints
for k = 1:K
    d = eig(W{k}, 'vector');
    viol = max(0, -min(d));
    viol_report.W_psd(k) = viol;
    max_violation = max(max_violation, viol);
end
for p = 1:P
    ds = eig(S_p(:,:,p), 'vector');
    viol_report.S_p_psd(p) = max(0, -min(ds));
    max_violation = max(max_violation, viol_report.S_p_psd(p));
end

% (C1) SINR
for k = 1:K
    hk = prm.H(:, k);
    if prm.use_s_procedure
        if isempty(mu) || numel(mu) ~= K
            viol = inf;
        else
            Ak = (1 / prm.gamma_k(k)) * W{k};
            for j = setdiff(1:K, k)
                Ak = Ak - W{j};
            end
            if ~isfield(prm, 'sensing_waveform_cancelled_at_ue') || ...
                    ~prm.sensing_waveform_cancelled_at_ue
                Ak = Ak - S_total;
            end
            top_left = Ak + mu(k) * eye(N);
            top_right = Ak * hk;
            bot_left = hk' * Ak;
            hk_norm2 = real(hk' * hk);
            bot_right = real(hk' * Ak * hk) - prm.sigma_c2 - mu(k) * prm.eps_h^2 * hk_norm2;
            L = [top_left, top_right; bot_left, bot_right];
            d = eig(L, 'vector');
            viol = max(0, -min(real(d)));
        end
    else
        sig = real(hk' * W{k} * hk);
        interf = 0;
        for j = setdiff(1:K, k)
            interf = interf + real(hk' * W{j} * hk);
        end
        if ~isfield(prm, 'sensing_waveform_cancelled_at_ue') || ...
                ~prm.sensing_waveform_cancelled_at_ue
            interf = interf + real(hk' * S_total * hk);
        end
        lhs = prm.gamma_k(k) * (interf + prm.sigma_c2);
        viol = max(0, lhs - sig);
    end
    viol_report.sinr(k) = viol;
    max_violation = max(max_violation, viol);
end

% The robust S-Procedure requires nonnegative multipliers.
if prm.use_s_procedure
    if isempty(mu) || numel(mu) ~= K
        viol_report.mu_nonnegative = inf;
    else
        viol_report.mu_nonnegative = max(0, -min(real(mu(:))));
    end
    max_violation = max(max_violation, viol_report.mu_nonnegative);
end

% (C2) Sensing SINR
if ~isfield(prm, 'enable_sensing_sinr') || prm.enable_sensing_sinr
    for p = 1:P
        gp = prm.G(:, p);
        lhs = prm.gamma_PoD(p) * prm.sigma_s2;
        val = real(gp' * S_p(:,:,p) * gp);
        viol = max(0, lhs - val);
        viol_report.sensing_sinr(p) = viol;
        max_violation = max(max_violation, viol);
    end
end

% (C3)(C4) Exact trace-of-inverse PCRB Schur LMI.
if ~isfield(prm, 'enable_pcrb') || prm.enable_pcrb
    for p = 1:P
        Dp = prm.D(:,:,p);
        Jp = 2 * real(Dp' * S_p(:,:,p) * Dp) / prm.sigma_s2;
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

% (C5b) AP-target dedicated sensing Big-M constraints
for p = 1:P
    for m = 1:M
        sensing_power = real(trace(E{m} * S_p(:,:,p)));
        viol = max(0, sensing_power - prm.Pmax * b(m,p));
        viol_report.sensing_association(m,p) = viol;
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

% (C10) box and strict binarity of a final physical solution.
viol_report.b_box = max([0, max(-b(:)), max(b(:) - 1)]);
max_violation = max(max_violation, viol_report.b_box);
viol_report.b_binary = max(min(abs(b(:)), abs(b(:) - 1)));
max_violation = max(max_violation, viol_report.b_binary);
viol_report.tolerance = tol;
viol_report.is_feasible = max_violation <= tol;

end
