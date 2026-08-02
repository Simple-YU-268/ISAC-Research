function metrics = evaluate_isac_metrics(prm, W, S_p, mu, b, M_p)
%EVALUATE_ISAC_METRICS  Extract paper-reportable metrics from a solution.
%   Nominal SINR margins are reported in dB.  Robust S-procedure LMI
%   eigenvalue slacks are retained separately; they are not converted to dB.

K = prm.K; P = prm.P; M = prm.M; Nt = prm.Nt;
N = prm.N; E = build_E_m(M, Nt);
S_total = sum(S_p, 3);
metrics.total_power_W = 0;
metrics.ap_comm_power_W = zeros(M,1);
metrics.ap_sensing_power_W = zeros(M,1);
metrics.pcrb_ratio = NaN(P,1);
metrics.nominal_sinr_margin_dB = NaN(K,1);
metrics.nominal_sinr_linear = NaN(K,1);
metrics.robust_lmi_min_eig = NaN(K,1);
metrics.sensing_sinr_margin_dB = NaN(P,1);
metrics.sensing_sinr_linear = NaN(P,1);

for m = 1:M
    for k = 1:K
        metrics.ap_comm_power_W(m) = metrics.ap_comm_power_W(m) + real(trace(E{m}*W{k}));
    end
    metrics.ap_sensing_power_W(m) = real(trace(E{m}*S_total));
end
metrics.total_power_W = sum(metrics.ap_comm_power_W + metrics.ap_sensing_power_W);

for k = 1:K
    hk = prm.H(:,k);
    desired = real(hk' * W{k} * hk);
    interference = real(hk' * S_total * hk) + prm.sigma_c2;
    for j = setdiff(1:K,k)
        interference = interference + real(hk' * W{j} * hk);
    end
    sinr_k = max(desired / max(interference, eps), eps);
    metrics.nominal_sinr_linear(k) = sinr_k;
    metrics.nominal_sinr_margin_dB(k) = 10*log10(sinr_k / prm.gamma_k(k));
    if prm.use_s_procedure && ~isempty(mu) && numel(mu) == K
        Ak = W{k}/prm.gamma_k(k);
        for j = setdiff(1:K,k), Ak = Ak - W{j}; end
        if ~isfield(prm, 'sensing_waveform_cancelled_at_ue') || ...
                ~prm.sensing_waveform_cancelled_at_ue
            Ak = Ak - S_total;
        end
        L = [Ak + mu(k)*eye(N), Ak*hk; hk'*Ak, ...
            real(hk'*Ak*hk) - prm.sigma_c2 - mu(k)*prm.eps_h^2*real(hk'*hk)];
        metrics.robust_lmi_min_eig(k) = min(real(eig(L,'vector')));
    end
end

for p = 1:P
    gp = prm.G(:,p);
    sensing_sinr = real(gp' * S_p(:,:,p) * gp) / prm.sigma_s2;
    metrics.sensing_sinr_linear(p) = max(sensing_sinr, eps);
    metrics.sensing_sinr_margin_dB(p) = 10*log10(max(sensing_sinr/prm.gamma_PoD(p), eps));
    Dp = prm.D(:,:,p);
    Jp = 2*real(Dp' * S_p(:,:,p) * Dp)/prm.sigma_s2;
    if min(real(eig(Jp,'vector'))) > 0
        metrics.pcrb_ratio(p) = real(trace(inv(Jp))) / prm.Gamma_track(p);
    end
end
% Report aggregate application metrics using the returned physical solution.
% These are descriptive metrics, not additional optimization objectives.
metrics.sum_rate_bpsHz = sum(log2(1 + metrics.nominal_sinr_linear));
metrics.mean_sensing_sinr_linear = mean(metrics.sensing_sinr_linear);
metrics.mean_sensing_sinr_dB = 10*log10(metrics.mean_sensing_sinr_linear);
metrics.mean_pcrb_ratio = mean(metrics.pcrb_ratio, 'omitnan');

% Treat only physically material dedicated-sensing power as active.  The
% threshold must exceed the feasibility tolerance-scale numerical leakage of
% an SDP solution; otherwise inactive AP-target pairs can be spuriously
% counted after a high-dimensional MOSEK solve.  It remains three orders of
% magnitude below the 1 mW reporting floor at the default Pmax = 0.1 W.
active_threshold_W = max(1e-6, 1e-5 * prm.Pmax);
metrics.ap_target_sensing_power_W = zeros(M,P);
for p = 1:P
    for m = 1:M
        metrics.ap_target_sensing_power_W(m,p) = real(trace(E{m} * S_p(:,:,p)));
    end
end
metrics.num_nonzero_sensing_pairs = nnz(metrics.ap_target_sensing_power_W > active_threshold_W);
metrics.num_nonzero_sensing_aps = nnz(sum(metrics.ap_target_sensing_power_W, 2) > active_threshold_W);
metrics.nonzero_sensing_threshold_W = active_threshold_W;
metrics.b = b;
end
