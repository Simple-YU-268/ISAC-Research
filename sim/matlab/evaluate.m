function [sum_rate, sens_sinr_db, pcrb] = evaluate(W, S_p, b, prm)
%EVALUATE  Compute metrics for dedicated target-specific sensing covariances.

K = prm.K; P = prm.P; N = prm.N;
R = zeros(N, N);
for k = 1:K
    R = R + W{k};
end
R = R + sum(S_p, 3);
S_total = sum(S_p, 3);

sinr = zeros(K, 1);
for k = 1:K
    hk = prm.H(:, k);
    sig = real(hk' * W{k} * hk);
    interf = 0;
    for j = setdiff(1:K, k)
        interf = interf + real(hk' * W{j} * hk);
    end
    if ~isfield(prm, 'sensing_waveform_cancelled_at_ue') || ...
            ~prm.sensing_waveform_cancelled_at_ue
        interf = interf + real(hk' * S_total * hk);
    end
    sinr(k) = sig / (interf + prm.sigma_c2);
end
sum_rate = sum(log2(1 + sinr));

sens_sinr_db = zeros(P, 1);
for p = 1:P
    gp = prm.G(:, p);
    sens_sinr_db(p) = 10 * log10(max(real(gp' * S_p(:,:,p) * gp), 1e-30) / prm.sigma_s2);
end

pcrb = zeros(P, 1);
for p = 1:P
    Dp = prm.D(:,:,p);
    Jp = 2 * real(Dp' * S_p(:,:,p) * Dp) / prm.sigma_s2;
    if prm.N_theta == 1
        if Jp > 1e-9
            pcrb(p) = 1 / Jp;
        else
            pcrb(p) = inf;
        end
    else
        if min(eig(Jp)) > 1e-9
            pcrb(p) = trace(inv(Jp));
        else
            pcrb(p) = inf;
        end
    end
end
end

