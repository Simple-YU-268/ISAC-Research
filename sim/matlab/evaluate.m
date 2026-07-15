function [sum_rate, sens_sinr_db, pcrb] = evaluate(W, Z, b, prm)
%EVALUATE  Compute sum-rate, sensing SINR (dB), and PCRB trace for a candidate.

K = prm.K; P = prm.P; N = prm.N;
R = zeros(N, N);
for k = 1:K
    R = R + W{k};
end
R = R + Z;

sinr = zeros(K, 1);
for k = 1:K
    hk = prm.H(:, k);
    sig = real(hk' * W{k} * hk);
    interf = 0;
    for j = setdiff(1:K, k)
        interf = interf + real(hk' * W{j} * hk);
    end
    sinr(k) = sig / (interf + prm.sigma_c2);
end
sum_rate = sum(log2(1 + sinr));

sens_sinr_db = zeros(P, 1);
for p = 1:P
    gp = prm.G(:, p);
    sens_sinr_db(p) = 10 * log10(max(real(gp' * Z * gp), 1e-30) / prm.sigma_s2);
end

pcrb = zeros(P, 1);
for p = 1:P
    gp = prm.G(:, p);
    Jp = real(gp' * R * gp) / prm.sigma_s2;
    if Jp > 1e-9
        pcrb(p) = 1 / Jp;
    else
        pcrb(p) = inf;
    end
end

end
