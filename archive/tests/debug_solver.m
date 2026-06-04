% Debug the actual solver vs theoretical limit
clear; rng(42);

% Config
cfg.M = 16; cfg.Nt = 4; cfg.K = 10; cfg.P = 4;
cfg.apGridSide = 4; cfg.apMin = -60; cfg.apMax = 60;
cfg.userMin = -50; cfg.userMax = 50;
cfg.targetMin = -50; cfg.targetMax = 50;
cfg.d0 = 10; cfg.pathLossExp = 2.5; cfg.minDistance = 5;
cfg.epsilonH = 0.10; cfg.epsilonG = 0.15;
cfg.sigmaC2 = 0.5; cfg.sigmaS2 = 0.5;
cfg.Pmax = 30.0; cfg.sinrReqDb = 0.0; cfg.snrReqDb = 3.0; cfg.crbReq = 1.0;
cfg.PmMax = cfg.Pmax / 4;
cfg.mmseRegMargin = 1 + 10 * cfg.epsilonH;

% Generate scenario
apPos = [];
for x = linspace(cfg.apMin, cfg.apMax, cfg.apGridSide)
    for y = linspace(cfg.apMin, cfg.apMax, cfg.apGridSide)
        apPos = [apPos; x, y];
    end
end
userPos = cfg.userMin + (cfg.userMax - cfg.userMin) * rand(cfg.K, 2);
targetPos = cfg.targetMin + (cfg.targetMax - cfg.targetMin) * rand(1, 2);

fprintf('Target position: [%.1f, %.1f]\n', targetPos);
apDists = sqrt(sum((apPos - targetPos).^2, 2));
fprintf('AP distances: min=%.1f, max=%.1f\n', min(apDists), max(apDists));

% Theoretical best SNR (all APs, rho=0.5)
pl = (cfg.d0./max(apDists, cfg.minDistance)).^cfg.pathLossExp;
Psens_total = cfg.Pmax * 0.5;
g_total = sum(sqrt(pl));
theoretical_snr = 10*log10((g_total * sqrt(Psens_total/cfg.M))^2 * cfg.M / cfg.sigmaS2 * (1-cfg.epsilonG)/(1+cfg.epsilonG));
fprintf('Theoretical best SNR = %.2f dB\n', theoretical_snr);

% Generate channels
Nt = cfg.Nt; K = cfg.K; M = cfg.M;
H = zeros(M*Nt, K);
for k = 1:K
    for m = 1:M
        d = max(norm(apPos(m,:) - userPos(k,:)), cfg.minDistance);
        pl_val = (cfg.d0/d)^cfg.pathLossExp;
        idx = (m-1)*Nt + (1:Nt);
        H(idx,k) = sqrt(pl_val) * (randn(Nt,1) + 1j*randn(Nt,1))/sqrt(2);
    end
end

G = zeros(M*Nt, 1);
for m = 1:M
    d = max(norm(apPos(m,:) - targetPos), cfg.minDistance);
    pl_val = (cfg.d0/d)^cfg.pathLossExp;
    idx = (m-1)*Nt + (1:Nt);
    G(idx) = sqrt(pl_val) * (randn(Nt,1) + 1j*randn(Nt,1))/sqrt(2);
end

% Communication beams (MMSE)
W = zeros(M*Nt, K);
for k = 1:K
    hk = H(:,k);
    interference = cfg.sigmaC2 * eye(M*Nt);
    for j = 1:K
        if j ~= k
            interference = interference + (H(:,j) * H(:,j)');
        end
    end
    interference = interference + cfg.mmseRegMargin * eye(M*Nt);
    wk = interference \ hk;
    wk = wk / norm(wk);
    W(:,k) = wk;
end

% Power allocation
Pcomm = zeros(M*Nt, K);
for k = 1:K
    hk = H(:,k);
    desired = abs(hk' * W(:,k))^2;
    interference = cfg.sigmaC2;
    for j = 1:K
        if j ~= k
            interference = interference + abs(hk' * W(:,j))^2;
        end
    end
    gammaLin = 10^(cfg.sinrReqDb/10);
    pk = gammaLin * interference / desired;
    Pcomm(:,k) = pk * W(:,k);
end

% Sensing beam
rho = 0.5;
z = G / norm(G);
PcommTotal = sum(sum(abs(Pcomm).^2));
PsensTotal = (1-rho)/rho * PcommTotal;
Psens = sqrt(PsensTotal) * z;

totalPower = sum(sum(abs(Pcomm).^2)) + sum(abs(Psens).^2);

% Evaluate SNR
snr = abs(G' * Psens)^2 / cfg.sigmaS2;
snr_wc = snr * (1-cfg.epsilonG)/(1+cfg.epsilonG);

fprintf('\nActual solver with rho=0.5:\n');
fprintf('PcommTotal=%.2f W, PsensTotal=%.2f W, Total=%.2f W\n', PcommTotal, PsensTotal, totalPower);
fprintf('SNR=%.2f dB (worst-case)\n', 10*log10(snr_wc));

if totalPower > cfg.Pmax
    fprintf('POWER EXCEEDED! Scaling down...\n');
    scale = cfg.Pmax / totalPower;
    Pcomm = Pcomm * sqrt(scale);
    Psens = Psens * sqrt(scale);
    totalPower = sum(sum(abs(Pcomm).^2)) + sum(abs(Psens).^2);
    snr = abs(G' * Psens)^2 / cfg.sigmaS2;
    snr_wc = snr * (1-cfg.epsilonG)/(1+cfg.epsilonG);
    fprintf('After scaling: Total=%.2f W, SNR=%.2f dB\n', totalPower, 10*log10(snr_wc));
end

% Check SINR
minSinr = inf;
for k = 1:K
    hk = H(:,k);
    desired = abs(hk' * Pcomm(:,k))^2;
    interference = cfg.sigmaC2;
    for j = 1:K
        if j ~= k
            interference = interference + abs(hk' * Pcomm(:,j))^2;
        end
    end
    sinr = desired / interference;
    sinr_wc = sinr * (1-cfg.epsilonH)/(1+cfg.epsilonH);
    minSinr = min(minSinr, sinr_wc);
end
fprintf('SINR=%.2f dB (worst-case, required %.2f)\n', 10*log10(minSinr), cfg.sinrReqDb);
