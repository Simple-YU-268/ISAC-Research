% Cell-Free ISAC Experiment Suite - Quick Version (20 trials)
clear; clc;

fprintf('============================================================\n');
fprintf('Cell-Free ISAC Experiment Suite (Quick)\n');
fprintf('============================================================\n\n');

if ~exist('results', 'dir'), mkdir('results'); end

baseCfg = defaultConfig();
baseCfg.nTrials = 20;

%% Exp 1: SNR Threshold
fprintf('\n=== Exp 1: SNR Threshold ===\n');
snrVals = [-5, -3, 0, 3, 5, 10];
for i = 1:length(snrVals)
    cfg = baseCfg; cfg.snrReqDb = snrVals(i);
    res = quickRun(cfg);
    fprintf('SNR=%3ddB: %2d/%2d=%3.0f%% | P=%5.1fW | SNR=%5.1fdB | SINR=%5.1fdB | N=%4.1f | rho=%.2f\n', ...
        snrVals(i), res.nSucc, cfg.nTrials, res.rate*100, res.avgP, res.avgSNR, res.avgSINR, res.avgN, res.avgRho);
end

%% Exp 2: Power Budget
fprintf('\n=== Exp 2: Power Budget ===\n');
pmaxVals = [10, 20, 30, 40, 50, 100];
for i = 1:length(pmaxVals)
    cfg = baseCfg; cfg.Pmax = pmaxVals(i);
    res = quickRun(cfg);
    fprintf('Pmax=%3dW: %2d/%2d=%3.0f%% | P=%5.1fW | SNR=%5.1fdB | SINR=%5.1fdB | N=%4.1f | rho=%.2f\n', ...
        pmaxVals(i), res.nSucc, cfg.nTrials, res.rate*100, res.avgP, res.avgSNR, res.avgSINR, res.avgN, res.avgRho);
end

%% Exp 3: AP Count
fprintf('\n=== Exp 3: AP Count ===\n');
apVals = [4, 8, 12, 16, 20, 25];
for i = 1:length(apVals)
    cfg = baseCfg; cfg.M = apVals(i);
    res = quickRun(cfg);
    fprintf('M=%2d:      %2d/%2d=%3.0f%% | P=%5.1fW | SNR=%5.1fdB | SINR=%5.1fdB | N=%4.1f | rho=%.2f\n', ...
        apVals(i), res.nSucc, cfg.nTrials, res.rate*100, res.avgP, res.avgSNR, res.avgSINR, res.avgN, res.avgRho);
end

%% Exp 4: User Count
fprintf('\n=== Exp 4: User Count ===\n');
userVals = [2, 5, 10, 15, 20];
for i = 1:length(userVals)
    cfg = baseCfg; cfg.K = userVals(i);
    res = quickRun(cfg);
    fprintf('K=%2d:      %2d/%2d=%3.0f%% | P=%5.1fW | SNR=%5.1fdB | SINR=%5.1fdB | N=%4.1f | rho=%.2f\n', ...
        userVals(i), res.nSucc, cfg.nTrials, res.rate*100, res.avgP, res.avgSNR, res.avgSINR, res.avgN, res.avgRho);
end

%% Exp 5: Target Range
fprintf('\n=== Exp 5: Target Range ===\n');
rangeVals = [20, 30, 40, 50, 60, 80];
for i = 1:length(rangeVals)
    cfg = baseCfg; cfg.targetMin = -rangeVals(i); cfg.targetMax = rangeVals(i);
    res = quickRun(cfg);
    fprintf('R=%2dm:     %2d/%2d=%3.0f%% | P=%5.1fW | SNR=%5.1fdB | SINR=%5.1fdB | N=%4.1f | rho=%.2f\n', ...
        rangeVals(i), res.nSucc, cfg.nTrials, res.rate*100, res.avgP, res.avgSNR, res.avgSINR, res.avgN, res.avgRho);
end

%% Exp 6: CSI Error
fprintf('\n=== Exp 6: CSI Error ===\n');
epsVals = [0, 0.05, 0.10, 0.15, 0.20, 0.30];
for i = 1:length(epsVals)
    cfg = baseCfg; cfg.epsilonH = epsVals(i); cfg.epsilonG = epsVals(i);
    res = quickRun(cfg);
    fprintf('eps=%.2f:  %2d/%2d=%3.0f%% | P=%5.1fW | SNR=%5.1fdB | SINR=%5.1fdB | N=%4.1f | rho=%.2f\n', ...
        epsVals(i), res.nSucc, cfg.nTrials, res.rate*100, res.avgP, res.avgSNR, res.avgSINR, res.avgN, res.avgRho);
end

%% Exp 7: Multi-seed
fprintf('\n=== Exp 7: Multi-seed Statistics ===\n');
seeds = 1:5;
rates = [];
for i = 1:length(seeds)
    cfg = baseCfg; cfg.seed = seeds(i);
    res = quickRun(cfg);
    rates(i) = res.rate;
    fprintf('Seed %d: %2d/%2d=%3.0f%%\n', seeds(i), res.nSucc, cfg.nTrials, res.rate*100);
end
fprintf('Mean=%.1f%%, Std=%.1f%%, 95%% CI=[%.1f, %.1f]\n', ...
    mean(rates)*100, std(rates)*100, (mean(rates)-1.96*std(rates)/sqrt(5))*100, (mean(rates)+1.96*std(rates)/sqrt(5))*100);

fprintf('\n=== DONE ===\n');

%% Helper Functions

function res = quickRun(cfg)
    rng(cfg.seed);
    nSucc = 0; powers = []; snrs = []; sinrs = []; nActives = []; rhos = [];
    for t = 1:cfg.nTrials
        sc = generateScenario(cfg);
        r = solveOneScenario(cfg, sc);
        if r.success
            nSucc = nSucc + 1;
            powers = [powers, r.totalPower];
            snrs = [snrs, r.minSnrWcDb];
            sinrs = [sinrs, r.minSinrWcDb];
            nActives = [nActives, r.nActive];
            rhos = [rhos, r.rho];
        end
    end
    res.nSucc = nSucc;
    res.rate = nSucc / cfg.nTrials;
    res.avgP = mean(powers);
    res.avgSNR = mean(snrs);
    res.avgSINR = mean(sinrs);
    res.avgN = mean(nActives);
    res.avgRho = mean(rhos);
end

function cfg = defaultConfig()
    cfg.seed = 42;
    cfg.M = 16; cfg.Nt = 4; cfg.K = 10;
    cfg.apGridSide = 4; cfg.apMin = -60; cfg.apMax = 60;
    cfg.userMin = -50; cfg.userMax = 50;
    cfg.targetMin = -50; cfg.targetMax = 50;
    cfg.d0 = 10; cfg.pathLossExp = 2.5; cfg.minDistance = 5;
    cfg.epsilonH = 0.10; cfg.epsilonG = 0.15;
    cfg.sigmaC2 = 0.5; cfg.sigmaS2 = 0.5;
    cfg.Pmax = 30.0; cfg.sinrReqDb = 0.0; cfg.snrReqDb = 0.0;
    cfg.nActiveCandidates = [16 14 12 10 8 6 4];
    cfg.rhoCandidates = 0.10:0.05:0.90;
    cfg.nTrials = 20;
end

function scenario = generateScenario(cfg)
    apPos = [];
    for x = linspace(cfg.apMin, cfg.apMax, cfg.apGridSide)
        for y = linspace(cfg.apMin, cfg.apMax, cfg.apGridSide)
            apPos = [apPos; x, y];
        end
    end
    nGrid = cfg.apGridSide^2;
    if cfg.M > nGrid
        % Extend with random positions if needed
        extra = cfg.M - nGrid;
        apPos = [apPos; (cfg.apMin + (cfg.apMax-cfg.apMin)*rand(extra,2))];
    end
    scenario.apPos = apPos(1:cfg.M,:);
    scenario.userPos = cfg.userMin + (cfg.userMax - cfg.userMin) * rand(cfg.K, 2);
    scenario.targetPos = cfg.targetMin + (cfg.targetMax - cfg.targetMin) * rand(1, 2);
    
    Nt = cfg.Nt; K = cfg.K; M = cfg.M;
    scenario.H = zeros(M*Nt, K);
    for k = 1:K
        for m = 1:M
            d = max(norm(scenario.apPos(m,:) - scenario.userPos(k,:)), cfg.minDistance);
            pl = (cfg.d0/d)^cfg.pathLossExp;
            idx = (m-1)*Nt + (1:Nt);
            scenario.H(idx,k) = sqrt(pl) * (randn(Nt,1) + 1j*randn(Nt,1))/sqrt(2);
        end
    end
    
    scenario.G = zeros(M*Nt, 1);
    for m = 1:M
        d = max(norm(scenario.apPos(m,:) - scenario.targetPos), cfg.minDistance);
        pl = (cfg.d0/d)^cfg.pathLossExp;
        idx = (m-1)*Nt + (1:Nt);
        scenario.G(idx) = sqrt(pl) * (randn(Nt,1) + 1j*randn(Nt,1))/sqrt(2);
    end
end

function result = solveOneScenario(cfg, scenario)
    result.success = false; result.nActive = 0; result.rho = 0;
    result.minSinrWcDb = -inf; result.minSnrWcDb = -inf;
    result.totalPower = inf; result.violation = inf;
    bestViolation = inf;
    wcFactorH = (1-cfg.epsilonH)/(1+cfg.epsilonH);
    wcFactorG = (1-cfg.epsilonG)/(1+cfg.epsilonG);
    
    for nActive = cfg.nActiveCandidates
        if nActive > cfg.M, continue; end
        apDists = sqrt(sum((scenario.apPos - scenario.targetPos).^2, 2));
        [~, sortIdx] = sort(apDists);
        activeAPs = sortIdx(1:nActive);
        activeIdx = [];
        for m = 1:nActive
            ap = activeAPs(m);
            activeIdx = [activeIdx; ((ap-1)*cfg.Nt + 1):(ap*cfg.Nt)];
        end
        H = scenario.H(activeIdx, :);
        G = scenario.G(activeIdx);
        
        if rank(H) >= cfg.K
            W_zf = H * inv(H'*H);
            W = zeros(size(W_zf));
            for k = 1:cfg.K, W(:,k) = W_zf(:,k)/norm(W_zf(:,k)); end
        else
            W = zeros(nActive*cfg.Nt, cfg.K);
            for k = 1:cfg.K, W(:,k) = H(:,k)/norm(H(:,k)); end
        end
        
        gammaSinrNominal = 10^(cfg.sinrReqDb/10) / wcFactorH;
        Pcomm = zeros(nActive*cfg.Nt, cfg.K);
        for k = 1:cfg.K
            hk = H(:,k); wk = W(:,k);
            gain = abs(hk'*wk)^2;
            pk = gammaSinrNominal * cfg.sigmaC2 / gain;
            Pcomm(:,k) = sqrt(pk) * wk;
        end
        PcommTotal = sum(sum(abs(Pcomm).^2));
        
        for rho = cfg.rhoCandidates
            PsensTotal = (1-rho)/rho * PcommTotal;
            TotalPower = PcommTotal + PsensTotal;
            if TotalPower > cfg.Pmax, continue; end
            
            z = G / norm(G);
            Psens = sqrt(PsensTotal) * z;
            snr = abs(G' * Psens)^2 / cfg.sigmaS2;
            snrWc = snr * wcFactorG;
            
            minSinr = inf;
            for k = 1:cfg.K
                hk = H(:,k);
                desired = abs(hk'*Pcomm(:,k))^2;
                interf = cfg.sigmaC2;
                for j = 1:cfg.K
                    if j~=k, interf = interf + abs(hk'*Pcomm(:,j))^2; end
                end
                sinr = desired/interf;
                sinrWc = sinr * wcFactorH;
                minSinr = min(minSinr, sinrWc);
            end
            
            gammaSnrReq = 10^(cfg.snrReqDb/10);
            gammaSinrReq = 10^(cfg.sinrReqDb/10);
            vSnr = max(0, gammaSnrReq - snrWc) / gammaSnrReq;
            vSinr = max(0, gammaSinrReq - minSinr) / gammaSinrReq;
            violation = max(vSnr, vSinr);
            
            if violation < bestViolation
                bestViolation = violation;
                result.nActive = nActive; result.rho = rho;
                result.minSinrWcDb = 10*log10(minSinr);
                result.minSnrWcDb = 10*log10(snrWc);
                result.totalPower = TotalPower;
                result.violation = violation;
                result.success = (violation <= 1e-6);
            end
        end
    end
end
