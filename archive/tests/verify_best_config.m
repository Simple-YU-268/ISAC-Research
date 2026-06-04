% Verify best configuration with multiple seeds
clear; clc;

bestCfg = struct();
bestCfg.M = 16; bestCfg.Nt = 4; bestCfg.K = 10;
bestCfg.snrReq = 0; bestCfg.sinrReq = 0;  % SNR=0dB
bestCfg.epsilonH = 0.10; bestCfg.epsilonG = 0.15;
bestCfg.targetRange = 50; bestCfg.Pmax = 30;
bestCfg.apGridSide = 4; bestCfg.apMin = -60; bestCfg.apMax = 60;
bestCfg.userMin = -50; bestCfg.userMax = 50;
bestCfg.targetMin = -50; bestCfg.targetMax = 50;
bestCfg.d0 = 10; bestCfg.pathLossExp = 2.5; bestCfg.minDistance = 5;
bestCfg.sigmaC2 = 0.5; bestCfg.sigmaS2 = 0.5;
bestCfg.nActiveCandidates = [16 14 12 10 8 6 4];
bestCfg.rhoCandidates = 0.10:0.05:0.90;

fprintf('Verifying best config (SNR=0dB) with 5 different seeds...\n');
fprintf('%-10s %-10s %-10s %-10s %-10s\n', 'Seed', 'Success', 'AvgPower', 'AvgSNR', 'AvgSINR');

for seed = [42, 123, 456, 789, 999]
    bestCfg.seed = seed;
    rng(seed);
    
    nSuccess = 0;
    powers = []; snrs = []; sinrs = [];
    
    for trial = 1:20
        scenario = generateScenario(bestCfg);
        result = solveOneScenario(bestCfg, scenario);
        if result.success
            nSuccess = nSuccess + 1;
            powers = [powers, result.totalPower];
            snrs = [snrs, result.minSnrWcDb];
            sinrs = [sinrs, result.minSinrWcDb];
        end
    end
    
    if nSuccess > 0
        fprintf('%-10d %-10d %-10.1f %-10.1f %-10.1f\n', seed, nSuccess, mean(powers), mean(snrs), mean(sinrs));
    else
        fprintf('%-10d %-10d %-10s %-10s %-10s\n', seed, nSuccess, 'N/A', 'N/A', 'N/A');
    end
end

function scenario = generateScenario(cfg)
    apPos = [];
    for x = linspace(cfg.apMin, cfg.apMax, cfg.apGridSide)
        for y = linspace(cfg.apMin, cfg.apMax, cfg.apGridSide)
            apPos = [apPos; x, y];
        end
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
    result.success = false;
    result.nActive = 0; result.rho = 0;
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
            for k = 1:cfg.K
                W(:,k) = W_zf(:,k)/norm(W_zf(:,k));
            end
        else
            W = zeros(nActive*cfg.Nt, cfg.K);
            for k = 1:cfg.K
                W(:,k) = H(:,k)/norm(H(:,k));
            end
        end
        
        gammaSinrNominal = 10^(cfg.sinrReq/10) / wcFactorH;
        Pcomm = zeros(nActive*cfg.Nt, cfg.K);
        for k = 1:cfg.K
            hk = H(:,k);
            wk = W(:,k);
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
            
            gammaSnrReq = 10^(cfg.snrReq/10);
            gammaSinrReq = 10^(cfg.sinrReq/10);
            vSnr = max(0, gammaSnrReq - snrWc) / gammaSnrReq;
            vSinr = max(0, gammaSinrReq - minSinr) / gammaSinrReq;
            violation = max(vSnr, vSinr);
            
            if violation < bestViolation
                bestViolation = violation;
                result.nActive = nActive;
                result.rho = rho;
                result.minSinrWcDb = 10*log10(minSinr);
                result.minSnrWcDb = 10*log10(snrWc);
                result.totalPower = TotalPower;
                result.violation = violation;
                result.success = (violation <= 1e-6);
            end
        end
    end
end
