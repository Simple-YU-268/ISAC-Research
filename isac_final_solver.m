% Cell-Free ISAC Final Solver - Combined ZF + MRT with proper power allocation
clear; clc;
cfg = defaultConfig();
rng(cfg.seed);

nSuccess = 0;
results = [];
for trial = 1:cfg.nTrials
    scenario = generateScenario(cfg);
    result = solveOneScenario(cfg, scenario);
    results = [results; result];
    
    if result.success
        nSuccess = nSuccess + 1;
        status = 'OK';
    else
        status = 'FAIL';
    end
    fprintf('Trial %02d: %s | N=%2d | rho=%.2f | SNR=%5.1fdB | SINR=%5.1fdB | P=%5.2fW\n', ...
        trial, status, result.nActive, result.rho, result.minSnrWcDb, ...
        result.minSinrWcDb, result.totalPower);
end

fprintf('\n=== RESULT: %d/%d = %.0f%% success ===\n', nSuccess, cfg.nTrials, nSuccess/cfg.nTrials*100);
if nSuccess > 0
    succ = results([results.success]);
    fprintf('Avg: Power=%.2fW, SNR=%.1fdB, SINR=%.1fdB\n', ...
        mean([succ.totalPower]), mean([succ.minSnrWcDb]), mean([succ.minSinrWcDb]));
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
    cfg.Pmax = 30.0; cfg.sinrReqDb = 0.0; cfg.snrReqDb = 3.0;
    cfg.nActiveCandidates = [16 14 12 10 8 6 4];  % Try more APs first
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
        
        % Try ZF first
        useZF = false;
        W = [];
        if rank(H) >= cfg.K
            W_zf = H * inv(H'*H);
            W = zeros(size(W_zf));
            for k = 1:cfg.K
                W(:,k) = W_zf(:,k)/norm(W_zf(:,k));
            end
            useZF = true;
        else
            % Fall back to MRT
            W = zeros(nActive*cfg.Nt, cfg.K);
            for k = 1:cfg.K
                W(:,k) = H(:,k)/norm(H(:,k));
            end
        end
        
        % Power allocation
        gammaSinrNominal = 10^(cfg.sinrReqDb/10) / wcFactorH;
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
            
            gammaSnrReq = 10^(cfg.snrReqDb/10);
            gammaSinrReq = 10^(cfg.sinrReqDb/10);
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
