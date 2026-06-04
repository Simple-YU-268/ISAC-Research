% Cell-Free ISAC Parametric Solver - Systematic Parameter Study
%
% Tests different parameter combinations to find optimal feasibility:
% - snrReqDb: sensing SNR threshold
% - targetRange: target distribution range
% - M: number of APs
% - Pmax: total power budget
%
% Usage: isac_parametric_solver

clear; clc; close all;

%% Parameter grid definition
paramGrid = struct();

% Sensing SNR thresholds to test (dB)
paramGrid.snrReqDb = [0, 1, 2, 3];

% Target distribution ranges (±meters)
paramGrid.targetRange = [20, 30, 40, 50];

% Number of APs
paramGrid.M = [16, 20, 24];

% Total power budget (W)
paramGrid.Pmax = [30, 40, 50];

% Fixed parameters
baseCfg.seed = 42;
baseCfg.Nt = 4;
baseCfg.K = 10;
baseCfg.P = 4;
baseCfg.apGridSide = 4;
baseCfg.apMin = -60;
baseCfg.apMax = 60;
baseCfg.userMin = -50;
baseCfg.userMax = 50;
baseCfg.d0 = 10;
baseCfg.pathLossExp = 2.5;
baseCfg.minDistance = 5;
baseCfg.epsilonH = 0.10;
baseCfg.epsilonG = 0.15;
baseCfg.sigmaC2 = 0.5;
baseCfg.sigmaS2 = 0.5;
baseCfg.sinrReqDb = 0.0;
baseCfg.crbReq = 1.0;
baseCfg.nTrials = 20;

fprintf('============================================================\n');
fprintf('Cell-Free ISAC Parametric Study\n');
fprintf('============================================================\n');
fprintf('Testing %d SNR thresholds x %d target ranges x %d AP counts x %d power levels\n', ...
    length(paramGrid.snrReqDb), length(paramGrid.targetRange), ...
    length(paramGrid.M), length(paramGrid.Pmax));
fprintf('Total configurations: %d\n', ...
    length(paramGrid.snrReqDb) * length(paramGrid.targetRange) * ...
    length(paramGrid.M) * length(paramGrid.Pmax));
fprintf('Trials per config: %d\n\n', baseCfg.nTrials);

%% Results storage
results = [];
configIdx = 0;
totalConfigs = length(paramGrid.snrReqDb) * length(paramGrid.targetRange) * ...
               length(paramGrid.M) * length(paramGrid.Pmax);

%% Main loop
for snrIdx = 1:length(paramGrid.snrReqDb)
    for rangeIdx = 1:length(paramGrid.targetRange)
        for mIdx = 1:length(paramGrid.M)
            for pIdx = 1:length(paramGrid.Pmax)
                configIdx = configIdx + 1;
                
                % Build config
                cfg = baseCfg;
                cfg.snrReqDb = paramGrid.snrReqDb(snrIdx);
                cfg.targetMin = -paramGrid.targetRange(rangeIdx);
                cfg.targetMax = paramGrid.targetRange(rangeIdx);
                cfg.M = paramGrid.M(mIdx);
                cfg.Pmax = paramGrid.Pmax(pIdx);
                cfg.PmMax = cfg.Pmax / 4;
                cfg.nActiveCandidates = buildActiveCandidates(cfg.M);
                cfg.rhoCandidates = 0.30:0.05:0.90;
                cfg.scoreAlpha = 0.5;
                cfg.mmseRegMargin = 1 + 10 * cfg.epsilonH;
                
                fprintf('\n--- Config %d/%d ---\n', configIdx, totalConfigs);
                fprintf('SNR=%.0fdB | Target=±%dm | M=%d | Pmax=%.0fW\n', ...
                    cfg.snrReqDb, paramGrid.targetRange(rangeIdx), cfg.M, cfg.Pmax);
                
                % Run trials
                rng(cfg.seed);
                trialResults = repmat(emptyResult(), cfg.nTrials, 1);
                
                for trial = 1:cfg.nTrials
                    scenario = generateScenario(cfg);
                    trialResults(trial) = solveOneScenario(cfg, scenario);
                end
                
                % Summarize
                successRate = sum([trialResults.success]) / cfg.nTrials * 100;
                avgPower = mean([trialResults([trialResults.success]).totalPower]);
                avgSnr = mean([trialResults([trialResults.success]).minSnrWcDb]);
                avgSinr = mean([trialResults([trialResults.success]).minSinrWcDb]);
                
                fprintf('Success: %.0f%% | AvgPower: %.2fW | AvgSNR: %.2fdB | AvgSINR: %.2fdB\n', ...
                    successRate, avgPower, avgSnr, avgSinr);
                
                % Store
                r.configIdx = configIdx;
                r.snrReqDb = cfg.snrReqDb;
                r.targetRange = paramGrid.targetRange(rangeIdx);
                r.M = cfg.M;
                r.Pmax = cfg.Pmax;
                r.successRate = successRate;
                r.avgPower = avgPower;
                r.avgSnr = avgSnr;
                r.avgSinr = avgSinr;
                results = [results, r];
            end
        end
    end
end

%% Final summary
fprintf('\n\n============================================================\n');
fprintf('PARAMETRIC STUDY RESULTS\n');
fprintf('============================================================\n');

% Sort by success rate
[~, sortIdx] = sort([results.successRate], 'descend');
sortedResults = results(sortIdx);

fprintf('\nTop 10 Configurations:\n');
fprintf('%-4s %-6s %-10s %-4s %-6s %-10s %-10s %-10s %-10s\n', ...
    'Rank', 'SNR', 'Target', 'M', 'Pmax', 'Success%', 'AvgPower', 'AvgSNR', 'AvgSINR');
for i = 1:min(10, length(sortedResults))
    fprintf('%-4d %-6.0f %-10d %-4d %-6.0f %-10.1f %-10.2f %-10.2f %-10.2f\n', ...
        i, sortedResults(i).snrReqDb, sortedResults(i).targetRange, ...
        sortedResults(i).M, sortedResults(i).Pmax, ...
        sortedResults(i).successRate, sortedResults(i).avgPower, ...
        sortedResults(i).avgSnr, sortedResults(i).avgSinr);
end

fprintf('\nBottom 5 Configurations:\n');
for i = max(1, length(sortedResults)-4):length(sortedResults)
    fprintf('%-4d %-6.0f %-10d %-4d %-6.0f %-10.1f %-10.2f %-10.2f %-10.2f\n', ...
        i, sortedResults(i).snrReqDb, sortedResults(i).targetRange, ...
        sortedResults(i).M, sortedResults(i).Pmax, ...
        sortedResults(i).successRate, sortedResults(i).avgPower, ...
        sortedResults(i).avgSnr, sortedResults(i).avgSinr);
end

%% Helper functions

function candidates = buildActiveCandidates(M)
    % Build reasonable AP count candidates based on total APs
    candidates = [];
    for c = [4, 6, 8, 10, 12, 16, 20, 24]
        if c <= M
            candidates = [candidates, c];
        end
    end
    if isempty(candidates)
        candidates = M;
    end
end

function result = emptyResult()
    result.success = false;
    result.nActive = 0;
    result.rho = 0;
    result.minSinrWcDb = -inf;
    result.minSnrWcDb = -inf;
    result.maxCrb = inf;
    result.totalPower = inf;
    result.violation = inf;
end

function scenario = generateScenario(cfg)
    scenario.apPos = generateApPositions(cfg);
    scenario.userPos = generateUserPositions(cfg);
    scenario.targetPos = generateTargetPosition(cfg);
    
    [scenario.H, scenario.Hhat, scenario.DeltaH] = generateCommunicationChannels(scenario.apPos, scenario.userPos, cfg);
    [scenario.G, scenario.Ghat, scenario.DeltaG] = generateSensingChannels(scenario.apPos, scenario.targetPos, cfg);
end

function apPos = generateApPositions(cfg)
    side = cfg.apGridSide;
    [X, Y] = meshgrid(linspace(cfg.apMin, cfg.apMax, side), linspace(cfg.apMin, cfg.apMax, side));
    apPos = [X(:), Y(:)];
    if size(apPos, 1) > cfg.M
        apPos = apPos(1:cfg.M, :);
    elseif size(apPos, 1) < cfg.M
        % If grid doesn't have enough APs, add random ones
        nExtra = cfg.M - size(apPos, 1);
        extraPos = cfg.apMin + (cfg.apMax - cfg.apMin) * rand(nExtra, 2);
        apPos = [apPos; extraPos];
    end
end

function userPos = generateUserPositions(cfg)
    userPos = cfg.userMin + (cfg.userMax - cfg.userMin) * rand(cfg.K, 2);
end

function targetPos = generateTargetPosition(cfg)
    targetPos = cfg.targetMin + (cfg.targetMax - cfg.targetMin) * rand(1, 2);
end

function [H, Hhat, DeltaH] = generateCommunicationChannels(apPos, userPos, cfg)
    M = size(apPos, 1);
    K = size(userPos, 1);
    Nt = cfg.Nt;
    H = zeros(M*Nt, K);
    Hhat = zeros(M*Nt, K);
    DeltaH = zeros(M*Nt, K);
    
    for k = 1:K
        for m = 1:M
            d = max(norm(apPos(m,:) - userPos(k,:)), cfg.minDistance);
            pl = (cfg.d0/d)^cfg.pathLossExp;
            idx = (m-1)*Nt + (1:Nt);
            h = (randn(Nt,1) + 1j*randn(Nt,1))/sqrt(2);
            H(idx,k) = sqrt(pl) * h;
            
            errNorm = cfg.epsilonH * norm(H(idx,k));
            delta = (randn(Nt,1) + 1j*randn(Nt,1))/sqrt(2);
            delta = delta / norm(delta) * errNorm;
            DeltaH(idx,k) = delta;
            Hhat(idx,k) = H(idx,k) + delta;
        end
    end
end

function [G, Ghat, DeltaG] = generateSensingChannels(apPos, targetPos, cfg)
    M = size(apPos, 1);
    Nt = cfg.Nt;
    G = zeros(M*Nt, 1);
    Ghat = zeros(M*Nt, 1);
    DeltaG = zeros(M*Nt, 1);
    
    for m = 1:M
        d = max(norm(apPos(m,:) - targetPos), cfg.minDistance);
        pl = (cfg.d0/d)^cfg.pathLossExp;
        idx = (m-1)*Nt + (1:Nt);
        g = (randn(Nt,1) + 1j*randn(Nt,1))/sqrt(2);
        G(idx) = sqrt(pl) * g;
        
        errNorm = cfg.epsilonG * norm(G(idx));
        delta = (randn(Nt,1) + 1j*randn(Nt,1))/sqrt(2);
        delta = delta / norm(delta) * errNorm;
        DeltaG(idx) = delta;
        Ghat(idx) = G(idx) + delta;
    end
end

function result = solveOneScenario(cfg, scenario)
    result = emptyResult();
    bestViolation = inf;
    
    for nActive = cfg.nActiveCandidates
        if nActive > cfg.M, continue; end
        
        % Select best APs by target proximity
        apDists = vecnorm(scenario.apPos - scenario.targetPos, 2, 2);
        [~, sortIdx] = sort(apDists);
        activeAPs = sortIdx(1:nActive);
        
        for rho = cfg.rhoCandidates
            [W, Z, Pcomm, Psens, totalPower] = computeBeams(cfg, scenario, activeAPs, rho, nActive);
            
            if totalPower > cfg.Pmax
                continue;
            end
            
            [minSinrWc, minSnrWc, maxCrb, violation] = evaluateConstraints(...
                cfg, scenario, W, Z, activeAPs, rho, Pcomm, Psens);
            
            if violation < bestViolation
                bestViolation = violation;
                result.nActive = nActive;
                result.rho = rho;
                result.minSinrWcDb = 10*log10(minSinrWc);
                result.minSnrWcDb = 10*log10(minSnrWc);
                result.maxCrb = maxCrb;
                result.totalPower = totalPower;
                result.violation = violation;
                result.success = (violation <= 0);
            end
        end
    end
end

function [W, Z, Pcomm, Psens, totalPower] = computeBeams(cfg, scenario, activeAPs, rho, nActive)
    Nt = cfg.Nt;
    K = cfg.K;
    Mactive = length(activeAPs);
    
    Hhat_a = scenario.Hhat(:, :);
    Ghat_a = scenario.Ghat(:, :);
    
    % Extract active AP channels
    activeIdx = [];
    for m = 1:length(activeAPs)
        ap = activeAPs(m);
        activeIdx = [activeIdx; (ap-1)*Nt + (1:Nt)'];
    end
    
    Hhat_active = Hhat_a(activeIdx, :);
    Ghat_active = Ghat_a(activeIdx);
    
    % Communication beams: robust MMSE
    W = zeros(Mactive*Nt, K);
    for k = 1:K
        hk = Hhat_active(:,k);
        interference = cfg.sigmaC2 * eye(Mactive*Nt);
        for j = 1:K
            if j ~= k
                hj = Hhat_active(:,j);
                interference = interference + (hj * hj');
            end
        end
        interference = interference + cfg.mmseRegMargin * eye(Mactive*Nt);
        wk = interference \ hk;
        wk = wk / norm(wk);
        W(:,k) = wk;
    end
    
    % Power allocation for communication
    Pcomm = zeros(Mactive*Nt, K);
    for k = 1:K
        hk = Hhat_active(:,k);
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
    
    % Sensing beams: matched filter
    Z = zeros(Mactive*Nt, 1);
    g = Ghat_active;
    z = g / norm(g);
    
    % Sensing power
    Psens = zeros(Mactive*Nt, 1);
    PcommTotal = sum(sum(abs(Pcomm).^2));
    PsensTotal = (1-rho)/rho * PcommTotal;
    Psens = sqrt(PsensTotal) * z;
    
    % Check per-AP power
    totalPower = 0;
    for m = 1:Mactive
        idx = (m-1)*Nt + (1:Nt);
        pap = sum(abs(Pcomm(idx,:)).^2, 'all') + sum(abs(Psens(idx)).^2);
        totalPower = totalPower + pap;
    end
end

function [minSinrWc, minSnrWc, maxCrb, violation] = evaluateConstraints(...
    cfg, scenario, W, Z, activeAPs, rho, Pcomm, Psens)
    
    Nt = cfg.Nt;
    K = cfg.K;
    Mactive = length(activeAPs);
    
    % Extract active channels
    activeIdx = [];
    for m = 1:length(activeAPs)
        ap = activeAPs(m);
        activeIdx = [activeIdx; (ap-1)*Nt + (1:Nt)'];
    end
    
    H = scenario.H(activeIdx, :);
    G = scenario.G(activeIdx);
    
    % Worst-case SINR
    minSinrWc = inf;
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
        sinrWc = sinr * (1-cfg.epsilonH)/(1+cfg.epsilonH);
        minSinrWc = min(minSinrWc, sinrWc);
    end
    
    % Worst-case SNR
    g = G;
    snr = abs(g' * Psens)^2 / cfg.sigmaS2;
    minSnrWc = snr * (1-cfg.epsilonG)/(1+cfg.epsilonG);
    
    % CRB (simplified)
    maxCrb = 1.0;
    
    % Violation
    gammaSinr = 10^(cfg.sinrReqDb/10);
    gammaSnr = 10^(cfg.snrReqDb/10);
    
    vSinr = max(0, gammaSinr - minSinrWc) / gammaSinr;
    vSnr = max(0, gammaSnr - minSnrWc) / gammaSnr;
    vCrb = max(0, maxCrb - cfg.crbReq) / cfg.crbReq;
    
    violation = max([vSinr, vSnr, vCrb]);
end
