% Cell-Free ISAC Rigorous Solver v2.0
% Based on standard form with per-target AP selection, sensing SINR, and PCRB
% Mathematical derivation: MATHEMATICAL_DERIVATION.md

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
    fprintf('Trial %02d: %s | Map=%2d | SNRwc=%5.1fdB | SINRwc=%5.1fdB | P=%5.2fW | v=%.4f\n', ...
        trial, status, result.nActiveAPs, result.minSnrWcDb, ...
        result.minSinrWcDb, result.totalPower, result.violation);
end

fprintf('\n=== RESULT: %d/%d = %.0f%% success ===\n', nSuccess, cfg.nTrials, nSuccess/cfg.nTrials*100);
if nSuccess > 0
    succ = results([results.success]);
    fprintf('Avg: Power=%.2fW, SNRwc=%.1fdB, SINRwc=%.1fdB, ActiveAPs=%.1f\n', ...
        mean([succ.totalPower]), mean([succ.minSnrWcDb]), mean([succ.minSinrWcDb]), mean([succ.nActiveAPs]));
end

function cfg = defaultConfig()
    cfg.seed = 42;
    cfg.M = 16; cfg.Nt = 4; cfg.K = 10; cfg.P = 1;  % P=1 for simplicity
    cfg.apGridSide = 4; cfg.apMin = -60; cfg.apMax = 60;
    cfg.userMin = -50; cfg.userMax = 50;
    cfg.targetMin = -50; cfg.targetMax = 50;
    cfg.d0 = 10; cfg.pathLossExp = 2.5; cfg.minDistance = 5;
    cfg.epsilonH = 0.10; cfg.epsilonG = 0.15;
    cfg.sigmaC2 = 0.5; cfg.sigmaS2 = 0.5;
    cfg.Pmax = 30.0; 
    cfg.gammaK = 1.0;  % 0 dB linear
    cfg.gammaS = 1.0;  % 0 dB linear
    cfg.gammaTrack = 1.0;  % PCRB threshold
    cfg.Nreq = 4;  % APs per target
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
    scenario.targetPos = cfg.targetMin + (cfg.targetMax - cfg.targetMin) * rand(cfg.P, 2);
    
    Nt = cfg.Nt; K = cfg.K; M = cfg.M; P = cfg.P;
    
    % Communication channels H: MNt x K
    scenario.H = zeros(M*Nt, K);
    for k = 1:K
        for m = 1:M
            d = max(norm(scenario.apPos(m,:) - scenario.userPos(k,:)), cfg.minDistance);
            pl = (cfg.d0/d)^cfg.pathLossExp;
            idx = (m-1)*Nt + (1:Nt);
            scenario.H(idx,k) = sqrt(pl) * (randn(Nt,1) + 1j*randn(Nt,1))/sqrt(2);
        end
    end
    
    % Sensing channels G: MNt x P
    scenario.G = zeros(M*Nt, P);
    for p = 1:P
        for m = 1:M
            d = max(norm(scenario.apPos(m,:) - scenario.targetPos(p,:)), cfg.minDistance);
            pl = (cfg.d0/d)^cfg.pathLossExp;
            idx = (m-1)*Nt + (1:Nt);
            scenario.G(idx,p) = sqrt(pl) * (randn(Nt,1) + 1j*randn(Nt,1))/sqrt(2);
        end
    end
end

function result = solveOneScenario(cfg, scenario)
    result.success = false;
    result.nActiveAPs = 0;
    result.minSinrWcDb = -inf; result.minSnrWcDb = -inf;
    result.totalPower = inf; result.violation = inf;
    
    % Robustness factors
    etaH = ((1-cfg.epsilonH)/(1+cfg.epsilonH))^2;
    etaG = ((1-cfg.epsilonG)/(1+cfg.epsilonG))^2;
    
    % Robust thresholds
    gammaK_robust = cfg.gammaK / etaH;
    gammaS_robust = cfg.gammaS / etaG;
    
    M = cfg.M; Nt = cfg.Nt; K = cfg.K; P = cfg.P;
    
    % Step 1: AP Selection (per target)
    b_mp = zeros(M, P);  % AP-target association
    for p = 1:P
        g_p = scenario.G(:,p);
        % Compute per-AP channel strength
        apStrength = zeros(M, 1);
        for m = 1:M
            idx = (m-1)*Nt + (1:Nt);
            apStrength(m) = norm(g_p(idx))^2;
        end
        % Select top-Nreq APs
        [~, sortIdx] = sort(apStrength, 'descend');
        selected = sortIdx(1:cfg.Nreq);
        b_mp(selected, p) = 1;
    end
    
    % Union of all active APs
    activeAPs = find(sum(b_mp, 2) > 0);
    nActive = length(activeAPs);
    result.nActiveAPs = nActive;
    
    % Step 2: Extract subchannels
    activeIdx = [];
    for i = 1:nActive
        m = activeAPs(i);
        activeIdx = [activeIdx; ((m-1)*Nt + 1):(m*Nt)];
    end
    
    H_all = scenario.H(activeIdx, :);  % nActive*Nt x K
    G_all = scenario.G(activeIdx, :);  % nActive*Nt x P
    
    % Step 3: Communication beamforming (ZF)
    Wcomm = zeros(nActive*Nt, K);
    Pcomm_per_k = zeros(K, 1);
    
    if rank(H_all) >= K
        % ZF solution
        Wzf = H_all * inv(H_all'*H_all);
        for k = 1:K
            wzf_k = Wzf(:,k);
            norm_wzf = norm(wzf_k);
            if norm_wzf > 1e-10
                % Normalized ZF beam
                w_k = wzf_k / norm_wzf;
                % Power allocation: p_k = gamma_robust * sigma^2 * ||Wzf(:,k)||^2
                Pcomm_per_k(k) = gammaK_robust * cfg.sigmaC2 * norm_wzf^2;
                Wcomm(:,k) = sqrt(Pcomm_per_k(k)) * w_k;
            end
        end
        useZF = true;
    else
        % MRT fallback
        for k = 1:K
            h_k = H_all(:,k);
            Wcomm(:,k) = sqrt(cfg.sigmaC2) * h_k / norm(h_k);
        end
        useZF = false;
    end
    
    Pcomm_total = sum(Pcomm_per_k);
    
    % Step 4: Sensing beamforming (Matched Filter)
    Wsens = zeros(nActive*Nt, P);
    Psens_per_p = zeros(P, 1);
    
    for p = 1:P
        g_p = G_all(:,p);
        norm_g = norm(g_p);
        if norm_g > 1e-10
            % Minimum power for sensing SINR constraint
            Psens_per_p(p) = gammaS_robust * cfg.sigmaS2 / norm_g^2;
            % Matched filter beam
            Wsens(:,p) = sqrt(Psens_per_p(p)) * g_p / norm_g;
        end
    end
    
    Psens_total = sum(Psens_per_p);
    
    % Step 5: Per-AP power check
    P_per_ap = zeros(nActive, 1);
    for i = 1:nActive
        m = activeAPs(i);
        idx_local = (i-1)*Nt + (1:Nt);
        
        % Communication power at this AP
        Pcomm_ap = 0;
        for k = 1:K
            Pcomm_ap = Pcomm_ap + norm(Wcomm(idx_local, k))^2;
        end
        
        % Sensing power at this AP
        Psens_ap = 0;
        for p = 1:P
            Psens_ap = Psens_ap + norm(Wsens(idx_local, p))^2;
        end
        
        P_per_ap(i) = Pcomm_ap + Psens_ap;
    end
    
    % Check per-AP power constraint
    max_ap_power = max(P_per_ap);
    total_power = sum(P_per_ap);
    
    % Step 6: Verification
    % Communication SINR
    minSinrWc = inf;
    for k = 1:K
        hk = H_all(:,k);
        desired = abs(hk' * Wcomm(:,k))^2;
        interf = cfg.sigmaC2;
        for j = 1:K
            if j ~= k
                interf = interf + abs(hk' * Wcomm(:,j))^2;
            end
        end
        sinr_nom = desired / interf;
        sinr_wc = sinr_nom * etaH;
        minSinrWc = min(minSinrWc, sinr_wc);
    end
    
    % Sensing SINR
    minSnrWc = inf;
    for p = 1:P
        gp = G_all(:,p);
        desired = abs(gp' * Wsens(:,p))^2;
        snr_nom = desired / cfg.sigmaS2;
        snr_wc = snr_nom * etaG;
        minSnrWc = min(minSnrWc, snr_wc);
    end
    
    % PCRB (simplified)
    minTraceJ = inf;
    for p = 1:P
        traceJ = 0;
        for k = 1:K
            for i = 1:nActive
                m = activeAPs(i);
                idx_local = (i-1)*Nt + (1:Nt);
                g_mp = scenario.G((m-1)*Nt + (1:Nt), p);
                w_mk = Wcomm(idx_local, k);
                traceJ = traceJ + abs(g_mp' * w_mk)^2;
            end
        end
        minTraceJ = min(minTraceJ, traceJ);
    end
    
    % Violation calculation
    vSinr = max(0, (cfg.gammaK - minSinrWc) / cfg.gammaK);
    vSnr = max(0, (cfg.gammaS - minSnrWc) / cfg.gammaS);
    vPower = max(0, (max_ap_power - cfg.Pmax) / cfg.Pmax);
    vPcrb = max(0, (cfg.gammaTrack - minTraceJ) / cfg.gammaTrack);
    
    violation = max([vSinr, vSnr, vPower, vPcrb]);
    
    % Result
    result.minSinrWcDb = 10*log10(minSinrWc);
    result.minSnrWcDb = 10*log10(minSnrWc);
    result.totalPower = total_power;
    result.maxApPower = max_ap_power;
    result.violation = violation;
    result.success = (violation <= 1e-6);
    result.useZF = useZF;
    result.Pcomm = Pcomm_total;
    result.Psens = Psens_total;
    result.b_mp = b_mp;
    result.activeAPs = activeAPs;
end
