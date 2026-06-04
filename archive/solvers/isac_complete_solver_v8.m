% Cell-Free ISAC Solver v8 - Local Power Allocation
% Each AP independently allocates power between comm and sensing

clear; clc; close all;

cfg = defaultConfig();
rng(cfg.seed);

fprintf('============================================================\n');
fprintf('Cell-Free ISAC Solver v8 (Local Power Allocation)\n');
fprintf('============================================================\n');
fprintf('M=%d, Nt=%d, K=%d, P=%d, Pmax=%.2f W\n', cfg.M, cfg.Nt, cfg.K, cfg.P, cfg.Pmax);
fprintf('SINR>=%.2f dB, SNR>=%.2f dB, CRB<=%.2f\n', cfg.sinrReqDb, cfg.snrReqDb, cfg.crbReq);
fprintf('Trials=%d\n\n', cfg.nTrials);

trialResults = repmat(emptyResult(), cfg.nTrials, 1);

for trial = 1:cfg.nTrials
    scenario = generateScenario(cfg);
    trialResults(trial) = solveOneScenario(cfg, scenario);

    fprintf(['Trial %02d/%02d | ok=%d | Nactive=%2d | rho=%.2f | ' ...
        'SINRwc=%.2f dB | SNRwc=%.2f dB | CRBmax=%.3f | P=%.2f W | violation=%.3f\n'], ...
        trial, cfg.nTrials, trialResults(trial).success, ...
        trialResults(trial).nActive, trialResults(trial).rho, ...
        trialResults(trial).minSinrWcDb, trialResults(trial).minSnrWcDb, ...
        trialResults(trial).maxCrb, trialResults(trial).totalPower, ...
        trialResults(trial).violation);
end

summarizeResults(cfg, trialResults);

function cfg = defaultConfig()
    cfg.seed = 42;
    cfg.M = 16; cfg.Nt = 4; cfg.K = 10; cfg.P = 4;
    cfg.apGridSide = 4;
    cfg.apMin = -60; cfg.apMax = 60;
    cfg.userMin = -50; cfg.userMax = 50;
    cfg.targetMin = -30; cfg.targetMax = 30;
    cfg.d0 = 10; cfg.pathLossExp = 2.5; cfg.minDistance = 5;
    cfg.epsilonH = 0.10; cfg.epsilonG = 0.15;
    cfg.sigmaC2 = 0.5; cfg.sigmaS2 = 0.5;
    cfg.Pmax = 30.0;
    cfg.sinrReqDb = 0.0;
    cfg.snrReqDb = 3.0;
    cfg.crbReq = 1.0;
    cfg.rhoCandidates = 0.20:0.05:0.95;
    cfg.scoreAlpha = 0.5;
    cfg.mmseRegMargin = 1 + 10 * cfg.epsilonH;
    cfg.PmMax = cfg.Pmax / 4;
    cfg.nTrials = 20;
end

function result = emptyResult()
    result.success = false;
    result.nActive = 0;
    result.rho = NaN;
    result.selected = [];
    result.minSinrDb = -Inf;
    result.minSinrWcDb = -Inf;
    result.minSnrDb = -Inf;
    result.minSnrWcDb = -Inf;
    result.maxCrb = Inf;
    result.totalPower = Inf;
    result.maxApPower = Inf;
    result.violation = Inf;
    result.sinrDb = [];
    result.sinrWcDb = [];
    result.snrDb = [];
    result.snrWcDb = [];
    result.crb = [];
    result.apPower = [];
end

function scenario = generateScenario(cfg)
    x = linspace(cfg.apMin, cfg.apMax, cfg.apGridSide);
    y = linspace(cfg.apMin, cfg.apMax, cfg.apGridSide);
    [X, Y] = meshgrid(x, y);
    scenario.apPos = [X(:), Y(:)];
    scenario.userPos = cfg.userMin + (cfg.userMax - cfg.userMin) * rand(cfg.K, 2);
    scenario.targetPos = cfg.targetMin + (cfg.targetMax - cfg.targetMin) * rand(cfg.P, 2);
    scenario.Htrue = zeros(cfg.M, cfg.K, cfg.Nt);
    scenario.Gtrue = zeros(cfg.M, cfg.P, cfg.Nt);
    for m = 1:cfg.M
        for k = 1:cfg.K
            d = max(norm(scenario.apPos(m,:) - scenario.userPos(k,:)), cfg.minDistance);
            pl = (d / cfg.d0)^(-cfg.pathLossExp);
            scenario.Htrue(m,k,:) = sqrt(pl / 2) * (randn(cfg.Nt,1) + 1i * randn(cfg.Nt,1));
        end
        for p = 1:cfg.P
            d = max(norm(scenario.apPos(m,:) - scenario.targetPos(p,:)), cfg.minDistance);
            pl = (d / cfg.d0)^(-cfg.pathLossExp);
            scenario.Gtrue(m,p,:) = sqrt(pl / 2) * (randn(cfg.Nt,1) + 1i * randn(cfg.Nt,1));
        end
    end
    scenario.Hhat = addBoundedCsiError(scenario.Htrue, cfg.epsilonH);
    scenario.Ghat = addBoundedCsiError(scenario.Gtrue, cfg.epsilonG);
end

function Hhat = addBoundedCsiError(Htrue, epsilon)
    Hhat = zeros(size(Htrue));
    for i = 1:size(Htrue, 1)
        for j = 1:size(Htrue, 2)
            h = squeeze(Htrue(i,j,:));
            e = randn(size(h)) + 1i * randn(size(h));
            if norm(e) > 0
                e = e / norm(e) * epsilon * max(norm(h), eps);
            end
            Hhat(i,j,:) = h + e;
        end
    end
end

function best = solveOneScenario(cfg, scenario)
    best = emptyResult();
    best.violation = Inf;

    scores = calculateApScores(cfg, scenario);
    [~, order] = sort(scores, 'descend');

    for nActive = [16 14 12 10 8 6 4]
        selected = order(1:nActive);
        [success, metrics] = trySolveLocal(cfg, scenario, selected);

        if success
            best = metrics;
            return;
        elseif metrics.violation < best.violation
            best = metrics;
        end
    end
end

function scores = calculateApScores(cfg, scenario)
    commScore = zeros(cfg.M, 1);
    sensScore = zeros(cfg.M, 1);
    for m = 1:cfg.M
        commScore(m) = sum(abs(reshape(scenario.Hhat(m,:,:), [], 1)).^2);
        sensScore(m) = sum(abs(reshape(scenario.Ghat(m,:,:), [], 1)).^2);
    end
    commScore = commScore / max(max(commScore), eps);
    sensScore = sensScore / max(max(sensScore), eps);
    scores = cfg.scoreAlpha * commScore + (1 - cfg.scoreAlpha) * sensScore;
end

function [success, metrics] = trySolveLocal(cfg, scenario, selected)
    metrics = emptyResult();
    metrics.violation = Inf;
    nActive = length(selected);

    % Calculate per-AP channel quality
    apCommQuality = zeros(nActive, 1);
    apSensQuality = zeros(nActive, 1);
    for idx = 1:nActive
        m = selected(idx);
        apCommQuality(idx) = sum(abs(reshape(scenario.Hhat(m,:,:), [], 1)).^2);
        apSensQuality(idx) = sum(abs(reshape(scenario.Ghat(m,:,:), [], 1)).^2);
    end
    apCommQuality = apCommQuality / max(apCommQuality);
    apSensQuality = apSensQuality / max(apSensQuality);

    % Try different global rho and local adjustments
    for rho_global = cfg.rhoCandidates
        % Local rho: APs with better sensing channels allocate more to sensing
        rho_local = rho_global * ones(nActive, 1);
        
        % Adjust based on local channel quality
        for m = 1:nActive
            if apSensQuality(m) > 0.7 && apCommQuality(m) < 0.5
                % Strong sensing, weak comm: increase sensing power
                rho_local(m) = max(0.1, rho_global - 0.15);
            elseif apCommQuality(m) > 0.7 && apSensQuality(m) < 0.5
                % Strong comm, weak sensing: increase comm power
                rho_local(m) = min(0.95, rho_global + 0.15);
            end
        end
        
        [newMetrics, ~, ~] = evaluateWithLocalPower(cfg, scenario, selected, rho_local);
        if newMetrics.violation < metrics.violation
            metrics = newMetrics;
        end
        if newMetrics.success
            success = true;
            return;
        end
    end

    success = false;
end

function [metrics, W, Z] = evaluateWithLocalPower(cfg, scenario, selected, rho_local)
    nActive = length(selected);
    W = zeros(nActive, cfg.Nt, cfg.K);
    Z = zeros(nActive, cfg.Nt, cfg.P);

    % Per-AP power allocation
    for idx = 1:nActive
        m = selected(idx);
        Pm = cfg.PmMax;  % Each AP uses max power
        Pcomm_m = rho_local(idx) * Pm;
        Psens_m = (1 - rho_local(idx)) * Pm;
        
        % Local MMSE beam for this AP
        Hm = squeeze(scenario.Hhat(m,:,:));  % K x Nt
        if Pcomm_m > 1e-6
            reg = cfg.sigmaC2 * cfg.mmseRegMargin;
            Wm = (Hm' * Hm + reg * eye(cfg.Nt)) \ Hm';
            Wm = Wm';  % K x Nt
            p = sum(abs(Wm(:)).^2);
            if p > 0
                Wm = Wm * sqrt(Pcomm_m / p);
            end
            W(idx,:,:) = Wm';  % Nt x K -> stored as 1 x Nt x K
        end
        
        % Local matched filtering for sensing
        if Psens_m > 1e-6
            Gm = squeeze(scenario.Ghat(m,:,:));  % P x Nt
            for p = 1:cfg.P
                gp = Gm(p,:);
                if norm(gp) > 0
                    zp = gp' / norm(gp) * sqrt(Psens_m / cfg.P);
                    Z(idx,:,p) = zp';
                end
            end
        end
    end

    metrics = evaluateSolution(cfg, scenario, selected, W, Z);
    metrics.nActive = nActive;
    metrics.rho = mean(rho_local);
    metrics.selected = selected;
end

function metrics = evaluateSolution(cfg, scenario, selected, W, Z)
    metrics = emptyResult();
    Hsel = scenario.Htrue(selected,:,:);
    Gsel = scenario.Gtrue(selected,:,:);
    nActive = numel(selected);

    metrics.sinrDb = computeSinrDb(cfg, Hsel, W);
    metrics.sinrWcDb = metrics.sinrDb + 20 * log10((1 - cfg.epsilonH) / (1 + cfg.epsilonH));
    metrics.snrDb = computeSnrDb(cfg, Gsel, Z);
    metrics.snrWcDb = metrics.snrDb + 20 * log10((1 - cfg.epsilonG) / (1 + cfg.epsilonG));
    metrics.crb = computeCrb(cfg, Gsel, W, Z);
    metrics.apPower = computeApPower(W, Z);

    metrics.minSinrDb = min(metrics.sinrDb);
    metrics.minSinrWcDb = min(metrics.sinrWcDb);
    metrics.minSnrDb = min(metrics.snrDb);
    metrics.minSnrWcDb = min(metrics.snrWcDb);
    metrics.maxCrb = max(metrics.crb);
    metrics.totalPower = sum(metrics.apPower);
    metrics.maxApPower = max(metrics.apPower);

    metrics.violation = computeViolation(cfg, metrics);
    metrics.success = metrics.violation <= 1e-9 && nActive <= cfg.M;
end

function sinrDb = computeSinrDb(cfg, Hsel, W)
    nActive = size(Hsel, 1);
    Hs = reshape(permute(Hsel, [1 3 2]), nActive * cfg.Nt, cfg.K);
    Wf = reshape(W, nActive * cfg.Nt, cfg.K);
    sinrDb = zeros(cfg.K, 1);
    for k = 1:cfg.K
        sig = abs(Wf(:,k)' * Hs(:,k))^2;
        inter = 0;
        for j = 1:cfg.K
            if j ~= k
                inter = inter + abs(Wf(:,j)' * Hs(:,k))^2;
            end
        end
        sinrDb(k) = 10 * log10(sig / (inter + cfg.sigmaC2 + eps));
    end
end

function snrDb = computeSnrDb(cfg, Gsel, Z)
    nActive = size(Gsel, 1);
    snrDb = zeros(cfg.P, 1);
    for p = 1:cfg.P
        gp = reshape(permute(Gsel(:,p,:), [1 3 2]), nActive * cfg.Nt, 1);
        zp = reshape(Z(:,:,p), nActive * cfg.Nt, 1);
        signal = abs(gp' * zp)^2;
        noise = cfg.sigmaS2 * max(sum(abs(zp).^2), eps);
        snrDb(p) = 10 * log10(signal / noise);
    end
end

function crb = computeCrb(cfg, Gsel, W, Z)
    nActive = size(Gsel, 1);
    Wf = reshape(W, nActive * cfg.Nt, cfg.K);
    crb = zeros(cfg.P, 1);
    for p = 1:cfg.P
        gp = reshape(permute(Gsel(:,p,:), [1 3 2]), nActive * cfg.Nt, 1);
        fisher = 0;
        for k = 1:cfg.K
            fisher = fisher + abs(gp' * Wf(:,k))^2;
        end
        zp = reshape(Z(:,:,p), nActive * cfg.Nt, 1);
        fisher = fisher + abs(gp' * zp)^2;
        crb(p) = cfg.sigmaS2 / max(fisher, eps);
    end
end

function apPower = computeApPower(W, Z)
    nActive = size(W, 1);
    apPower = zeros(nActive, 1);
    for m = 1:nActive
        apPower(m) = sum(abs(reshape(W(m,:,:), [], 1)).^2) + sum(abs(reshape(Z(m,:,:), [], 1)).^2);
    end
end

function violation = computeViolation(cfg, metrics)
    violation = 0;
    violation = violation + sum(max(0, cfg.sinrReqDb - metrics.sinrWcDb));
    violation = violation + sum(max(0, cfg.snrReqDb - metrics.snrWcDb));
    violation = violation + sum(max(0, metrics.crb - cfg.crbReq));
    violation = violation + max(0, metrics.totalPower - cfg.Pmax);
    violation = violation + sum(max(0, metrics.apPower - cfg.PmMax));
end

function summarizeResults(cfg, results)
    success = [results.success]';
    minSinrWc = [results.minSinrWcDb]';
    minSnrWc = [results.minSnrWcDb]';
    maxCrb = [results.maxCrb]';
    power = [results.totalPower]';
    violation = [results.violation]';
    nActive = [results.nActive]';

    fprintf('\n============================================================\n');
    fprintf('Summary v8 (Local Power Allocation)\n');
    fprintf('============================================================\n');
    fprintf('Success rate: %d/%d = %.1f%%\n', sum(success), cfg.nTrials, 100 * mean(success));
    fprintf('Mean robust min SINR: %.2f dB\n', mean(minSinrWc));
    fprintf('Mean robust min SNR:  %.2f dB\n', mean(minSnrWc));
    fprintf('Mean max CRB:         %.3f\n', mean(maxCrb));
    fprintf('Mean total power:     %.2f W\n', mean(power));
    fprintf('Mean active APs:      %.1f\n', mean(nActive));
    fprintf('Mean violation:       %.3f\n', mean(violation));

    if any(~success)
        [~, idx] = max(violation);
        fprintf('\nWorst trial diagnostics:\n');
        fprintf('  trial=%d\n', idx);
        fprintf('  robust min SINR=%.2f dB, required %.2f dB\n', results(idx).minSinrWcDb, cfg.sinrReqDb);
        fprintf('  robust min SNR=%.2f dB, required %.2f dB\n', results(idx).minSnrWcDb, cfg.snrReqDb);
        fprintf('  max CRB=%.3f, required <= %.3f\n', results(idx).maxCrb, cfg.crbReq);
        fprintf('  total power=%.2f W, required <= %.2f W\n', results(idx).totalPower, cfg.Pmax);
        fprintf('  max AP power=%.2f W, required <= %.2f W\n', results(idx).maxApPower, cfg.PmMax);
    end

    fprintf('\nv8 Key Feature: Local power allocation per AP\n');
    fprintf('  - Each AP independently allocates power\n');
    fprintf('  - Adapts to local channel conditions\n');
end
