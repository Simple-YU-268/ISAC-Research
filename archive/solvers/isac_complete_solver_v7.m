% Cell-Free ISAC Solver v7 - Full-Rank MIMO Sensing Channel
% Key change: G is now a full-rank MIMO channel matrix

clear; clc; close all;

cfg = defaultConfig();
rng(cfg.seed);

fprintf('============================================================\n');
fprintf('Cell-Free ISAC Solver v7 (Full-Rank MIMO Sensing)\n');
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
    % v7: Full-rank MIMO channel
    cfg.sensingRicianK = 3;  % Rician K-factor for sensing channel
    cfg.nPaths = 4;          % Number of multipath components
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
    % v7: G is now M x P x Nt x Nt (full MIMO matrix per AP-target pair)
    scenario.Gtrue = zeros(cfg.M, cfg.P, cfg.Nt, cfg.Nt);
    for m = 1:cfg.M
        for k = 1:cfg.K
            d = max(norm(scenario.apPos(m,:) - scenario.userPos(k,:)), cfg.minDistance);
            pl = (d / cfg.d0)^(-cfg.pathLossExp);
            scenario.Htrue(m,k,:) = sqrt(pl / 2) * (randn(cfg.Nt,1) + 1i * randn(cfg.Nt,1));
        end
        for p = 1:cfg.P
            d = max(norm(scenario.apPos(m,:) - scenario.targetPos(p,:)), cfg.minDistance);
            pl = (d / cfg.d0)^(-cfg.pathLossExp);
            % Full-rank MIMO channel with LOS + NLOS components
            Gm = zeros(cfg.Nt, cfg.Nt);
            % LOS component (deterministic, based on angle)
            angle_ap_target = atan2(scenario.targetPos(p,2) - scenario.apPos(m,2), ...
                                   scenario.targetPos(p,1) - scenario.apPos(m,1));
            a_los = exp(1i * pi * (0:cfg.Nt-1)' * sin(angle_ap_target));
            G_los = a_los * a_los';
            % NLOS component (random scattering)
            G_nlos = (randn(cfg.Nt) + 1i * randn(cfg.Nt)) / sqrt(2);
            % Rician model
            K = cfg.sensingRicianK;
            Gm = sqrt(pl * K / (K + 1)) * G_los + sqrt(pl / (K + 1)) * G_nlos;
            scenario.Gtrue(m,p,:,:) = Gm;
        end
    end
    scenario.Hhat = addBoundedCsiError(scenario.Htrue, cfg.epsilonH);
    scenario.Ghat = addBoundedCsiErrorG(scenario.Gtrue, cfg.epsilonG, cfg.Nt);
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

function Ghat = addBoundedCsiErrorG(Gtrue, epsilon, Nt)
    Ghat = zeros(size(Gtrue));
    for i = 1:size(Gtrue, 1)
        for j = 1:size(Gtrue, 2)
            Gm = squeeze(Gtrue(i,j,:,:));
            E = randn(Nt) + 1i * randn(Nt);
            if norm(E, 'fro') > 0
                E = E / norm(E, 'fro') * epsilon * max(norm(Gm, 'fro'), eps);
            end
            Ghat(i,j,:,:) = Gm + E;
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
        [success, metrics] = trySolve(cfg, scenario, selected);

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
        % v7: Use nuclear norm for MIMO channel
        for p = 1:cfg.P
            Gm = squeeze(scenario.Ghat(m,p,:,:));
            sensScore(m) = sensScore(m) + sum(svd(Gm));
        end
    end
    commScore = commScore / max(max(commScore), eps);
    sensScore = sensScore / max(max(sensScore), eps);
    scores = cfg.scoreAlpha * commScore + (1 - cfg.scoreAlpha) * sensScore;
end

function [success, metrics] = trySolve(cfg, scenario, selected)
    metrics = emptyResult();
    metrics.violation = Inf;

    for rho = cfg.rhoCandidates
        Pcomm = rho * cfg.Pmax;
        Psens = (1 - rho) * cfg.Pmax;

        W = robustMmseBeam(cfg, scenario.Hhat(selected,:,:), Pcomm);
        Z = mimoSensingBeam(cfg, scenario.Ghat(selected,:,:,:), Psens);
        [W, Z] = enforcePerApPower(cfg, W, Z);

        newMetrics = evaluateSolution(cfg, scenario, selected, W, Z);
        newMetrics.nActive = length(selected);
        newMetrics.rho = rho;
        newMetrics.selected = selected;

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

function W = robustMmseBeam(cfg, Hsel, Pcomm)
    nActive = size(Hsel, 1);
    Hs = reshape(permute(Hsel, [1 3 2]), nActive * cfg.Nt, cfg.K);
    reg = cfg.sigmaC2 * cfg.mmseRegMargin;
    Wflat = (Hs * Hs' + reg * eye(nActive * cfg.Nt)) \ Hs;
    p = sum(abs(Wflat(:)).^2);
    if p > 0
        Wflat = Wflat * sqrt(Pcomm / p);
    end
    W = reshape(Wflat, nActive, cfg.Nt, cfg.K);
end

function Z = mimoSensingBeam(cfg, Gsel, Psens)
    % Gsel: nActive x P x Nt x Nt
    nActive = size(Gsel, 1);
    Z = zeros(nActive, cfg.Nt, cfg.P);
    for p = 1:cfg.P
        % Stack all AP MIMO channels for target p
        G_total = zeros(nActive * cfg.Nt, cfg.Nt);
        for m = 1:nActive
            Gm = squeeze(Gsel(m,p,:,:));
            G_total((m-1)*cfg.Nt+1:m*cfg.Nt, :) = Gm;
        end
        % Dominant eigenmode beamforming
        [U, S, V] = svd(G_total, 'econ');
        if size(S, 1) > 0 && S(1,1) > 0
            % Use dominant right singular vector
            beam = V(:,1);
            % Distribute among APs
            for m = 1:nActive
                Z(m,:,p) = beam' * sqrt(Psens / cfg.P / nActive);
            end
        end
    end
end

function [W, Z] = enforcePerApPower(cfg, W, Z)
    nActive = size(W, 1);
    for m = 1:nActive
        pm = sum(abs(reshape(W(m,:,:), [], 1)).^2) + sum(abs(reshape(Z(m,:,:), [], 1)).^2);
        if pm > cfg.PmMax
            scale = sqrt(cfg.PmMax / pm);
            W(m,:,:) = W(m,:,:) * scale;
            Z(m,:,:) = Z(m,:,:) * scale;
        end
    end
end

function metrics = evaluateSolution(cfg, scenario, selected, W, Z)
    metrics = emptyResult();
    Hsel = scenario.Htrue(selected,:,:);
    Gsel = scenario.Gtrue(selected,:,:,:);
    nActive = numel(selected);

    metrics.sinrDb = computeSinrDb(cfg, Hsel, W);
    metrics.sinrWcDb = metrics.sinrDb + 20 * log10((1 - cfg.epsilonH) / (1 + cfg.epsilonH));
    metrics.snrDb = computeSnrDbMIMO(cfg, Gsel, Z);
    metrics.snrWcDb = metrics.snrDb + 20 * log10((1 - cfg.epsilonG) / (1 + cfg.epsilonG));
    metrics.crb = computeCrbMIMO(cfg, Gsel, W, Z);
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

function snrDb = computeSnrDbMIMO(cfg, Gsel, Z)
    % Gsel: nActive x P x Nt x Nt
    nActive = size(Gsel, 1);
    snrDb = zeros(cfg.P, 1);
    for p = 1:cfg.P
        % Sum received signal from all APs
        y_total = zeros(cfg.Nt, 1);
        tx_power = 0;
        for m = 1:nActive
            Gm = squeeze(Gsel(m,p,:,:));
            zm = Z(m,:,p)';
            y_total = y_total + Gm * zm;
            tx_power = tx_power + norm(zm)^2;
        end
        signal = norm(y_total)^2;
        noise = cfg.sigmaS2 * tx_power;
        snrDb(p) = 10 * log10(signal / noise);
    end
end

function crb = computeCrbMIMO(cfg, Gsel, W, Z)
    nActive = size(Gsel, 1);
    crb = zeros(cfg.P, 1);
    for p = 1:cfg.P
        % Fisher information
        fisher = 0;
        for k = 1:cfg.K
            y_k = zeros(cfg.Nt, 1);
            for m = 1:nActive
                Gm = squeeze(Gsel(m,p,:,:));
                y_k = y_k + Gm' * W(m,:,k)';
            end
            fisher = fisher + norm(y_k)^2;
        end
        y_z = zeros(cfg.Nt, 1);
        for m = 1:nActive
            Gm = squeeze(Gsel(m,p,:,:));
            y_z = y_z + Gm' * Z(m,:,p)';
        end
        fisher = fisher + norm(y_z)^2;
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
    fprintf('Summary v7 (Full-Rank MIMO Sensing)\n');
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

    fprintf('\nv7 Key Feature: Full-rank MIMO sensing channel\n');
    fprintf('  - G is now M x P x Nt x Nt (full matrix)\n');
    fprintf('  - Uses dominant eigenmode beamforming\n');
    fprintf('  - Better spatial diversity exploitation\n');
end
