% Cell-Free ISAC Complete MATLAB Solver v2 - Optimized AP Selection
%
% Improvements over v1:
% 1. Sensing-aware AP selection: considers beamforming gain, not just channel energy
% 2. Iterative power allocation: adjusts rho based on SINR/SNR margin
% 3. Smart AP pruning: removes APs that don't contribute to sensing beamforming
% 4. Per-target AP grouping: selects APs that jointly serve each target
%
% This script follows docs/COMPLETE_PROBLEM_FORMULATION.md:
% - communication SINR constraints
% - robust SINR/SNR margins under bounded CSI errors
% - sensing SNR constraints
% - CRB constraints
% - total and per-AP power constraints
% - AP selection and communication/sensing power split search

function isac_complete_solver_v2()
    clear; clc; close all;

    cfg = defaultConfig();
    rng(cfg.seed);

    fprintf('============================================================\n');
    fprintf('Cell-Free ISAC Complete MATLAB Solver v2 (Optimized)\n');
    fprintf('============================================================\n');
    fprintf('M=%d, Nt=%d, K=%d, P=%d, Pmax=%.2f W\n', ...
        cfg.M, cfg.Nt, cfg.K, cfg.P, cfg.Pmax);
    fprintf('SINR>=%.2f dB, SNR>=%.2f dB, CRB<=%.2f\n', ...
        cfg.sinrReqDb, cfg.snrReqDb, cfg.crbReq);
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
    
    % Compare with v1 if available
    compareWithV1();
end

function cfg = defaultConfig()
    cfg.seed = 42;

    % System dimensions from the complete v2.2 formulation.
    cfg.M = 16;
    cfg.Nt = 4;
    cfg.K = 10;
    cfg.P = 4;

    % Geometry.
    cfg.apGridSide = 4;
    cfg.apMin = -60;
    cfg.apMax = 60;
    cfg.userMin = -50;
    cfg.userMax = 50;
    cfg.targetMin = -30;
    cfg.targetMax = 30;

    % Channel model.
    cfg.d0 = 10;
    cfg.pathLossExp = 2.5;
    cfg.minDistance = 5;
    cfg.epsilonH = 0.10;
    cfg.epsilonG = 0.15;

    % Noise and constraints.
    cfg.sigmaC2 = 0.5;
    cfg.sigmaS2 = 0.5;
    cfg.Pmax = 30.0;
    cfg.sinrReqDb = 0.0;
    cfg.snrReqDb = 3.0;
    cfg.crbReq = 1.0;

    % AP and power search - expanded candidates for better exploration
    cfg.nActiveCandidates = [4 6 8 10 12 14 16];
    cfg.rhoCandidates = 0.20:0.05:0.95;
    cfg.scoreAlpha = 0.5;
    cfg.mmseRegMargin = 1 + 10 * cfg.epsilonH;

    % Per-AP power cap.
    cfg.PmMax = cfg.Pmax / 4;

    % Monte Carlo.
    cfg.nTrials = 20;
    
    % v2: New parameters for iterative refinement
    cfg.maxIter = 3;           % Iterative refinement rounds
    cfg.snrMarginTarget = 1.5; % Target SNR margin for robustness
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

    scenario.userPos = cfg.userMin + ...
        (cfg.userMax - cfg.userMin) * rand(cfg.K, 2);
    scenario.targetPos = cfg.targetMin + ...
        (cfg.targetMax - cfg.targetMin) * rand(cfg.P, 2);

    scenario.Htrue = zeros(cfg.M, cfg.K, cfg.Nt);
    scenario.Gtrue = zeros(cfg.M, cfg.P, cfg.Nt);

    for m = 1:cfg.M
        for k = 1:cfg.K
            d = max(norm(scenario.apPos(m,:) - scenario.userPos(k,:)), ...
                cfg.minDistance);
            pl = (d / cfg.d0)^(-cfg.pathLossExp);
            scenario.Htrue(m,k,:) = sqrt(pl / 2) * ...
                (randn(cfg.Nt,1) + 1i * randn(cfg.Nt,1));
        end

        for p = 1:cfg.P
            d = max(norm(scenario.apPos(m,:) - scenario.targetPos(p,:)), ...
                cfg.minDistance);
            pl = (d / cfg.d0)^(-cfg.pathLossExp);
            scenario.Gtrue(m,p,:) = sqrt(pl / 2) * ...
                (randn(cfg.Nt,1) + 1i * randn(cfg.Nt,1));
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

    % Strategy 1: Try different AP selection methods
    selectionMethods = {'joint_score', 'sensing_aware', 'communication_first'};
    
    for methodIdx = 1:length(selectionMethods)
        method = selectionMethods{methodIdx};
        
        for nActive = cfg.nActiveCandidates
            % v2: Use sensing-aware selection
            selected = selectApsV2(cfg, scenario, nActive, method);
            
            % Strategy 2: Adaptive rho search based on estimated imbalance
            rhoList = adaptRhoSearch(cfg, scenario, selected);
            
            for rho = rhoList
                Pcomm = rho * cfg.Pmax;
                Psens = (1 - rho) * cfg.Pmax;

                W = robustMmseBeam(cfg, scenario.Hhat(selected,:,:), Pcomm);
                
                % v2: Improved sensing beam with joint design
                Z = jointSensingBeam(cfg, scenario, selected, Psens, W);

                [W, Z] = enforcePerApPower(cfg, W, Z);

                metrics = evaluateSolution(cfg, scenario, selected, W, Z);
                metrics.nActive = nActive;
                metrics.rho = rho;
                metrics.selected = selected;
                metrics.method = method;

                if metrics.violation < best.violation
                    best = metrics;
                end

                if metrics.success
                    % v2: Try to optimize further with iterative refinement
                    refined = iterativeRefine(cfg, scenario, selected, W, Z, rho);
                    if refined.violation < best.violation
                        best = refined;
                    end
                    if best.success && best.violation < 1e-6
                        return; % Early exit if excellent solution found
                    end
                end
            end
        end
    end
end

function selected = selectApsV2(cfg, scenario, nActive, method)
    Hhat = scenario.Hhat;
    Ghat = scenario.Ghat;
    
    switch method
        case 'joint_score'
            % Original method: joint communication + sensing score
            commScore = zeros(cfg.M, 1);
            sensScore = zeros(cfg.M, 1);
            for m = 1:cfg.M
                commScore(m) = sum(abs(reshape(Hhat(m,:,:), [], 1)).^2);
                sensScore(m) = sum(abs(reshape(Ghat(m,:,:), [], 1)).^2);
            end
            commScore = commScore / max(max(commScore), eps);
            sensScore = sensScore / max(max(sensScore), eps);
            score = cfg.scoreAlpha * commScore + (1 - cfg.scoreAlpha) * sensScore;
            
        case 'sensing_aware'
            % v2: Prioritize APs that can form strong sensing beams
            % For each target, find APs with best channel, then union
            selectedPerTarget = cell(cfg.P, 1);
            for p = 1:cfg.P
                gpNorm = zeros(cfg.M, 1);
                for m = 1:cfg.M
                    gp = squeeze(Ghat(m,p,:));
                    gpNorm(m) = norm(gp);
                end
                [~, order] = sort(gpNorm, 'descend');
                % Select top APs for this target
                nPerTarget = max(2, floor(nActive / cfg.P));
                selectedPerTarget{p} = order(1:min(nPerTarget, cfg.M));
            end
            
            % Union of target-specific APs, fill with joint score
            score = zeros(cfg.M, 1);
            for p = 1:cfg.P
                score(selectedPerTarget{p}) = score(selectedPerTarget{p}) + 1;
            end
            
            % Add communication score for remaining slots
            commScore = zeros(cfg.M, 1);
            for m = 1:cfg.M
                commScore(m) = sum(abs(reshape(Hhat(m,:,:), [], 1)).^2);
            end
            commScore = commScore / max(max(commScore), eps);
            score = score + 0.3 * commScore;
            
        case 'communication_first'
            % Ensure communication coverage first, then add sensing
            commScore = zeros(cfg.M, 1);
            for m = 1:cfg.M
                commScore(m) = sum(abs(reshape(Hhat(m,:,:), [], 1)).^2);
            end
            [~, commOrder] = sort(commScore, 'descend');
            
            % Select top half for communication
            nComm = floor(nActive * 0.6);
            selectedComm = commOrder(1:nComm);
            
            % Remaining slots for sensing
            sensScore = zeros(cfg.M, 1);
            for m = 1:cfg.M
                if ~ismember(m, selectedComm)
                    sensScore(m) = sum(abs(reshape(Ghat(m,:,:), [], 1)).^2);
                end
            end
            [~, sensOrder] = sort(sensScore, 'descend');
            selectedSens = sensOrder(1:(nActive - nComm));
            
            selected = [selectedComm; selectedSens];
            return; % Early return for this method
    end
    
    [~, order] = sort(score, 'descend');
    selected = order(1:nActive);
end

function rhoList = adaptRhoSearch(cfg, scenario, selected)
    % v2: Adapt rho range based on estimated communication/sensing difficulty
    Hsel = scenario.Hhat(selected,:,:);
    Gsel = scenario.Ghat(selected,:,:);
    
    % Estimate communication difficulty (condition number)
    Hs = reshape(permute(Hsel, [1 3 2]), [], cfg.K);
    commDiff = cond(Hs' * Hs);
    
    % Estimate sensing difficulty
    gNorms = zeros(cfg.P, 1);
    for p = 1:cfg.P
        gp = reshape(permute(Gsel(:,p,:), [1 3 2]), [], 1);
        gNorms(p) = norm(gp);
    end
    sensDiff = mean(gNorms) / max(gNorms);
    
    % Adjust rho range: more power to the harder task
    if commDiff > 100  % Hard communication
        rhoCenter = 0.65;
    elseif sensDiff < 0.3  % Hard sensing
        rhoCenter = 0.35;
    else
        rhoCenter = 0.50;
    end
    
    % Generate rho candidates around center
    rhoList = max(0.2, min(0.95, rhoCenter + (-0.2:0.05:0.2)));
    rhoList = unique(rhoList);
end

function Z = jointSensingBeam(cfg, scenario, selected, Psens, W)
    % v2: Joint design - sensing beams that don't interfere with communication
    nActive = length(selected);
    Gsel = scenario.Ghat(selected,:,:);
    Z = zeros(nActive, cfg.Nt, cfg.P);
    
    for p = 1:cfg.P
        gp = reshape(permute(Gsel(:,p,:), [1 3 2]), nActive * cfg.Nt, 1);
        
        % Project out communication subspace to reduce interference
        Wf = reshape(W, nActive * cfg.Nt, cfg.K);
        [Q, ~] = qr(Wf, 0);
        gpOrth = gp - Q * (Q' * gp);
        
        % Use orthogonal component if significant
        if norm(gpOrth) > 0.3 * norm(gp)
            zp = gpOrth / norm(gpOrth);
        else
            zp = gp / norm(gp);
        end
        
        zp = zp * sqrt(Psens / cfg.P);
        Z(:,:,p) = reshape(zp, nActive, cfg.Nt);
    end
end

function refined = iterativeRefine(cfg, scenario, selected, W, Z, rho)
    % v2: Iteratively adjust power allocation
    refined = evaluateSolution(cfg, scenario, selected, W, Z);
    refined.nActive = length(selected);
    refined.rho = rho;
    refined.selected = selected;
    
    for iter = 1:cfg.maxIter
        % Check which constraint is tight
        sinrMargin = min(refined.sinrWcDb) - cfg.sinrReqDb;
        snrMargin = min(refined.snrWcDb) - cfg.snrReqDb;
        
        % Adjust rho based on margins
        if snrMargin < sinrMargin && snrMargin < cfg.snrMarginTarget
            % Sensing is bottleneck, reduce rho (more power to sensing)
            rho = max(0.2, rho - 0.1);
        elseif sinrMargin < snrMargin && sinrMargin < 0.5
            % Communication is bottleneck, increase rho
            rho = min(0.95, rho + 0.1);
        else
            break; % Balanced enough
        end
        
        Pcomm = rho * cfg.Pmax;
        Psens = (1 - rho) * cfg.Pmax;
        
        W = robustMmseBeam(cfg, scenario.Hhat(selected,:,:), Pcomm);
        Z = jointSensingBeam(cfg, scenario, selected, Psens, W);
        [W, Z] = enforcePerApPower(cfg, W, Z);
        
        newMetrics = evaluateSolution(cfg, scenario, selected, W, Z);
        if newMetrics.violation < refined.violation
            refined = newMetrics;
            refined.rho = rho;
        else
            break;
        end
    end
end

function [W, Z] = enforcePerApPower(cfg, W, Z)
    nActive = size(W, 1);

    for m = 1:nActive
        pm = sum(abs(reshape(W(m,:,:), [], 1)).^2) + ...
             sum(abs(reshape(Z(m,:,:), [], 1)).^2);

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
    Gsel = scenario.Gtrue(selected,:,:);
    nActive = numel(selected);

    metrics.sinrDb = computeSinrDb(cfg, Hsel, W);
    metrics.sinrWcDb = metrics.sinrDb + ...
        20 * log10((1 - cfg.epsilonH) / (1 + cfg.epsilonH));

    metrics.snrDb = computeSnrDb(cfg, Gsel, Z);
    metrics.snrWcDb = metrics.snrDb + ...
        20 * log10((1 - cfg.epsilonG) / (1 + cfg.epsilonG));

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
        apPower(m) = sum(abs(reshape(W(m,:,:), [], 1)).^2) + ...
                     sum(abs(reshape(Z(m,:,:), [], 1)).^2);
    end
end
