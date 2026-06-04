% Quick feasibility analysis
clear; rng(42);

% Original config
M = 16; Nt = 4; K = 10; Pmax = 30;
snrReqDb = 3; sinrReqDb = 0;
d0 = 10; plExp = 2.5; sigmaS2 = 0.5;
epsilonG = 0.15;

% Generate AP positions (4x4 grid)
apPos = [];
for x = linspace(-60,60,4)
    for y = linspace(-60,60,4)
        apPos = [apPos; x, y];
    end
end

nTrials = 20;
success = 0;
for t = 1:nTrials
    targetPos = -50 + 100*rand(1,2);  % Original: ±50m
    dists = sqrt(sum((apPos - targetPos).^2, 2));
    pl = (d0./max(dists,5)).^plExp;
    
    % Best case: use all 16 APs, rho=0.5, equal power
    Psens_total = 15;  % rho=0.5
    g_total = sum(sqrt(pl));
    snr_lin = (g_total * sqrt(Psens_total/M))^2 * M / sigmaS2;
    snr_wc = snr_lin * (1-epsilonG)/(1+epsilonG);
    snr_db = 10*log10(snr_wc);
    
    if snr_db >= snrReqDb
        success = success + 1;
    end
    
    status = 'X';
    if snr_db >= snrReqDb
        status = 'OK';
    end
    fprintf('Trial %02d: target=[%5.1f,%5.1f], minDist=%4.1f, SNR=%5.1fdB %s\n', ...
        t, targetPos(1), targetPos(2), min(dists), snr_db, status);
end

fprintf('\nSuccess rate: %d/%d = %.0f%%\n', success, nTrials, success/nTrials*100);
