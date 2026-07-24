%% Smoke test for large-scale params (single trial)
clear; clc;
addpath(pwd);

M = 16; Nt = 4; K = 4; P = 3; N_theta = 2;
Pmax_dBm = 20;
Gamma_track = 10;
seed = 2026;

fprintf('Generating scenario...\n');
prm = generate_scenario(M, Nt, K, P, N_theta, Pmax_dBm, Gamma_track, ...
    'AreaSize', 400, 'N_req', 1, 'eps_h', 0.05, ...
    'seed', seed);

fprintf('Running baseline_alg2 with verbose...\n');
tic;
res = baseline_alg2(prm, 80, 1e-5, 1.0, 1.0, 1.3, true);
t_elapsed = toc;

fprintf('\nStatus: %s\n', res.status);
fprintf('Iters: %d\n', res.iters);
fprintf('Final obj: %.4f\n', res.final_obj);
fprintf('Sum rate: %.4f\n', res.sum_rate);
fprintf('Max violation: %.2e\n', res.max_violation);
fprintf('Elapsed: %.1f s\n', t_elapsed);
