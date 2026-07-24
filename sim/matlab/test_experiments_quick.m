function test_experiments_quick()
%TEST_EXPERIMENTS_QUICK  Quick smoke test for the full experimental pipeline
%   Verifies generate_scenario + proposed/robust + heuristic AP baseline +
%   non-robust baseline + comm-only + sensing-only in a few minutes.

M = 8; Nt = 4; K = 4; P = 2; N_theta = 2;
Pmax_dBm = 20; Gamma_track = 'auto';  % physical isotropic-reference calibration

fprintf('=== Generate scenario ===\n');
prm = generate_scenario(M, Nt, K, P, N_theta, Pmax_dBm, Gamma_track, ...
    'AreaSize', 400, 'N_req', 3, 'eps_h', 0.05, 'seed', 1, 'noise_snr_target', 1e4);
fprintf('AP positions: %d x %d\n', size(prm.AP_pos,1), size(prm.AP_pos,2));
fprintf('H: %d x %d\n', size(prm.H,1), size(prm.H,2));
fprintf('G: %d x %d\n', size(prm.G,1), size(prm.G,2));
fprintf('D: %d x %d x %d\n', size(prm.D,1), size(prm.D,2), size(prm.D,3));
fprintf('Gamma_track (auto): '); fprintf('%.2f ', prm.Gamma_track); fprintf('\n');
prm.solver = 'mosek';  % MOSEK verified ready via cvx_setup; much faster than SDPT3

fprintf('\n=== Proposed algorithm ===\n');
tic;
res = baseline_alg2(prm, 30, 1e-5, 1.0, 1.0, 1.3, true);
t = toc;
fprintf('Time: %.2f s\n', t);
fprintf('Status: %s\n', res.status);
if ~contains(res.status, 'Solved')
    fprintf('Proposed solution unavailable; stopping quick test.\n');
    return;
end
fprintf('Iters: %d, Final obj: %.4f\n', res.iters, res.final_obj);
fprintf('Sum rate: %.4f, Sensing SINR (dB): ', res.sum_rate);
fprintf('%.2f ', res.sens_sinr_db); fprintf('\n');
fprintf('PCRB trace: '); fprintf('%.4f ', res.pcrb); fprintf('\n');
fprintf('Max violation: %.2e\n', res.max_violation);

fprintf('\n=== Heuristic AP baseline ===\n');
b = heuristic_b(prm);
tic;
res_heur = solve_p3_with_fixed_b(prm, b, 30, 1e-5, 1.0, 1.0, 1.3);
t = toc;
fprintf('Time: %.2f s\n', t);
fprintf('Status: %s\n', res_heur.status);
if contains(res_heur.status, 'Solved')
    fprintf('Iters: %d, Final obj: %.4f\n', res_heur.iters, res_heur.final_obj);
end

fprintf('\n=== Non-robust baseline ===\n');
prm_nr = prm; prm_nr.eps_h = 0;
tic;
res_nr = baseline_alg2(prm_nr, 20, 1e-5, 1.0, 1.0, 1.3, false);
t = toc;
fprintf('Time: %.2f s\n', t);
fprintf('Status: %s\n', res_nr.status);
if contains(res_nr.status, 'Solved')
    fprintf('Iters: %d, Final obj: %.4f\n', res_nr.iters, res_nr.final_obj);
end

fprintf('\n=== Comm-only baseline ===\n');
prm_comm = prm;
prm_comm.enable_sensing_sinr = false;
prm_comm.enable_pcrb = false;
b_comm = heuristic_b(prm_comm);
tic;
res_comm = solve_p3_with_fixed_b(prm_comm, b_comm, 30, 1e-5, 1.0, 1.0, 1.3);
t = toc;
fprintf('Time: %.2f s\n', t);
fprintf('Status: %s\n', res_comm.status);
if contains(res_comm.status, 'Solved')
    fprintf('Iters: %d, Final obj: %.4f\n', res_comm.iters, res_comm.final_obj);
end

fprintf('\n=== Sensing-only baseline ===\n');
prm_sens = prm; prm_sens.gamma_k = 1e-6 * ones(K,1);
b_sens = heuristic_b(prm_sens);
tic;
res_sens = solve_p3_with_fixed_b(prm_sens, b_sens, 30, 1e-5, 1.0, 1.0, 1.3);
t = toc;
fprintf('Time: %.2f s\n', t);
fprintf('Status: %s\n', res_sens.status);
if contains(res_sens.status, 'Solved')
    fprintf('Iters: %d, Final obj: %.4f\n', res_sens.iters, res_sens.final_obj);
end

fprintf('\n=== All smoke tests completed ===\n');

end

function b = heuristic_b(prm)
M = prm.M; P = prm.P;
b = zeros(M, P);
for p = 1:P
    dists = sqrt(sum((prm.AP_pos - prm.Target_pos(p,:)).^2, 2));
    [~, idx] = sort(dists, 'ascend');
    b(idx(1:prm.N_req), p) = 1;
end
end
