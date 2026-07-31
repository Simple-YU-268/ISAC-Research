function debug_speed()
%DEBUG_SPEED  Time single solve_p3_sca_t call.

cfg.M = 8; cfg.Nt = 4; cfg.K = 4; cfg.P = 2; cfg.N_theta = 2;
cfg.Pmax_dBm = 20; cfg.AreaSize = 400; cfg.eps_h = 0.05;
cfg.Gamma_track = 'auto'; cfg.N_req = 3;

prm = generate_scenario(cfg.M, cfg.Nt, cfg.K, cfg.P, cfg.N_theta, ...
    cfg.Pmax_dBm, cfg.Gamma_track, 'AreaSize', cfg.AreaSize, ...
    'N_req', cfg.N_req, 'eps_h', cfg.eps_h, 'seed', 2026);

K = prm.K; P = prm.P; N = prm.N; M = prm.M;
W_init = cell(K,1);
for k = 1:K
    W_init{k} = eye(N) * (prm.Pmax / prm.K / prm.M);
end
b0 = ones(M,P) * (prm.N_req / M);

fprintf('=== Timing solve_p3_sca_t (relaxed, eta=0) ===\n');
tic;
[W, Z, mu, b, M_p, status, S_p] = solve_p3_sca_t(prm, W_init, b0, 0, 0);
t1 = toc;
fprintf('status: %s, time: %.2f s\n', status, t1);

fprintf('=== Timing solve_p3_sca_t (fixed b, eta=0) ===\n');
b_fixed = round(b);
tic;
[W2, Z2, mu2, b2, M_p2, status2, S_p2] = solve_p3_sca_t(prm, W_init, b_fixed, 0, 0, b_fixed);
t2 = toc;
fprintf('status: %s, time: %.2f s\n', status2, t2);
end
