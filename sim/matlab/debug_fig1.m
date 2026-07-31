function debug_fig1()
%DEBUG_FIG1  Single trial debug for Figure 1 representative N_req=3.

cfg.M = 8; cfg.Nt = 4; cfg.K = 4; cfg.P = 2; cfg.N_theta = 2;
cfg.Pmax_dBm = 20; cfg.AreaSize = 400; cfg.eps_h = 0.05;
cfg.Gamma_track = 'auto'; cfg.N_req = 3;

prm = generate_scenario(cfg.M, cfg.Nt, cfg.K, cfg.P, cfg.N_theta, ...
    cfg.Pmax_dBm, cfg.Gamma_track, 'AreaSize', cfg.AreaSize, ...
    'N_req', cfg.N_req, 'eps_h', cfg.eps_h, 'seed', 2026);

res = baseline_alg2(prm, 30, 1e-5, 1.0, 1.0, 1.3, true);
fprintf('status: %s\n', res.status);
fprintf('final_obj: %.4f\n', res.final_obj);
end
