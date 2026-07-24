function debug_feasibility_fast()
%DEBUG_FEASIBILITY_FAST  Quick check: is the initial relaxed problem feasible?
%   Scans many parameter combinations and reports relaxed feasibility rate.

rng(2026);
Base_seed = 2026;
N_mc = 10;

M = 8; Nt = 4; K = 4; P = 2; N_theta = 2;
Pmax_dBm = 20; AreaSize = 400; N_req = 2;

eps_list = [0, 0.02, 0.05, 0.08];
gamma_list = [5, 10, 20, 50];

fprintf('Scanning %d scenarios, %d trials each...\n', numel(eps_list)*numel(gamma_list), N_mc);
for eps_h = eps_list
    for Gamma = gamma_list
        ok = 0; use_s_proc = (eps_h > 0);
        for n = 1:N_mc
            prm = generate_scenario(M, Nt, K, P, N_theta, Pmax_dBm, Gamma, ...
                'AreaSize', AreaSize, 'N_req', N_req, 'eps_h', eps_h, 'seed', Base_seed + n);
            prm.use_s_procedure = use_s_proc;
            W0 = cell(K,1);
            for k = 1:K, W0{k} = eye(prm.N) * (prm.Pmax / prm.K / prm.M); end
            b0 = ones(prm.M, prm.P) * (prm.N_req / prm.M);
            [~,~,~,~,~,status] = solve_p3_sca_t(prm, W0, b0, 0, 0);
            if contains(status, 'Solved'), ok = ok + 1; end
        end
        fprintf('eps_h=%.2f, Gamma=%.2f, robust=%d: relaxed feasible %d/%d\n', ...
            eps_h, Gamma, use_s_proc, ok, N_mc);
    end
end
end
