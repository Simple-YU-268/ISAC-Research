function debug_nreq_scan()
%DEBUG_NREQ_SCAN  Quick scan N_req=1..4 feasibility after decoupling power gate.

Base_seed = 2027;
N_mc = 10;
M = 8; Nt = 4; K = 4; P = 2; N_theta = 2;
Pmax_dBm = 20; AreaSize = 400; Gamma = 30; eps_h = 0;

for N_req = 1:4
    fprintf('\n=== N_req=%d ===\n', N_req);
    ok_rel = 0; ok_round = 0; ok_heur = 0;
    for n = 1:N_mc
        prm = generate_scenario(M, Nt, K, P, N_theta, Pmax_dBm, Gamma, ...
            'AreaSize', AreaSize, 'N_req', N_req, 'eps_h', eps_h, ...
            'gamma_k_dB', 0, 'seed', Base_seed + n);
        prm.use_s_procedure = false;  % eps_h=0, robust off
        
        W0 = cell(K,1);
        for k = 1:K, W0{k} = eye(prm.N) * (prm.Pmax / prm.K / prm.M); end
        b0 = ones(prm.M, prm.P) * (N_req / prm.M);
        
        [W_sdr,~,~,b_sdr,~,status0] = solve_p3_sca_t(prm, W0, b0, 0, 0);
        if contains(status0, 'Solved'), ok_rel = ok_rel + 1; end
        
        if contains(status0, 'Solved')
            b_round = greedy_round(b_sdr, N_req, prm.P, prm.active_targets);
            [~,~,~,~,~,status_r] = solve_p3_sca_t(prm, W_sdr, b_round, 0, 0, b_round);
            if contains(status_r, 'Solved'), ok_round = ok_round + 1; end
            
            b_heur = nearest_assignment(prm, N_req);
            [~,~,~,~,~,status_h] = solve_p3_sca_t(prm, W_sdr, b_heur, 0, 0, b_heur);
            if contains(status_h, 'Solved'), ok_heur = ok_heur + 1; end
        end
    end
    fprintf('  relaxed: %d/%d, rounded: %d/%d, nearest-heur: %d/%d\n', ...
        ok_rel, N_mc, ok_round, N_mc, ok_heur, N_mc);
end

end

function b = greedy_round(b_relaxed, N_req, P, active_targets)
b = zeros(size(b_relaxed));
for p = active_targets
    [~, idx] = sort(b_relaxed(:, p), 'descend');
    b(idx(1:N_req), p) = 1;
end
end

function b = nearest_assignment(prm, N_req)
b = zeros(prm.M, prm.P);
for p = prm.active_targets
    d = sqrt(sum((prm.AP_pos - prm.Target_pos(p,:)).^2, 2));
    [~, idx] = sort(d, 'ascend');
    b(idx(1:N_req), p) = 1;
end
end
