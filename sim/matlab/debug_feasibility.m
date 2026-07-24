function debug_feasibility()
%DEBUG_FEASIBILITY  Short diagnosis: why is fixed-b re-optimization infeasible?
%   Runs 5 sequential trials under three relaxed configurations and reports
%   feasibility rates for (i) relaxed DC warm-start, (ii) greedy rounded,
%   and (iii) distance-heuristic topologies.

rng(2026);
Base_seed = 2026;
N_mc = 5;

% Base configuration
base.M = 8; base.Nt = 4; base.K = 4; base.P = 2; base.N_theta = 2;
base.Pmax_dBm = 20; base.AreaSize = 400; base.N_req = 2;
base.T_max = 5; base.eps = 1e-5; base.eta_rank = 1.0; base.eta_b = 1.0; base.eta_growth = 1.3;

cases = {
    struct('name', 'A_nominal',        'eps_h', 0.05, 'Gamma_track', 10, 'N_req_extra', 0, 'use_s_proc', true)
    struct('name', 'B_no_robust',      'eps_h', 0.00, 'Gamma_track', 10, 'N_req_extra', 0, 'use_s_proc', false)
    struct('name', 'C_loose_PCRB',     'eps_h', 0.05, 'Gamma_track', 20, 'N_req_extra', 0, 'use_s_proc', true)
    struct('name', 'D_extra_AP',       'eps_h', 0.05, 'Gamma_track', 10, 'N_req_extra', 2, 'use_s_proc', true)
    struct('name', 'E_no_robust_extra','eps_h', 0.00, 'Gamma_track', 10, 'N_req_extra', 2, 'use_s_proc', false)
};

for c = 1:numel(cases)
    cfg = base;
    cfg.eps_h = cases{c}.eps_h;
    cfg.Gamma_track = cases{c}.Gamma_track;
    cfg.use_s_proc = cases{c}.use_s_proc;
    extra = cases{c}.N_req_extra;
    fprintf('\n=== %s: eps_h=%.2f, Gamma=%.1f, extra=%d, robust=%d ===\n', ...
        cases{c}.name, cfg.eps_h, cfg.Gamma_track, extra, cfg.use_s_proc);

    cnt = struct('relaxed_ok',0, 'rounded_ok',0, 'heur_ok',0, 'extra_ok',0, 'total',0, 'relaxed_status',{{}});
    for n = 1:N_mc
        prm = generate_scenario(cfg.M, cfg.Nt, cfg.K, cfg.P, cfg.N_theta, ...
            cfg.Pmax_dBm, cfg.Gamma_track, 'AreaSize', cfg.AreaSize, ...
            'N_req', cfg.N_req, 'eps_h', cfg.eps_h, 'seed', Base_seed + n);
        prm.use_s_procedure = cfg.use_s_proc;

        W0 = cell(cfg.K,1);
        for k = 1:cfg.K
            W0{k} = eye(prm.N) * (prm.Pmax / prm.K / prm.M);
        end
        b0 = ones(prm.M, prm.P) * (prm.N_req / prm.M);

        [W_sdr, ~, ~, b_sdr, ~, status0] = solve_p3_sca_t(prm, W0, b0, 0, 0);
        cnt.relaxed_status{end+1} = status0;
        if contains(status0, 'Solved')
            cnt.relaxed_ok = cnt.relaxed_ok + 1;
        else
            fprintf('  trial %d: initial relaxed=%s; skip rounding\n', n, status0(1:min(12,end)));
            cnt.total = cnt.total + 1;
            continue;
        end

        % Greedy top-N_req
        b_round = greedy_round(b_sdr, prm.N_req, prm.P, prm.active_targets);
        [~,~,~,~,~,status_r] = solve_p3_sca_t(prm, W_sdr, b_round, 0, 0, b_round);
        if contains(status_r, 'Solved'), cnt.rounded_ok = cnt.rounded_ok + 1; end

        % Distance heuristic
        b_heur = nearest_assignment(prm);
        [~,~,~,~,~,status_h] = solve_p3_sca_t(prm, W_sdr, b_heur, 0, 0, b_heur);
        if contains(status_h, 'Solved'), cnt.heur_ok = cnt.heur_ok + 1; end

        % Extra APs
        if extra > 0
            b_extra = greedy_round(b_sdr, prm.N_req + extra, prm.P, prm.active_targets);
            [~,~,~,~,~,status_e] = solve_p3_sca_t(prm, W_sdr, b_extra, 0, 0, b_extra);
            if contains(status_e, 'Solved'), cnt.extra_ok = cnt.extra_ok + 1; end
        end

        cnt.total = cnt.total + 1;
        fprintf('  trial %d: relaxed=%s | rounded=%s | heur=%s', ...
            n, status0(1:min(12,end)), status_r(1:min(12,end)), status_h(1:min(12,end)));
        if extra > 0
            fprintf(' | extra=%s', status_e(1:min(12,end)));
        end
        fprintf('\n');
    end

    fprintf('Summary: relaxed_ok=%d/%d, rounded_ok=%d/%d, heur_ok=%d/%d', ...
        cnt.relaxed_ok, cnt.total, cnt.rounded_ok, cnt.total, cnt.heur_ok, cnt.total);
    if extra > 0
        fprintf(', extra_ok=%d/%d', cnt.extra_ok, cnt.total);
    end
    fprintf('\n');
end

end

function b = greedy_round(b_relaxed, N_req, P, active_targets)
b = zeros(size(b_relaxed));
for p = active_targets
    [~, idx] = sort(b_relaxed(:, p), 'descend');
    b(idx(1:N_req), p) = 1;
end
end

function b = nearest_assignment(prm)
b = zeros(prm.M, prm.P);
for p = prm.active_targets
    d = sqrt(sum((prm.AP_pos - prm.Target_pos(p,:)).^2, 2));
    [~, idx] = sort(d, 'ascend');
    b(idx(1:prm.N_req), p) = 1;
end
end
