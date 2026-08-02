function summary = run_participation_topology_ablation(varargin)
%RUN_PARTICIPATION_TOPOLOGY_ABLATION  Topology versus beamforming validation.
%   Compares oracle nearest-AP and FIM-guided fixed topologies, both followed
%   by identical fixed-b joint beamforming/covariance optimization, against
%   DC-only recovery and the complete finite candidate pool.

ip = inputParser;
addParameter(ip, 'Seeds', 1:10, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'T_max', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'Recovery_max_candidates', 7, @(x) isnumeric(x) && isscalar(x) && x >= 2);
addParameter(ip, 'Output_dir', fullfile(pwd, 'experiment_packages', 'v1.0', ...
    'results', 'participation_topology_ablation'), @(x) ischar(x) || isstring(x));
parse(ip, varargin{:}); opt = ip.Results;
out_dir = char(opt.Output_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

labels = ["fim_fixed_b", "oracle_nearest_fixed_b", "dc_topn", "full_pool"];
records = repmat(empty_record(), numel(opt.Seeds), 1);
for i = 1:numel(opt.Seeds)
    seed = opt.Seeds(i);
    fprintf('\nTopology ablation %d/%d, seed=%d\n', i, numel(opt.Seeds), seed);
    prm = generate_scenario(6, 2, 3, 2, 2, 20, 'auto', ...
        'AreaSize', 400, 'N_req', 3, 'eps_h', 0.05, 'seed', seed);
    prm.solver = 'mosek';
    prm.recovery_mosek_max_time = 30;
    records(i).seed = seed;

    [b_fim, ~] = construct_greedy_fim_assignment(prm, 'doptimal');
    records(i).methods(1) = solve_fixed(prm, b_fim, labels(1));
    records(i).methods(2) = solve_fixed(prm, oracle_nearest_assignment(prm), labels(2));

    prm_dc = prm;
    prm_dc.recovery_max_candidates = 1;
    prm_dc.recovery_stop_first_feasible = true;
    prm_dc.recovery_include_greedy_fim = false;
    prm_dc.recovery_slack_diagnosis = false;
    records(i).methods(3) = solve_proposed(prm_dc, opt.T_max, labels(3));

    prm_full = prm;
    prm_full.recovery_max_candidates = opt.Recovery_max_candidates;
    prm_full.recovery_stop_first_feasible = false;
    prm_full.recovery_include_greedy_fim = true;
    prm_full.recovery_slack_diagnosis = true;
    prm_full.recovery_slack_guided_slots = 2;
    records(i).methods(4) = solve_proposed(prm_full, opt.T_max, labels(4));
    fprintf('  FIM=%d, nearest=%d, DC=%d, full=%d\n', records(i).methods.feasible);
    save(fullfile(out_dir, 'checkpoint.mat'), 'records', 'opt', 'labels');
end

summary.labels = labels;
for m = 1:numel(labels)
    method_records = [records.methods];
    method_records = method_records(m:numel(labels):end);
    feasible = [method_records.feasible];
    power = [method_records(feasible).power_W];
    summary.feasibility_rate(m) = mean(feasible);
    summary.power_mean_W(m) = mean(power, 'omitnan');
    summary.power_median_W(m) = median(power, 'omitnan');
    summary.time_median_s(m) = median([method_records.time_s], 'omitnan');
end
save(fullfile(out_dir, 'final.mat'), 'records', 'summary', 'opt', 'labels');
end

function method = solve_fixed(prm, b, label)
timer = tic;
res = solve_p3_with_fixed_b(prm, b, 2, 1e-5, 1, 0, 1);
method = summarize(res, label, toc(timer));
end

function method = solve_proposed(prm, t_max, label)
timer = tic;
res = baseline_alg2(prm, t_max, 1e-5, 1, 1, 1, false);
method = summarize(res, label, toc(timer));
end

function method = summarize(res, label, elapsed)
method = struct('label', label, 'feasible', false, 'power_W', NaN, ...
    'time_s', elapsed, 'status', string(get_field(res, 'status', 'unknown')), ...
    'candidate_index', get_field(res, 'recovery_candidate_index', NaN), ...
    'candidates_tested', get_field(res, 'recovery_candidates_tested', NaN));
method.feasible = isfield(res, 'is_physical_feasible') && res.is_physical_feasible;
method.power_W = get_field(res, 'final_obj', NaN);
end

function b = oracle_nearest_assignment(prm)
b = zeros(prm.M, prm.P);
for p = prm.active_targets
    distance = sqrt(sum((prm.AP_pos - prm.Target_pos(p,:)).^2, 2));
    [~, order] = sort(distance, 'ascend');
    b(order(1:prm.N_req), p) = 1;
end
end

function value = get_field(s, name, default_value)
if isfield(s, name), value = s.(name); else, value = default_value; end
end

function record = empty_record()
method = struct('label', "", 'feasible', false, 'power_W', NaN, ...
    'time_s', NaN, 'status', "not_run", 'candidate_index', NaN, ...
    'candidates_tested', NaN);
record = struct('seed', NaN, 'methods', repmat(method, 1, 4));
end
