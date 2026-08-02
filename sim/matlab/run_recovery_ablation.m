function summary = run_recovery_ablation(varargin)
%RUN_RECOVERY_ABLATION  Compare FIM, DC Top-N, and full recovery.

ip = inputParser;
addParameter(ip, 'Seeds', 1:30, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'T_max', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'Output_dir', fullfile(pwd, '..', '..', 'experiment_packages', ...
    'v1.0', 'results', 'recovery_ablation'), @(x) ischar(x) || isstring(x));
parse(ip, varargin{:}); opt = ip.Results;
out_dir = char(opt.Output_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
diary(fullfile(out_dir, 'ablation.log')); cleanup_diary = onCleanup(@() diary('off'));

labels = ["fim_only", "dc_topn", "full_recovery"];
records = repmat(empty_record(), numel(opt.Seeds), 1);
for i = 1:numel(opt.Seeds)
    seed = opt.Seeds(i);
    fprintf('\nAblation %d/%d, seed=%d\n', i, numel(opt.Seeds), seed);
    prm = generate_scenario(6, 2, 3, 2, 2, 20, 'auto', ...
        'AreaSize', 400, 'N_req', 3, 'eps_h', 0.05, 'seed', seed);
    prm.solver = 'mosek';
    prm.recovery_mosek_max_time = 30;
    [records(i).sdr_status, records(i).sdr_power_W] = solve_sdr(prm);
    records(i).seed = seed;

    [b_fim, ~] = construct_greedy_fim_assignment(prm, 'doptimal');
    records(i).methods(1) = run_fixed_method(prm, b_fim, labels(1), ...
        records(i).sdr_power_W);

    prm_dc = prm;
    prm_dc.recovery_max_candidates = 1;
    prm_dc.recovery_stop_first_feasible = true;
    prm_dc.recovery_include_greedy_fim = false;
    prm_dc.recovery_slack_diagnosis = false;
    records(i).methods(2) = run_baseline_method(prm_dc, opt.T_max, labels(2), ...
        records(i).sdr_power_W);

    prm_full = prm;
    prm_full.recovery_max_candidates = 3;
    prm_full.recovery_stop_first_feasible = false;
    prm_full.recovery_include_greedy_fim = true;
    prm_full.recovery_slack_diagnosis = true;
    prm_full.recovery_slack_guided_slots = 1;
    records(i).methods(3) = run_baseline_method(prm_full, opt.T_max, labels(3), ...
        records(i).sdr_power_W);
    fprintf('  FIM=%d, DC=%d, Full=%d\n', records(i).methods.feasible);
    save(fullfile(out_dir, 'ablation_checkpoint.mat'), 'records', 'opt', 'labels');
end
summary.labels = labels;
for m = 1:numel(labels)
    method_cells = arrayfun(@(r) r.methods(m), records, 'UniformOutput', false);
    method_records = [method_cells{:}];
    feasible = [method_records.feasible];
    summary.feasibility_rate(m) = mean(feasible);
    gaps = [method_records(feasible).power_penalty_pct];
    times = [method_records.time_s];
    summary.power_gap_mean_pct(m) = mean(gaps, 'omitnan');
    summary.power_gap_median_pct(m) = median(gaps, 'omitnan');
    summary.time_median_s(m) = median(times, 'omitnan');
    summary.time_p90_s(m) = prctile(times, 90);
end
save(fullfile(out_dir, 'ablation_final.mat'), 'records', 'summary', 'opt', 'labels');
end

function method = run_fixed_method(prm, b, label, p_sdr)
timer = tic;
res = solve_p3_with_fixed_b(prm, b, 1, 1e-5, 1, 0, 1);
if isfield(res, 'solver_status') && contains(res.solver_status, 'Solved') && ...
        isfield(res, 'is_physical_feasible') && ~res.is_physical_feasible
    res = solve_p3_with_fixed_b(prm, b, 2, 1e-5, 1, 0, 1);
end
method = summarize(label, res, toc(timer), p_sdr);
end

function method = run_baseline_method(prm, tmax, label, p_sdr)
timer = tic;
res = baseline_alg2(prm, tmax, 1e-5, 1, 1, 1.0, false);
method = summarize(label, res, toc(timer), p_sdr);
end

function method = summarize(label, res, elapsed, p_sdr)
method.label = label;
method.time_s = elapsed;
method.feasible = isfield(res, 'is_physical_feasible') && res.is_physical_feasible;
method.power_W = get_field(res, 'final_obj', NaN);
method.status = string(get_field(res, 'status', 'unknown'));
method.candidates_tested = get_field(res, 'recovery_candidates_tested', NaN);
method.power_penalty_pct = NaN;
if method.feasible && isfinite(p_sdr)
    method.power_penalty_pct = 100 * (method.power_W - p_sdr) / p_sdr;
end
end

function [status, power] = solve_sdr(prm)
W0 = cell(prm.K,1);
for k = 1:prm.K, W0{k} = eye(prm.N) * prm.Pmax / (prm.K * prm.M); end
b0 = ones(prm.M,prm.P) * prm.N_req / prm.M;
[W, Z, ~, ~, ~, status] = solve_p3_sca_t(prm, W0, b0, 0, 0);
power = NaN;
if contains(status, 'Solved')
    power = sum(cellfun(@(X) real(trace(X)), W)) + real(trace(Z));
end
end

function value = get_field(s, name, default_value)
if isfield(s, name), value = s.(name); else, value = default_value; end
end

function record = empty_record()
method = struct('label', "", 'time_s', NaN, 'feasible', false, 'power_W', NaN, ...
    'status', "not_run", 'candidates_tested', NaN, 'power_penalty_pct', NaN);
record = struct('seed', NaN, 'sdr_status', "not_run", 'sdr_power_W', NaN, ...
    'methods', repmat(method, 1, 3));
end
