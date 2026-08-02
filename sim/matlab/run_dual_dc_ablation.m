function summary = run_dual_dc_ablation(varargin)
%RUN_DUAL_DC_ABLATION  Attribute the effects of rank and binary DC penalties.
%   FIM and local-swap candidates are disabled. Each mode uses the same
%   top-N projection and one fixed-b recovery attempt, so differences arise
%   only from the continuous DC phase.

ip = inputParser;
addParameter(ip, 'Seeds', 1:10, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'T_dc', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'Output_dir', fullfile(pwd, 'experiment_packages', 'v1.0', ...
    'results', 'participation_dual_dc_ablation'), @(x) ischar(x) || isstring(x));
parse(ip, varargin{:}); opt = ip.Results;
out_dir = char(opt.Output_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

modes = struct('label', {"sdr_relaxation", "rank_dc_only", "binary_dc_only", "dual_dc"}, ...
    'eta_rank', {0, 1, 0, 1}, 'eta_b', {0, 0, 1, 1}, ...
    'iterations', {1, opt.T_dc, opt.T_dc, opt.T_dc});
records = repmat(empty_record(numel(modes)), numel(opt.Seeds), 1);
for i = 1:numel(opt.Seeds)
    seed = opt.Seeds(i);
    fprintf('\nDual-DC ablation %d/%d, seed=%d\n', i, numel(opt.Seeds), seed);
    prm = generate_scenario(6, 2, 3, 2, 2, 20, 'auto', ...
        'AreaSize', 400, 'N_req', 3, 'eps_h', 0.05, 'seed', seed);
    prm.solver = 'mosek'; prm.recovery_mosek_max_time = 30;
    prm.recovery_max_candidates = 1;
    prm.recovery_stop_first_feasible = true;
    prm.recovery_include_greedy_fim = false;
    prm.recovery_slack_diagnosis = false;
    prm.enable_topology_early_stop = false;
    records(i).seed = seed;
    [records(i).sdr_status, records(i).sdr_power_W] = solve_sdr(prm);
    prm.sdr_power_for_ablation = records(i).sdr_power_W;
    for q = 1:numel(modes)
        records(i).methods(q) = run_mode(prm, modes(q));
    end
    fprintf('  feasible: '); fprintf('%d ', [records(i).methods.feasible]); fprintf('\n');
    save(fullfile(out_dir, 'checkpoint.mat'), 'records', 'modes', 'opt');
end

summary.labels = string({modes.label});
for q = 1:numel(modes)
    x = [records.methods]; x = x(q:numel(modes):end);
    f = [x.feasible];
    summary.feasibility_rate(q) = mean(f);
    summary.rank_residual_median(q) = median([x.rank_residual], 'omitnan');
    summary.binary_distance_median(q) = median([x.binary_distance], 'omitnan');
    summary.iterations_median(q) = median([x.iterations], 'omitnan');
    summary.runtime_median_s(q) = median([x.runtime_s], 'omitnan');
    summary.power_gap_median_pct(q) = median([x(f).power_gap_pct], 'omitnan');
end
save(fullfile(out_dir, 'final.mat'), 'records', 'summary', 'modes', 'opt');
end

function method = run_mode(prm, mode)
timer = tic;
res = baseline_alg2(prm, mode.iterations, 1e-5, mode.eta_rank, mode.eta_b, 1, false, mode.iterations);
method = struct('label', string(mode.label), 'feasible', false, 'power_W', NaN, ...
    'power_gap_pct', NaN, 'runtime_s', toc(timer), 'iterations', NaN, ...
    'rank_residual', NaN, 'binary_distance', NaN, 'status', string(get_field(res, 'status', 'unknown')));
method.feasible = isfield(res, 'is_physical_feasible') && res.is_physical_feasible;
method.power_W = get_field(res, 'final_obj', NaN);
method.iterations = get_field(res, 'dc_iterations', NaN);
method.rank_residual = get_field(res, 'dc_rank_deficiency', NaN);
method.binary_distance = get_field(res, 'dc_binary_distance', NaN);
if method.feasible && isfield(prm, 'sdr_power_for_ablation')
    method.power_gap_pct = 100 * (method.power_W - prm.sdr_power_for_ablation) / prm.sdr_power_for_ablation;
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

function record = empty_record(n_methods)
method = struct('label', "", 'feasible', false, 'power_W', NaN, ...
    'power_gap_pct', NaN, 'runtime_s', NaN, 'iterations', NaN, ...
    'rank_residual', NaN, 'binary_distance', NaN, 'status', "not_run");
record = struct('seed', NaN, 'sdr_status', "not_run", 'sdr_power_W', NaN, ...
    'methods', repmat(method, 1, n_methods));
end
