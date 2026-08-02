function summary = run_cooperative_sensing_ablation(varargin)
%RUN_COOPERATIVE_SENSING_ABLATION  Isolate cooperative sensing covariance gain.
%   Both methods use the same FIM-selected binary topology and jointly
%   optimize communication beams. The ablation prohibits cross-AP entries of
%   each S_p, representing noncoherent independently generated sensing signals.

ip = inputParser;
addParameter(ip, 'Seeds', 1:10, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'Output_dir', fullfile(pwd, 'experiment_packages', 'v1.0', ...
    'results', 'participation_cooperative_sensing_ablation'), @(x) ischar(x) || isstring(x));
parse(ip, varargin{:}); opt = ip.Results;
out_dir = char(opt.Output_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

labels = ["proposed_cooperative", "noncoherent_block_diagonal"];
records = repmat(empty_record(), numel(opt.Seeds), 1);
for i = 1:numel(opt.Seeds)
    seed = opt.Seeds(i);
    fprintf('\nCooperative sensing ablation %d/%d, seed=%d\n', i, numel(opt.Seeds), seed);
    prm = generate_scenario(6, 2, 3, 2, 2, 20, 'auto', ...
        'AreaSize', 400, 'N_req', 3, 'eps_h', 0.05, 'seed', seed);
    prm.solver = 'mosek'; prm.recovery_mosek_max_time = 30;
    [b_fim, ~] = construct_greedy_fim_assignment(prm, 'doptimal');
    records(i).seed = seed;
    records(i).methods(1) = solve_fixed(prm, b_fim, labels(1));
    prm_noncoherent = prm;
    prm_noncoherent.sensing_covariance_structure = 'block_diagonal';
    records(i).methods(2) = solve_fixed(prm_noncoherent, b_fim, labels(2));
    fprintf('  cooperative=%d, noncoherent=%d\n', records(i).methods.feasible);
    save(fullfile(out_dir, 'checkpoint.mat'), 'records', 'opt', 'labels');
end

summary.labels = labels;
for m = 1:numel(labels)
    x = [records.methods]; x = x(m:numel(labels):end);
    feasible = [x.feasible]; power = [x(feasible).power_W];
    summary.feasibility_rate(m) = mean(feasible);
    summary.power_mean_W(m) = mean(power, 'omitnan');
    summary.power_median_W(m) = median(power, 'omitnan');
    summary.time_median_s(m) = median([x.time_s], 'omitnan');
end
save(fullfile(out_dir, 'final.mat'), 'records', 'summary', 'opt', 'labels');
end

function method = solve_fixed(prm, b, label)
timer = tic;
res = solve_p3_with_fixed_b(prm, b, 2, 1e-5, 1, 0, 1);
method = struct('label', label, 'feasible', false, 'power_W', NaN, ...
    'time_s', toc(timer), 'status', string(get_field(res, 'status', 'unknown')));
method.feasible = isfield(res, 'is_physical_feasible') && res.is_physical_feasible;
method.power_W = get_field(res, 'final_obj', NaN);
end

function value = get_field(s, name, default_value)
if isfield(s, name), value = s.(name); else, value = default_value; end
end

function record = empty_record()
method = struct('label', "", 'feasible', false, 'power_W', NaN, ...
    'time_s', NaN, 'status', "not_run");
record = struct('seed', NaN, 'methods', repmat(method, 1, 2));
end
