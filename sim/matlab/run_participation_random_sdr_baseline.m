function summary = run_participation_random_sdr_baseline(varargin)
%RUN_PARTICIPATION_RANDOM_SDR_BASELINE  Complete random and SDR references.
%   Uses the same 30 common scenario seeds as the topology ablation. Random
%   associations preserve exactly N_req APs per target and are optimized by
%   the same fixed-b joint beamforming/covariance solver.

ip = inputParser;
addParameter(ip, 'Seeds', 1:30, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'Output_dir', fullfile(pwd, 'experiment_packages', 'v1.0', ...
    'results', 'participation_random_sdr_baseline'), @(x) ischar(x) || isstring(x));
parse(ip, varargin{:}); opt = ip.Results;
out_dir = char(opt.Output_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

records = repmat(empty_record(), numel(opt.Seeds), 1);
for i = 1:numel(opt.Seeds)
    seed = opt.Seeds(i);
    fprintf('\nRandom/SDR baseline %d/%d, seed=%d\n', i, numel(opt.Seeds), seed);
    prm = generate_scenario(6, 2, 3, 2, 2, 20, 'auto', ...
        'AreaSize', 400, 'N_req', 3, 'eps_h', 0.05, 'seed', seed);
    prm.solver = 'mosek'; prm.recovery_mosek_max_time = 30;
    records(i).seed = seed;
    [records(i).sdr_status, records(i).sdr_power_W, records(i).sdr_time_s] = solve_sdr(prm);
    b_random = random_assignment(prm, 10000 + seed);
    timer = tic;
    res = solve_p3_with_fixed_b(prm, b_random, 2, 1e-5, 1, 0, 1);
    records(i).random_time_s = toc(timer);
    records(i).random_status = string(get_field(res, 'status', 'unknown'));
    records(i).random_feasible = isfield(res, 'is_physical_feasible') && res.is_physical_feasible;
    records(i).random_power_W = get_field(res, 'final_obj', NaN);
    if records(i).random_feasible && isfinite(records(i).sdr_power_W)
        records(i).random_gap_pct = 100 * (records(i).random_power_W - records(i).sdr_power_W) / records(i).sdr_power_W;
    end
    fprintf('  SDR=%s, random feasible=%d\n', records(i).sdr_status, records(i).random_feasible);
    save(fullfile(out_dir, 'checkpoint.mat'), 'records', 'opt');
end
summary.n_total = numel(records);
summary.sdr_feasibility_rate = mean(contains(string({records.sdr_status}), 'Solved'));
summary.random_feasibility_rate = mean([records.random_feasible]);
summary.random_power_mean_W = mean([records([records.random_feasible]).random_power_W], 'omitnan');
summary.random_power_median_W = median([records([records.random_feasible]).random_power_W], 'omitnan');
summary.random_gap_median_pct = median([records([records.random_feasible]).random_gap_pct], 'omitnan');
save(fullfile(out_dir, 'final.mat'), 'records', 'summary', 'opt');
end

function [status, power, elapsed] = solve_sdr(prm)
W0 = cell(prm.K,1);
for k = 1:prm.K, W0{k} = eye(prm.N) * prm.Pmax / (prm.K * prm.M); end
b0 = ones(prm.M,prm.P) * prm.N_req / prm.M;
timer = tic;
[W, Z, ~, ~, ~, status] = solve_p3_sca_t(prm, W0, b0, 0, 0);
elapsed = toc(timer); power = NaN;
if contains(status, 'Solved'), power = sum(cellfun(@(X) real(trace(X)), W)) + real(trace(Z)); end
end

function b = random_assignment(prm, seed)
state = rng; cleanup = onCleanup(@() rng(state)); %#ok<NASGU>
rng(seed, 'twister'); b = zeros(prm.M, prm.P);
for p = prm.active_targets
    b(randperm(prm.M, prm.N_req), p) = 1;
end
end

function value = get_field(s, name, default_value)
if isfield(s, name), value = s.(name); else, value = default_value; end
end

function r = empty_record()
r = struct('seed', NaN, 'sdr_status', "not_run", 'sdr_power_W', NaN, 'sdr_time_s', NaN, ...
    'random_status', "not_run", 'random_feasible', false, 'random_power_W', NaN, ...
    'random_gap_pct', NaN, 'random_time_s', NaN);
end
