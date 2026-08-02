function summary = run_feasibility_gap_pilot(varargin)
%RUN_FEASIBILITY_GAP_PILOT  Small Monte-Carlo feasibility/gap pilot.
%   Compares the convex SDR power lower bound against the proposed recovery
%   pipeline.  Results are checkpointed after every seed.

ip = inputParser;
addParameter(ip, 'Seeds', 1:3, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'T_max', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'M', 8, @(x) isnumeric(x) && isscalar(x) && x >= 2);
addParameter(ip, 'Nt', 4, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'K', 4, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'P', 2, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'N_req', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'Output_dir', fullfile(pwd, '..', '..', 'experiment_packages', ...
    'v1.0', 'results', 'feasibility_gap_pilot'), @(x) ischar(x) || isstring(x));
parse(ip, varargin{:});
opt = ip.Results;
out_dir = char(opt.Output_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
diary(fullfile(out_dir, 'pilot.log')); cleanup_diary = onCleanup(@() diary('off'));

seeds = opt.Seeds(:).';
records = repmat(empty_record(), numel(seeds), 1);
for i = 1:numel(seeds)
    seed = seeds(i);
    fprintf('\nPilot %d/%d, seed=%d, M=%d, Nt=%d, K=%d, P=%d\n', ...
        i, numel(seeds), seed, opt.M, opt.Nt, opt.K, opt.P);
    prm = generate_scenario(opt.M, opt.Nt, opt.K, opt.P, 2, 20, 'auto', ...
        'AreaSize', 400, 'N_req', opt.N_req, 'eps_h', 0.05, 'seed', seed);
    prm.solver = 'mosek';
    prm.recovery_max_candidates = 3;
    prm.recovery_stop_first_feasible = false;
    prm.recovery_mosek_max_time = 60;
    prm.recovery_slack_diagnosis = true;
    prm.recovery_slack_guided_slots = 1;
    prm.recovery_include_greedy_fim = true;

    records(i).seed = seed;
    [records(i).sdr_status, records(i).sdr_power_W, records(i).sdr_time_s] = ...
        solve_sdr_lower_bound(prm);
    fprintf('  SDR: %s, P=%.5g W, %.1f s\n', records(i).sdr_status, ...
        records(i).sdr_power_W, records(i).sdr_time_s);
    if ~contains(records(i).sdr_status, 'Solved')
        save(fullfile(out_dir, 'pilot_checkpoint.mat'), 'records', 'opt');
        continue;
    end

    timer = tic;
    res = baseline_alg2(prm, opt.T_max, 1e-5, 1, 1, 1.0, true);
    records(i).proposed_time_s = toc(timer);
    records(i).proposed_status = string(res.status);
    records(i).proposed_feasible = isfield(res, 'is_physical_feasible') && ...
        res.is_physical_feasible;
    records(i).dc_iterations = get_field(res, 'dc_iterations', NaN);
    records(i).candidates_tested = get_field(res, 'recovery_candidates_tested', NaN);
    records(i).proposed_power_W = get_field(res, 'final_obj', NaN);
    if records(i).proposed_feasible
        records(i).power_penalty_pct = 100 * ...
            (records(i).proposed_power_W - records(i).sdr_power_W) / records(i).sdr_power_W;
    end
    fprintf('  Proposed: %s, feasible=%d, P=%.5g W, gap=%.2f%%, %.1f s\n', ...
        records(i).proposed_status, records(i).proposed_feasible, ...
        records(i).proposed_power_W, records(i).power_penalty_pct, ...
        records(i).proposed_time_s);
    save(fullfile(out_dir, 'pilot_checkpoint.mat'), 'records', 'opt');
end

valid_sdr = arrayfun(@(r) contains(r.sdr_status, 'Solved'), records);
valid_sdr = valid_sdr(:);
valid_prop = valid_sdr & [records.proposed_feasible].';
summary.n_total = numel(records);
summary.n_sdr_feasible = nnz(valid_sdr);
summary.n_proposed_feasible = nnz(valid_prop);
summary.feasibility_rate = nnz(valid_prop) / max(nnz(valid_sdr), 1);
summary.power_penalty_pct = [records(valid_prop).power_penalty_pct];
summary.power_penalty_mean_pct = mean(summary.power_penalty_pct, 'omitnan');
summary.power_penalty_median_pct = median(summary.power_penalty_pct, 'omitnan');
save(fullfile(out_dir, 'pilot_final.mat'), 'records', 'summary', 'opt');
fprintf('\nPilot summary: feasible %d/%d (%.1f%%), mean gap %.2f%%\n', ...
    summary.n_proposed_feasible, summary.n_sdr_feasible, ...
    100 * summary.feasibility_rate, summary.power_penalty_mean_pct);
end

function [status, power_W, elapsed_s] = solve_sdr_lower_bound(prm)
K = prm.K; N = prm.N; M = prm.M; P = prm.P;
W0 = cell(K,1);
for k = 1:K, W0{k} = eye(N) * prm.Pmax / (K * M); end
b0 = ones(M,P) * prm.N_req / M;
timer = tic;
[W, Z, ~, ~, ~, status] = solve_p3_sca_t(prm, W0, b0, 0, 0);
elapsed_s = toc(timer);
power_W = NaN;
if contains(status, 'Solved')
    power_W = sum(cellfun(@(X) real(trace(X)), W)) + real(trace(Z));
end
end

function value = get_field(s, name, default_value)
if isfield(s, name), value = s.(name); else, value = default_value; end
end

function r = empty_record()
r = struct('seed', NaN, 'sdr_status', "not_run", 'sdr_power_W', NaN, ...
    'sdr_time_s', NaN, 'proposed_status', "not_run", 'proposed_feasible', false, ...
    'proposed_power_W', NaN, 'power_penalty_pct', NaN, 'proposed_time_s', NaN, ...
    'dc_iterations', NaN, 'candidates_tested', NaN);
end
