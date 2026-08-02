function summary = run_topology_stability_mc(varargin)
%RUN_TOPOLOGY_STABILITY_MC  Test whether trade-off extremes change b_mp.
%   The experiment compares a communication-stringent/sensing-loose point
%   against a sensing-stringent/communication-loose point over random seeds.

ip = inputParser;
addParameter(ip, 'Seeds', 1:30, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'T_max', 2, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'Mosek_max_time', 10, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(ip, 'Recovery_max_candidates', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'Recovery_stop_first_feasible', false, @(x) islogical(x) && isscalar(x));
addParameter(ip, 'N_req', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'Endpoints', [6.0, 9.0; 1.5, -3.0], ...
    @(x) isnumeric(x) && isequal(size(x), [2,2]));
addParameter(ip, 'Output_dir', fullfile(pwd, '..', '..', 'experiment_packages', ...
    'v1.0', 'results', 'topology_stability_mc'), @(x) ischar(x) || isstring(x));
addParameter(ip, 'Resume', true, @(x) islogical(x) && isscalar(x));
parse(ip, varargin{:}); opt = ip.Results;
seeds = opt.Seeds(:).'; out_dir = char(opt.Output_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
checkpoint_file = fullfile(out_dir, 'checkpoint.mat');

blank = struct('seed', NaN, 'both_feasible', false, 'exact_same', false, ...
    'mean_jaccard', NaN, 'changed_fraction', NaN, 'power_comm_W', NaN, ...
    'power_sense_W', NaN);
records = repmat(blank, numel(seeds), 1);
if opt.Resume && exist(checkpoint_file, 'file')
    saved = load(checkpoint_file, 'records', 'seeds_saved');
    if isequal(saved.seeds_saved, seeds), records = saved.records; end
end

for i = 1:numel(seeds)
    if ~isnan(records(i).seed), continue; end
    seed = seeds(i); fprintf('Topology-stability seed %d (%d/%d)\n', seed, i, numel(seeds));
    % [Gamma_alpha, gamma_k_dB]: loose sensing/strict communication; then converse.
    endpoints = opt.Endpoints;
    b = cell(2,1); power = NaN(2,1); feasible = false(2,1);
    for q = 1:2
        prm = generate_scenario(6, 2, 3, 2, 2, 20, 'auto', ...
            'AreaSize', 400, 'N_req', opt.N_req, 'eps_h', 0.05, 'seed', seed, ...
            'Gamma_alpha', endpoints(q,1), 'gamma_k_dB', endpoints(q,2));
        prm.solver = 'mosek'; prm.mosek_max_time = opt.Mosek_max_time;
        prm.recovery_max_candidates = opt.Recovery_max_candidates;
        prm.recovery_stop_first_feasible = opt.Recovery_stop_first_feasible;
        prm.recovery_mosek_max_time = opt.Mosek_max_time;
        res = baseline_alg2(prm, opt.T_max, 1e-5, 1, 1, 1.0, false);
        feasible(q) = isfield(res, 'is_physical_feasible') && res.is_physical_feasible;
        if feasible(q), b{q} = round(res.b); power(q) = res.final_obj; end
    end
    records(i).seed = seed;
    records(i).both_feasible = all(feasible);
    records(i).power_comm_W = power(1); records(i).power_sense_W = power(2);
    if records(i).both_feasible
        records(i).exact_same = isequal(b{1}, b{2});
        records(i).changed_fraction = nnz(b{1} ~= b{2}) / numel(b{1});
        jac = NaN(1,size(b{1},2));
        for p = 1:size(b{1},2)
            u = nnz((b{1}(:,p) > 0) | (b{2}(:,p) > 0));
            jac(p) = nnz((b{1}(:,p) > 0) & (b{2}(:,p) > 0)) / max(u,1);
        end
        records(i).mean_jaccard = mean(jac);
    end
    seeds_saved = seeds; save(checkpoint_file, 'records', 'seeds_saved', 'opt');
end

valid = [records.both_feasible].';
summary.num_seeds = numel(seeds); summary.num_both_feasible = nnz(valid);
summary.both_feasible_rate = mean(valid);
summary.exact_same_rate = mean([records(valid).exact_same]);
summary.mean_jaccard = mean([records(valid).mean_jaccard]);
summary.median_jaccard = median([records(valid).mean_jaccard]);
summary.mean_changed_fraction = mean([records(valid).changed_fraction]);
summary.median_power_comm_W = median([records(valid).power_comm_W]);
summary.median_power_sense_W = median([records(valid).power_sense_W]);
save(fullfile(out_dir, 'topology_stability_final.mat'), 'records', 'summary', 'opt');
fprintf(['Both feasible: %d/%d; exact-same b: %.1f%%; mean Jaccard: %.3f; ' ...
    'mean changed entries: %.1f%%\n'], summary.num_both_feasible, summary.num_seeds, ...
    100*summary.exact_same_rate, summary.mean_jaccard, 100*summary.mean_changed_fraction);
end
