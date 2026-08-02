function result = time_end_to_end_optimization(varargin)
%TIME_END_TO_END_OPTIMIZATION  Timed single-scenario proposed optimization.

p = inputParser;
addParameter(p, 'Seed', 2028, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'N_req', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'T_max', 30, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'Recovery_max_candidates', 21, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'Candidate_time_limit_s', 120, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Recovery_slack_diagnosis', false, @(x) islogical(x) && isscalar(x));
addParameter(p, 'Eta_b_schedule', [], @(x) isempty(x) || (isnumeric(x) && isvector(x) && all(x > 0)));
addParameter(p, 'Eta_rank_schedule', [], @(x) isempty(x) || (isnumeric(x) && isvector(x) && all(x >= 0)));
addParameter(p, 'Rho_b_prox_schedule', [], @(x) isempty(x) || (isnumeric(x) && isvector(x) && all(x >= 0)));
addParameter(p, 'Rounding_gap_min', 0.05, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'Rounding_binary_distance_max', 0.02, @(x) isnumeric(x) && isscalar(x) && x > 0 && x <= 0.5);
addParameter(p, 'Rounding_stable_iters', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'Minimum_binary_dc_iters', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'Output_dir', fullfile(pwd, 'experiment_packages', 'v1.0', 'results', 'timing'), @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opt = p.Results;
out_dir = char(opt.Output_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
tag = sprintf('e2e_seed%d_nreq%d_t%d', opt.Seed, opt.N_req, opt.T_max);
diary_file = fullfile(out_dir, [tag, '.log']);
diary(diary_file); cleanup_diary = onCleanup(@() diary('off'));

prm = generate_scenario(8, 4, 4, 2, 2, 20, 'auto', ...
    'AreaSize', 400, 'N_req', opt.N_req, 'eps_h', 0.05, 'seed', opt.Seed);
prm.solver = 'mosek';
prm.recovery_max_candidates = opt.Recovery_max_candidates;
prm.recovery_stop_first_feasible = false;
prm.recovery_mosek_max_time = opt.Candidate_time_limit_s;
prm.recovery_slack_diagnosis = opt.Recovery_slack_diagnosis;
prm.recovery_checkpoint_file = fullfile(out_dir, [tag, '_recovery_checkpoint.mat']);
prm.binary_penalty_schedule = opt.Eta_b_schedule;
prm.rank_penalty_schedule = opt.Eta_rank_schedule;
prm.rho_b_prox_schedule = opt.Rho_b_prox_schedule;
prm.rounding_gap_min = opt.Rounding_gap_min;
prm.rounding_binary_distance_max = opt.Rounding_binary_distance_max;
prm.rounding_stable_iters = opt.Rounding_stable_iters;
prm.minimum_binary_dc_iters = opt.Minimum_binary_dc_iters;

fprintf('E2E timing start: seed=%d N_req=%d T_max=%d candidates=%d\\n', ...
    opt.Seed, opt.N_req, opt.T_max, opt.Recovery_max_candidates);
timer = tic;
res = baseline_alg2(prm, opt.T_max, 1e-5, 1, 1, 1.0, true);
result.elapsed_s = toc(timer);
result.seed = opt.Seed;
result.n_req = opt.N_req;
result.t_max = opt.T_max;
result.recovery_max_candidates = opt.Recovery_max_candidates;
result.status = string(res.status);
result.physical_feasible = isfield(res, 'is_physical_feasible') && res.is_physical_feasible;
result.dc_iterations = get_field_or(res, 'dc_iterations', NaN);
result.recovery_candidates_tested = get_field_or(res, 'recovery_candidates_tested', NaN);
result.recovery_runtime_s = sum(get_field_or(res, 'recovery_candidate_runtime_s', []));
result.final_power_W = get_field_or(res, 'final_obj', NaN);
result.sens_sinr_db = get_field_or(res, 'sens_sinr_db', NaN);
save(fullfile(out_dir, [tag, '.mat']), 'result', 'res');
fprintf('E2E timing end: %.2f s, status=%s, feasible=%d, DC=%g, candidates=%g\\n', ...
    result.elapsed_s, result.status, result.physical_feasible, ...
    result.dc_iterations, result.recovery_candidates_tested);
end

function value = get_field_or(s, field_name, default_value)
if isfield(s, field_name), value = s.(field_name); else, value = default_value; end
end
