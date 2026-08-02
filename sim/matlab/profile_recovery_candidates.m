function profile = profile_recovery_candidates(varargin)
%PROFILE_RECOVERY_CANDIDATES  One-run candidate-budget profile for recovery.
%   A full bounded recovery search is executed once. Prefixes of its ordered
%   candidate trace represent the candidate budgets 1, 2, ..., max_candidates.

p = inputParser;
addParameter(p, 'Seed', 2026, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'N_req', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'T_max', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'Max_candidates', 21, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'Solver', 'mosek', @(x) ischar(x) || isstring(x));
addParameter(p, 'Candidate_time_limit_s', 120, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Output_file', '', @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opt = p.Results;

prm = generate_scenario(8, 4, 4, 2, 2, 20, 'auto', ...
    'AreaSize', 400, 'N_req', opt.N_req, 'eps_h', 0.05, 'seed', opt.Seed);
prm.solver = char(opt.Solver);
prm.recovery_mosek_max_time = opt.Candidate_time_limit_s;
prm.recovery_max_candidates = opt.Max_candidates;
prm.recovery_stop_first_feasible = false;
if strlength(string(opt.Output_file)) > 0
    [out_dir, out_name] = fileparts(char(opt.Output_file));
    if ~isempty(out_dir) && ~exist(out_dir, 'dir'), mkdir(out_dir); end
    prm.recovery_checkpoint_file = fullfile(out_dir, [out_name, '_checkpoint.mat']);
end
res = baseline_alg2(prm, opt.T_max, 1e-5, 1, 1, 1.3, false);

profile.seed = opt.Seed;
profile.n_req = opt.N_req;
profile.status = string(res.status);
if isfield(res, 'recovery_candidate_runtime_s')
    profile.candidate_runtime_s = res.recovery_candidate_runtime_s;
    profile.candidate_feasible = res.recovery_candidate_feasible;
    profile.candidate_status = res.recovery_candidate_status;
    profile.candidate_objective = res.recovery_candidate_objective;
    profile.cumulative_runtime_s = cumsum(profile.candidate_runtime_s);
    profile.first_feasible_candidate = find(profile.candidate_feasible, 1, 'first');
    if isempty(profile.first_feasible_candidate)
        profile.first_feasible_candidate = NaN;
    end
else
    profile.candidate_runtime_s = [];
    profile.candidate_feasible = [];
    profile.candidate_status = strings(0,1);
    profile.candidate_objective = [];
    profile.cumulative_runtime_s = [];
    profile.first_feasible_candidate = NaN;
end
if strlength(string(opt.Output_file)) > 0
    out_file = char(opt.Output_file);
    out_dir = fileparts(out_file);
    if ~isempty(out_dir) && ~exist(out_dir, 'dir'), mkdir(out_dir); end
    save(out_file, 'profile');
end
end
