function result = smoke_recovery_vs_nearest(varargin)
%SMOKE_RECOVERY_VS_NEAREST  Focused proposed-versus-nearest recovery check.
%   Runs one common scenario without the random baseline or figure pipeline.

p = inputParser;
addParameter(p, 'Seed', 2026, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'N_req', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'T_max', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'Recovery_max_candidates', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'Recovery_stop_first_feasible', true, @islogical);
addParameter(p, 'Solver', 'mosek', @(x) ischar(x) || isstring(x));
addParameter(p, 'Output_file', '', @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opt = p.Results;

prm = generate_scenario(8, 4, 4, 2, 2, 20, 'auto', ...
    'AreaSize', 400, 'N_req', opt.N_req, 'eps_h', 0.05, 'seed', opt.Seed);
prm.solver = char(opt.Solver);
prm.recovery_max_candidates = opt.Recovery_max_candidates;
prm.recovery_stop_first_feasible = opt.Recovery_stop_first_feasible;

tic;
proposed = baseline_alg2(prm, opt.T_max, 1e-5, 1, 1, 1.3, false);
result.proposed_runtime_s = toc;
result.seed = opt.Seed;
result.n_req = opt.N_req;
result.recovery_max_candidates = opt.Recovery_max_candidates;
result.recovery_stop_first_feasible = opt.Recovery_stop_first_feasible;
result.proposed = summarize_solution(proposed);

% Persist the expensive proposed result before running the comparison arm.
save_result_if_requested(result, opt.Output_file);

try
    tic;
    nearest = solve_p3_with_fixed_b(prm, nearest_assignment_local(prm), ...
        opt.T_max, 1e-5, 1, 1, 1.3);
    result.nearest_runtime_s = toc;
    result.nearest = summarize_solution(nearest);
catch err
    result.nearest_runtime_s = NaN;
    result.nearest = struct('status', "runtime_error", 'physical_feasible', false, ...
        'sensing_sinr_db', NaN, 'power_W', NaN, 'recovery_candidates_tested', NaN);
    result.nearest_error = getReport(err, 'basic', 'hyperlinks', 'off');
end

save_result_if_requested(result, opt.Output_file);

fprintf('Proposed: %s, %.1f s, candidates=%g, sensing SINR=%s dB\\n', ...
    result.proposed.status, result.proposed_runtime_s, ...
    result.proposed.recovery_candidates_tested, mat2str(result.proposed.sens_sinr_db, 5));
fprintf('Nearest:  %s, %.1f s, sensing SINR=%s dB\\n', ...
    result.nearest.status, result.nearest_runtime_s, ...
    mat2str(result.nearest.sens_sinr_db, 5));

end

function save_result_if_requested(result, output_file)
if strlength(string(output_file)) > 0
    output_file = char(output_file);
    output_dir = fileparts(output_file);
    if ~isempty(output_dir) && ~exist(output_dir, 'dir'), mkdir(output_dir); end
    save(output_file, 'result');
end
end

function b = nearest_assignment_local(prm)
b = zeros(prm.M, prm.P);
if isfield(prm, 'Target_pred_pos'), target_reference = prm.Target_pred_pos;
else, target_reference = prm.Target_pos; end
for p = prm.active_targets
    distance = sqrt(sum((prm.AP_pos - target_reference(p,:)).^2, 2));
    [~, order] = sort(distance, 'ascend');
    b(order(1:prm.N_req), p) = 1;
end
end

function summary = summarize_solution(res)
summary = struct('status', string(res.status), 'physical_feasible', false, ...
    'sensing_sinr_db', NaN, 'power_W', NaN, 'recovery_candidates_tested', NaN);
if isfield(res, 'is_physical_feasible')
    summary.physical_feasible = res.is_physical_feasible;
end
if isfield(res, 'sens_sinr_db'), summary.sensing_sinr_db = res.sens_sinr_db(:).'; end
if isfield(res, 'final_obj'), summary.power_W = res.final_obj; end
if isfield(res, 'recovery_candidates_tested')
    summary.recovery_candidates_tested = res.recovery_candidates_tested;
end
end
