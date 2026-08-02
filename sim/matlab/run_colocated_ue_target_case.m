function result = run_colocated_ue_target_case(varargin)
%RUN_COLOCATED_UE_TARGET_CASE  Geometry stress test for AP-association methods.
%   UE 1 is intentionally placed 5 m from Target 1. This exposes whether a
%   sensing-oriented nearest-AP association remains adequate when dedicated
%   sensing leakage and robust communication constraints share the same APs.

p = inputParser;
addParameter(p, 'Seed', 2026, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'N_req', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'T_max', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'Recovery_max_candidates', 21, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'Recovery_stop_first_feasible', true, @islogical);
addParameter(p, 'Recovery_include_prediction_anchor', false, @islogical);
addParameter(p, 'Candidate_time_limit_s', Inf, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Checkpoint_file', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'Solver', 'mosek', @(x) ischar(x) || isstring(x));
addParameter(p, 'Output_file', '', @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opt = p.Results;

ap_pos = [25 25; 200 25; 375 25; 25 200; 375 200; 25 375; 200 375; 375 375];
target_pos = [80 80; 320 300];
ue_pos = [85 80; 260 70; 120 330; 340 120];
target_pred_pos = [90 75; 305 310];
prm = generate_scenario(8, 4, 4, 2, 2, 20, 'auto', ...
    'AreaSize', 400, 'N_req', opt.N_req, 'eps_h', 0.05, 'seed', opt.Seed, ...
    'AP_pos', ap_pos, 'UE_pos', ue_pos, 'Target_pos', target_pos, ...
    'Target_pred_pos', target_pred_pos);
prm.solver = char(opt.Solver);
prm.recovery_max_candidates = opt.Recovery_max_candidates;
prm.recovery_stop_first_feasible = opt.Recovery_stop_first_feasible;
prm.recovery_include_prediction_anchor = opt.Recovery_include_prediction_anchor;
prm.recovery_mosek_max_time = opt.Candidate_time_limit_s;
if strlength(string(opt.Checkpoint_file)) > 0
    checkpoint_file = char(opt.Checkpoint_file);
    checkpoint_dir = fileparts(checkpoint_file);
    if ~isempty(checkpoint_dir) && ~exist(checkpoint_dir, 'dir'), mkdir(checkpoint_dir); end
    prm.recovery_checkpoint_file = checkpoint_file;
end

tic;
proposed = baseline_alg2(prm, opt.T_max, 1e-5, 1, 1, 1.3, false);
result.proposed_runtime_s = toc;
tic;
nearest = solve_p3_with_fixed_b(prm, nearest_assignment_local(prm), ...
    opt.T_max, 1e-5, 1, 1, 1.3);
result.nearest_runtime_s = toc;
result.ap_pos = ap_pos;
result.ue_pos = ue_pos;
result.target_pos = target_pos;
result.target_pred_pos = target_pred_pos;
result.ue_target_separation_m = norm(ue_pos(1,:) - target_pos(1,:));
result.recovery_max_candidates = opt.Recovery_max_candidates;
result.proposed = summarize_solution(proposed);
result.nearest = summarize_solution(nearest);
if strlength(string(opt.Output_file)) > 0
    out_file = char(opt.Output_file);
    out_dir = fileparts(out_file);
    if ~isempty(out_dir) && ~exist(out_dir, 'dir'), mkdir(out_dir); end
    save(out_file, 'result');
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
if isfield(res, 'is_physical_feasible'), summary.physical_feasible = res.is_physical_feasible; end
if isfield(res, 'sens_sinr_db'), summary.sensing_sinr_db = res.sens_sinr_db(:).'; end
if isfield(res, 'final_obj'), summary.power_W = res.final_obj; end
if isfield(res, 'recovery_candidates_tested')
    summary.recovery_candidates_tested = res.recovery_candidates_tested;
end
end
