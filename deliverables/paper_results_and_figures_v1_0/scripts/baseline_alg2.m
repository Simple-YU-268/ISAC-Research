function res = baseline_alg2(prm, T_max, eps, eta_rank, eta_b, eta_growth, verbose, T_dc_max)
%BASELINE_ALG2  Adaptive binary-penalty DC-SCA with verified recovery.
%   The binary penalty is raised only when the binary residual stagnates.
%   Once the relaxed association is effectively binary, only its top-N
%   projection is refined. Bounded candidate recovery is a fallback.

if nargin < 2, T_max = 80; end
if nargin < 3, eps = 1e-5; end
if nargin < 4, eta_rank = 1.0; end
if nargin < 5, eta_b = 1.0; end
if nargin < 6, eta_growth = 1.0; end
if nargin < 7, verbose = false; end
if nargin < 8 || isempty(T_dc_max), T_dc_max = T_max; end

K = prm.K; P = prm.P; N = prm.N; M = prm.M; Nt = prm.N / prm.M;
W_init = cell(K,1);
for k = 1:K
    W_init{k} = eye(N) * (prm.Pmax / prm.K / prm.M);
end

b0 = ones(M,P) * (prm.N_req / M);
[W_sdr, Z_sdr, ~, b_sdr, ~, status0] = solve_p3_sca_t(prm, W_init, b0, 0, 0);
if ~contains(status0, 'Solved')
    % Fast gatekeeper: continuous relaxation infeasible => integer problem infeasible
    res = struct('status', 'infeasible_relaxed_gatekeeper', ...
                 'final_obj', NaN, 'sum_rate', NaN, 'max_violation', inf, ...
                 'W', [], 'Z', [], 'S_p', [], 'b', [], 'mu', [], 'M_p', [], ...
                 'iters', 0, 'obj_trace', [], 'binary_converged', false, ...
                 'init_label', 'gatekeeper', 'rounding_label', '');
    return;
end

W_prev = W_sdr;
b_prev = b_sdr;
sdr_power = sum(cellfun(@(W) real(trace(W)), W_sdr)) + real(trace(Z_sdr));
sdr_rank_residual = sum(cellfun(@(W) max(0, real(trace(W)) - ...
    max(real(eig(W,'vector')))), W_sdr));
sdr_binary_distance = max(min(b_sdr(:), 1 - b_sdr(:)));
fast_max_iter = min(T_max, T_dc_max);
true_obj_trace = zeros(fast_max_iter,1);
power_trace = zeros(fast_max_iter,1);
rank_trace = zeros(fast_max_iter,1);
binary_trace = zeros(fast_max_iter,1);
binary_distance_trace = zeros(fast_max_iter,1);
eta_b_trace = zeros(fast_max_iter,1);
eta_rank_trace = zeros(fast_max_iter,1);
rho_b_prox_trace = zeros(fast_max_iter,1);
rounding_gap_trace = zeros(fast_max_iter,1);
topology_changed_trace = false(fast_max_iter,1);
dc_converged = false;
dc_stopped_on_stable_topology = false;
last_status = status0;
adaptive_binary_penalty = get_prm_option(prm, 'adaptive_binary_penalty', true);
eta_b_max = get_prm_option(prm, 'eta_b_max', 50);
eta_b_growth = get_prm_option(prm, 'eta_b_growth', max(eta_growth, 2));
binary_penalty_schedule = get_prm_option(prm, 'binary_penalty_schedule', []);
rank_penalty_schedule = get_prm_option(prm, 'rank_penalty_schedule', []);
rho_b_prox_schedule = get_prm_option(prm, 'rho_b_prox_schedule', []);
rounding_gap_min = get_prm_option(prm, 'rounding_gap_min', 0.05);
% A stable top-N support alone is not enough: a fractional point can keep
% the same support while its fixed-b projection is infeasible.  Permit the
% fast path only after the relaxed association is also sufficiently close
% to binary for a reliable projection.
rounding_binary_distance_max = get_prm_option(prm, ...
    'rounding_binary_distance_max', 0.05);
rounding_stable_iters = get_prm_option(prm, 'rounding_stable_iters', 2);
minimum_binary_dc_iters = get_prm_option(prm, 'minimum_binary_dc_iters', 2);
enable_topology_early_stop = get_prm_option(prm, 'enable_topology_early_stop', true);
% Disabled by default until a scenario-specific feasibility screen confirms
% that hardening the selected APs does not remove the PCRB/SINR feasible set.
successive_fixing_enabled = get_prm_option(prm, 'successive_fixing_enabled', false);
successive_fixing_start_iter = get_prm_option(prm, 'successive_fixing_start_iter', 3);
successive_fixing_one_threshold = get_prm_option(prm, ...
    'successive_fixing_one_threshold', 0.95);
successive_fixing_zero_threshold = get_prm_option(prm, ...
    'successive_fixing_zero_threshold', 0.05);
prm.b_fixed_mask = false(M,P);
prm.b_fixed_values = zeros(M,P);
eta_b_current = eta_b;
previous_rounded = [];
stable_rounds = 0;
for t = 1:fast_max_iter
    if ~isempty(binary_penalty_schedule)
        eta_b_current = binary_penalty_schedule(min(t, numel(binary_penalty_schedule)));
    end
    eta_rank_current = eta_rank;
    if ~isempty(rank_penalty_schedule)
        eta_rank_current = rank_penalty_schedule(min(t, numel(rank_penalty_schedule)));
    end
    prm.rho_b_prox = 0;
    if ~isempty(rho_b_prox_schedule)
        prm.rho_b_prox = rho_b_prox_schedule(min(t, numel(rho_b_prox_schedule)));
    end
    fixed_mask_before_solve = prm.b_fixed_mask;
    fixed_values_before_solve = prm.b_fixed_values;
    [W_new, Z_new, ~, b_new, ~, status] = ...
        solve_p3_sca_t(prm, W_prev, b_prev, eta_rank_current, eta_b_current);
    if ~contains(status, 'Solved') && any(fixed_mask_before_solve(:))
        % Variable fixing is a heuristic, not a relaxation of the original
        % problem.  If it cuts away the feasible set, fall back to the
        % unfixed SCA subproblem instead of declaring the instance infeasible.
        prm.b_fixed_mask = false(M,P);
        prm.b_fixed_values = zeros(M,P);
        [W_new, Z_new, ~, b_new, ~, status] = ...
            solve_p3_sca_t(prm, W_prev, b_prev, eta_rank_current, eta_b_current);
        if contains(status, 'Solved')
            successive_fixing_enabled = false;
            if verbose
                fprintf('  DC %d: successive fixing rolled back after infeasible subproblem.\n', t);
            end
        else
            prm.b_fixed_mask = fixed_mask_before_solve;
            prm.b_fixed_values = fixed_values_before_solve;
        end
    end
    if ~contains(status, 'Solved'), break; end
    last_status = status;
    rank_trace(t) = sum(cellfun(@(W) max(0, real(trace(W)) - max(real(eig(W,'vector')))), W_new));
    binary_trace(t) = sum(b_new(:) .* (1 - b_new(:)));
    binary_distance_trace(t) = max(min(b_new(:), 1 - b_new(:)));
    rounded_now = round_assignment(b_new, prm);
    rounding_gap_trace(t) = topn_assignment_gap(b_new, prm);
    topology_changed_trace(t) = ~isempty(previous_rounded) && ...
        ~isequal(rounded_now, previous_rounded);
    if ~topology_changed_trace(t)
        stable_rounds = stable_rounds + 1;
    else
        stable_rounds = 1;
    end
    previous_rounded = rounded_now;
    eta_b_trace(t) = eta_b_current;
    eta_rank_trace(t) = eta_rank_current;
    rho_b_prox_trace(t) = prm.rho_b_prox;
    base_power = sum(cellfun(@(W) real(trace(W)), W_new)) + real(trace(Z_new));
    power_trace(t) = base_power;
    true_obj_trace(t) = base_power + eta_rank_current * rank_trace(t) + eta_b_current * binary_trace(t);
    if verbose
        fprintf('  DC %d: F=%.6f, power=%.6f, rank=%.2e, bin=%.2e, eta_b=%.3g\n', ...
            t, true_obj_trace(t), base_power, rank_trace(t), binary_distance_trace(t), eta_b_current);
    end
    W_prev = W_new;
    b_prev = b_new;
    if successive_fixing_enabled && t >= successive_fixing_start_iter
        [prm.b_fixed_mask, prm.b_fixed_values] = update_successive_fixes( ...
            b_new, prm.b_fixed_mask, prm.b_fixed_values, prm, ...
            successive_fixing_one_threshold, successive_fixing_zero_threshold);
    end
    strict_binary = binary_distance_trace(t) <= eps;
    stable_rounding = t >= minimum_binary_dc_iters && ...
        stable_rounds >= rounding_stable_iters && ...
        rounding_gap_trace(t) >= rounding_gap_min && ...
        binary_distance_trace(t) <= rounding_binary_distance_max;
    if rank_trace(t) <= eps && strict_binary
        dc_converged = true;
        break;
    end
    if enable_topology_early_stop && rank_trace(t) <= eps && stable_rounding
        % End only the continuous phase.  The full binary recovery pool is
        % still required because stable fractional support is not a binary
        % feasibility certificate.
        dc_stopped_on_stable_topology = true;
        break;
    end
    if isempty(binary_penalty_schedule) && adaptive_binary_penalty && ...
            binary_distance_trace(t) > rounding_binary_distance_max
        % Binary feasibility, rather than objective stagnation, controls
        % the continuation parameter.  This avoids spending several costly
        % SDPs at a weak penalty after the topology has already stabilized.
        eta_b_current = min(eta_b_current * eta_b_growth, eta_b_max);
    end
end
iters = find(true_obj_trace ~= 0, 1, 'last');
if isempty(iters)
    res = struct('status', last_status, 'dc_converged', false, 'dc_iterations', 0);
    return;
end
true_obj_trace = true_obj_trace(1:iters);
power_trace = power_trace(1:iters);
rank_trace = rank_trace(1:iters);
binary_trace = binary_trace(1:iters);
binary_distance_trace = binary_distance_trace(1:iters);
eta_b_trace = eta_b_trace(1:iters);
eta_rank_trace = eta_rank_trace(1:iters);
rho_b_prox_trace = rho_b_prox_trace(1:iters);
topology_changed_trace = topology_changed_trace(1:iters);

% The DC projection is the primary candidate.  Geometry-aware FIM and local
% swaps are independent recovery competitors, all evaluated before selection.
% Candidates differ only by a single selected/unselected AP swap and are
% ordered by their loss in the relaxed b score.  This is a recovery stage,
% not an additional random or nearest-AP baseline.
recovery_max_candidates = get_prm_option(prm, 'recovery_max_candidates', 21);
recovery_stop_first_feasible = get_prm_option(prm, ...
    'recovery_stop_first_feasible', false);
recovery_include_prediction_anchor = get_prm_option(prm, ...
    'recovery_include_prediction_anchor', false);
recovery_slack_diagnosis = get_prm_option(prm, ...
    'recovery_slack_diagnosis', false);
recovery_slack_guided_slots = get_prm_option(prm, ...
    'recovery_slack_guided_slots', 6);
recovery_slack_after_candidates = get_prm_option(prm, ...
    'recovery_slack_after_candidates', 2);
recovery_fixed_iterations = get_prm_option(prm, ...
    'recovery_fixed_iterations', 1);
recovery_refine_iterations = get_prm_option(prm, ...
    'recovery_refine_iterations', 2);
recovery_include_greedy_fim = get_prm_option(prm, ...
    'recovery_include_greedy_fim', true);
skip_recovery = get_prm_option(prm, 'skip_recovery', false);
if skip_recovery
    % Diagnostic-only mode: preserve the complete continuous DC-SCA traces
    % without spending time on discrete candidate recovery.  This mode is
    % used only for convergence statistics and must never be reported as a
    % physical solution because fixed-b re-optimization is intentionally
    % omitted.
    res = struct('status', 'continuous_trace_only', ...
        'is_physical_feasible', false, 'final_obj', NaN, 'sum_rate', NaN, ...
        'W', W_prev, 'Z', Z_new, 'S_p', [], 'b', b_prev, 'mu', [], 'M_p', [], ...
        'sdr_power', sdr_power, 'sdr_rank_residual', sdr_rank_residual, ...
        'sdr_binary_distance', sdr_binary_distance, ...
        'true_obj_trace', true_obj_trace, 'power_trace', power_trace, ...
        'rank_residual_trace', rank_trace, 'binary_residual_trace', binary_trace, ...
        'binary_distance_trace', binary_distance_trace, 'eta_b_trace', eta_b_trace, ...
        'eta_rank_trace', eta_rank_trace, 'rho_b_prox_trace', rho_b_prox_trace, ...
        'rounding_gap_trace', rounding_gap_trace(1:iters), ...
        'topology_changed_trace', topology_changed_trace, ...
        'obj_trace_dc', true_obj_trace, 'dc_converged', dc_converged, ...
        'dc_stopped_on_stable_topology', dc_stopped_on_stable_topology, ...
        'dc_iterations', iters, 'dc_rank_deficiency', rank_trace(end), ...
        'dc_binary_distance', binary_distance_trace(end), ...
        'init_label', 'unpenalized_sdr', ...
        'rounding_label', 'not_run_trace_only');
    return;
end
initial_candidate_limit = recovery_max_candidates;
if recovery_slack_diagnosis
    % Reserve diagnostic repair slots while preserving both fast-track
    % greedy FIM and DC Top-N candidates.
    initial_candidate_limit = min(recovery_max_candidates, ...
        max(2, recovery_max_candidates - recovery_slack_guided_slots));
end
candidates = recovery_assignments(b_prev, prm, initial_candidate_limit, ...
    recovery_include_prediction_anchor, recovery_include_greedy_fim);
res = [];
tested_candidates = 0;
candidate_runtime_s = NaN(numel(candidates), 1);
candidate_feasible = false(numel(candidates), 1);
candidate_status = strings(numel(candidates), 1);
candidate_objective = NaN(numel(candidates), 1);
candidate_slack = NaN(numel(candidates), 1);
candidate_slack_diagnosis = cell(numel(candidates), 1);
c = 1;
while c <= numel(candidates)
    tested_candidates = c;
    candidate_timer = tic;
    candidate_res = solve_p3_with_fixed_b(prm, candidates{c}, ...
        recovery_fixed_iterations, eps, eta_rank, 0, 1.0);
    if isfield(candidate_res, 'solver_status') && ...
            contains(candidate_res.solver_status, 'Solved') && ...
            isfield(candidate_res, 'is_physical_feasible') && ...
            ~candidate_res.is_physical_feasible && ...
            recovery_refine_iterations > recovery_fixed_iterations
        candidate_res = solve_p3_with_fixed_b(prm, candidates{c}, ...
            recovery_refine_iterations, eps, eta_rank, 0, 1.0);
    end
    candidate_runtime_s(c) = toc(candidate_timer);
    if isfield(candidate_res, 'status')
        candidate_status(c) = string(candidate_res.status);
    end
    if isfield(candidate_res, 'is_physical_feasible') && ...
            candidate_res.is_physical_feasible
        candidate_feasible(c) = true;
        if isfield(candidate_res, 'final_obj')
            candidate_objective(c) = candidate_res.final_obj;
        end
        if isempty(res) || candidate_res.final_obj < res.final_obj
            res = candidate_res;
            res.recovery_candidate_index = c;
        end
        if recovery_stop_first_feasible
            save_recovery_checkpoint(prm, c, candidate_runtime_s, candidate_feasible, ...
                candidate_status, candidate_objective);
            break;
        end
    elseif recovery_slack_diagnosis && c == recovery_slack_after_candidates
        candidate_slack_diagnosis{c} = diagnose_fixed_b_slack(prm, candidates{c});
        candidate_slack(c) = candidate_slack_diagnosis{c}.total_slack;
        % The first (top-N) candidate supplies a physical violation map.
        % Use it once to add a small number of AP swaps that explicitly
        % improve the weak FIM directions of slack-dominant targets.
        if c == 1 && numel(candidates) < recovery_max_candidates
            old_count = numel(candidates);
            guided = pcrb_guided_swap_candidates(candidates{c}, prm, ...
                candidate_slack_diagnosis{c}, recovery_max_candidates - old_count);
            for q = 1:numel(guided)
                candidates = append_unique_candidate(candidates, guided{q});
                if numel(candidates) >= recovery_max_candidates, break; end
            end
            if numel(candidates) > old_count
                new_indices = old_count+1:numel(candidates);
                candidate_runtime_s(new_indices,1) = NaN;
                candidate_feasible(new_indices,1) = false;
                candidate_status(new_indices,1) = "";
                candidate_objective(new_indices,1) = NaN;
                candidate_slack(new_indices,1) = NaN;
                candidate_slack_diagnosis(new_indices,1) = {[]};
            end
        end
    end
    save_recovery_checkpoint(prm, c, candidate_runtime_s, candidate_feasible, ...
        candidate_status, candidate_objective);
    c = c + 1;
end
if isempty(res)
    res = struct('status', 'infeasible_after_recovery', ...
        'is_physical_feasible', false, 'b', candidates{1});
end
res.true_obj_trace = true_obj_trace;
res.sdr_power = sdr_power;
res.sdr_rank_residual = sdr_rank_residual;
res.sdr_binary_distance = sdr_binary_distance;
res.power_trace = power_trace;
res.rank_residual_trace = rank_trace;
res.binary_residual_trace = binary_trace;
res.binary_distance_trace = binary_distance_trace;
res.eta_b_trace = eta_b_trace;
res.eta_rank_trace = eta_rank_trace;
res.rho_b_prox_trace = rho_b_prox_trace;
res.rounding_gap_trace = rounding_gap_trace(1:iters);
res.topology_changed_trace = topology_changed_trace;
res.obj_trace_dc = true_obj_trace; % legacy plotting name
res.dc_converged = dc_converged;
res.dc_stopped_on_stable_topology = dc_stopped_on_stable_topology;
res.dc_iterations = iters;
res.dc_rank_deficiency = rank_trace(end);
res.dc_binary_distance = binary_distance_trace(end);
res.init_label = 'unpenalized_sdr';
res.rounding_label = 'topN_local_swap_fixed_b_recovery';
res.successive_b_fixed_count = nnz(prm.b_fixed_mask);
res.recovery_candidates_tested = tested_candidates;
res.recovery_candidate_limit = recovery_max_candidates;
res.recovery_stop_on_first_feasible = recovery_stop_first_feasible;
res.recovery_candidate_runtime_s = candidate_runtime_s(1:tested_candidates);
res.recovery_candidate_feasible = candidate_feasible(1:tested_candidates);
res.recovery_candidate_status = candidate_status(1:tested_candidates);
res.recovery_candidate_objective = candidate_objective(1:tested_candidates);
res.recovery_candidate_total_slack = candidate_slack(1:tested_candidates);
res.recovery_candidate_slack_diagnosis = candidate_slack_diagnosis(1:tested_candidates);
end

function value = get_prm_option(prm, field_name, default_value)
if isfield(prm, field_name) && ~isempty(prm.(field_name))
    value = prm.(field_name);
else
    value = default_value;
end
end

function [mask, values] = update_successive_fixes(b, mask, values, prm, one_thr, zero_thr)
% Fix only decisions that cannot violate the exact per-target cardinality.
% This preserves a nonempty continuous subproblem at every DC iteration.
for p = prm.active_targets
    fixed_one = mask(:,p) & values(:,p) > 0.5;
    fixed_zero = mask(:,p) & values(:,p) < 0.5;
    free = find(~mask(:,p));

    % Admit only the strongest near-one associations, up to N_req.
    slots = prm.N_req - nnz(fixed_one);
    if slots > 0
        high = free(b(free,p) >= one_thr);
        [~, order] = sort(b(high,p), 'descend');
        chosen = high(order(1:min(slots, numel(high))));
        mask(chosen,p) = true;
        values(chosen,p) = 1;
    end

    fixed_one = mask(:,p) & values(:,p) > 0.5;
    fixed_zero = mask(:,p) & values(:,p) < 0.5;
    free = find(~mask(:,p));
    if nnz(fixed_one) == prm.N_req
        % The equality sum_m b_mp = N_req fixes all remaining entries to 0.
        mask(free,p) = true;
        values(free,p) = 0;
        continue;
    end

    % Reject only as many near-zero entries as cardinality still permits.
    zero_budget = prm.M - prm.N_req - nnz(fixed_zero);
    if zero_budget > 0
        low = free(b(free,p) <= zero_thr);
        [~, order] = sort(b(low,p), 'ascend');
        rejected = low(order(1:min(zero_budget, numel(low))));
        mask(rejected,p) = true;
        values(rejected,p) = 0;
    end

    % If exactly the remaining number of APs is free, their values are known.
    fixed_one = mask(:,p) & values(:,p) > 0.5;
    free = find(~mask(:,p));
    if numel(free) == prm.N_req - nnz(fixed_one)
        mask(free,p) = true;
        values(free,p) = 1;
    end
end
end

function save_recovery_checkpoint(prm, candidate_index, runtime_s, feasible, status, objective)
if ~isfield(prm, 'recovery_checkpoint_file') || isempty(prm.recovery_checkpoint_file)
    return;
end
checkpoint.candidates_completed = candidate_index;
checkpoint.runtime_s = runtime_s(1:candidate_index);
checkpoint.feasible = feasible(1:candidate_index);
checkpoint.status = status(1:candidate_index);
checkpoint.objective = objective(1:candidate_index);
checkpoint.cumulative_runtime_s = cumsum(checkpoint.runtime_s);
checkpoint.updated_at = datetime('now');
save(char(prm.recovery_checkpoint_file), 'checkpoint');
end

function b = round_assignment(b_relaxed, prm)
b = zeros(prm.M,prm.P);
for p = prm.active_targets
    [~,idx] = sort(b_relaxed(:,p),'descend');
    b(idx(1:prm.N_req),p) = 1;
end
end

function gap = topn_assignment_gap(b_relaxed, prm)
gap = inf;
for p = prm.active_targets
    values = sort(real(b_relaxed(:,p)), 'descend');
    if prm.N_req < numel(values)
        gap = min(gap, values(prm.N_req) - values(prm.N_req + 1));
    end
end
if isinf(gap), gap = 1; end
end

function candidates = recovery_assignments(b_relaxed, prm, max_candidates, include_prediction_anchor, include_greedy_fim)
base = round_assignment(b_relaxed, prm);
if nargin < 5, include_greedy_fim = true; end
candidates = {};
% Preserve the topology selected by the dual DC continuous phase first.
candidates = append_unique_candidate(candidates, base);
if include_greedy_fim
    candidates = append_unique_candidate(candidates, ...
        construct_greedy_fim_assignment(prm, 'doptimal'));
end
if include_prediction_anchor
    candidates = append_unique_candidate(candidates, prediction_nearest_assignment(prm));
end
if max_candidates <= 1
    candidates = candidates(1);
    return;
end
swaps = struct('leave', {}, 'enter', {}, 'target', {}, 'loss', {});
for p = prm.active_targets
    selected = find(base(:,p) > 0.5);
    unselected = find(base(:,p) < 0.5);
    for leave = selected(:).'
        for enter = unselected(:).'
            swaps(end+1) = struct('leave', leave, 'enter', enter, ...
                'target', p, 'loss', b_relaxed(leave,p)-b_relaxed(enter,p)); %#ok<AGROW>
        end
    end
end
[~, order] = sort([swaps.loss], 'ascend');
single_limit = min(8, numel(order));
for q = 1:single_limit
    candidates = append_unique_candidate(candidates, apply_swaps(base, swaps(order(q))));
    if numel(candidates) >= max_candidates, return; end
end

% A one-step projection can be trapped in a poor local association.  Add a
% bounded deterministic two-swap beam; this remains a post-SCA recovery
% search because all candidates preserve the exact per-target cardinality.
pair_list = {};
pair_loss = [];
for a = 1:single_limit
    for b = a+1:single_limit
        candidate = apply_swaps(base, [swaps(order(a)), swaps(order(b))]);
        if all(sum(candidate(:,prm.active_targets), 1) == prm.N_req)
            pair_list{end+1,1} = candidate; %#ok<AGROW>
            pair_loss(end+1,1) = swaps(order(a)).loss + swaps(order(b)).loss; %#ok<AGROW>
        end
    end
end
[~, pair_order] = sort(pair_loss, 'ascend');
for q = 1:numel(pair_order)
    candidates = append_unique_candidate(candidates, pair_list{pair_order(q)});
    if numel(candidates) >= max_candidates, break; end
end
end

function candidates = pcrb_guided_swap_candidates(base, prm, diagnosis, max_candidates)
% Targeted one-swap candidates ranked by slack-weighted FIM conditioning gain.
% The score rewards information in the weakest geometric direction, rather
% than merely the Frobenius energy of a likely collinear AP contribution.
candidates = {};
if max_candidates <= 0 || ~isfield(diagnosis, 'pcrb_slack') || ...
        ~isfinite(diagnosis.total_slack)
    return;
end
Nt = prm.N / prm.M;
swaps = struct('leave', {}, 'enter', {}, 'target', {}, 'score', {});
for p = prm.active_targets
    pressure = max(0, real(diagnosis.pcrb_slack(p)));
    if pressure <= 1e-6, continue; end
    selected = find(base(:,p) > 0.5);
    unselected = find(base(:,p) < 0.5);
    base_score = fim_geometry_score(selected, prm.D(:,:,p), Nt);
    for leave = selected(:).'
        for enter = unselected(:).'
            proposed = selected;
            proposed(proposed == leave) = enter;
            gain = fim_geometry_score(proposed, prm.D(:,:,p), Nt) - base_score;
            swaps(end+1) = struct('leave', leave, 'enter', enter, ...
                'target', p, 'score', pressure * gain); %#ok<AGROW>
        end
    end
end
if isempty(swaps), return; end
[~, order] = sort([swaps.score], 'descend');
for q = 1:min(max_candidates, numel(order))
    candidates{end+1,1} = apply_swaps(base, swaps(order(q))); %#ok<AGROW>
end
end

function score = fim_geometry_score(selected, Dp, Nt)
N_theta = size(Dp,2);
G = zeros(N_theta);
for m = selected(:).'
    rows = (m-1)*Nt + (1:Nt);
    Dm = Dp(rows,:);
    G = G + real(Dm' * Dm);
end
% Dimensionless weak-direction information; a small trace-scaled ridge only
% stabilizes the score when a candidate is geometrically rank deficient.
scale = max(trace(G) / max(N_theta,1), 1e-12);
score = min(real(eig(G + 1e-9 * scale * eye(N_theta)))) / scale;
end

function b = prediction_nearest_assignment(prm)
b = zeros(prm.M, prm.P);
if isfield(prm, 'Target_pred_pos')
    target_reference = prm.Target_pred_pos;
else
    target_reference = prm.Target_pos;
end
for p = prm.active_targets
    distance = sqrt(sum((prm.AP_pos - target_reference(p,:)).^2, 2));
    [~, order] = sort(distance, 'ascend');
    b(order(1:prm.N_req), p) = 1;
end
end

function candidate = apply_swaps(base, swaps)
candidate = base;
for q = 1:numel(swaps)
    p = swaps(q).target;
    candidate(swaps(q).leave,p) = 0;
    candidate(swaps(q).enter,p) = 1;
end
end

function candidates = append_unique_candidate(candidates, candidate)
if ~any(cellfun(@(existing) isequal(existing, candidate), candidates))
    candidates{end+1,1} = candidate;
end
end
