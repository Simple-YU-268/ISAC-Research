function res = baseline_alg2(prm, T_max, eps, eta_rank, eta_b, eta_growth, verbose, T_dc_max)
%BASELINE_ALG2  Paper-consistent fixed-penalty DC-SCA and verified recovery.
%   The main loop keeps eta_rank and eta_b fixed, records the true DCP
%   objective and summed rank/binary residuals, then projects each target's
%   relaxed assignment onto exactly N_req APs. If that projection is
%   infeasible, a deterministic, bounded one-AP-swap repair set is tried;
%   every candidate is re-solved with b fixed and physically verified.
%   eta_growth is retained only for backwards-compatible call signatures.

if nargin < 2, T_max = 80; end
if nargin < 3, eps = 1e-5; end
if nargin < 4, eta_rank = 1.0; end
if nargin < 5, eta_b = 1.0; end
if nargin < 6, eta_growth = 1.0; end %#ok<NASGU>
if nargin < 7, verbose = false; end
if nargin < 8 || isempty(T_dc_max), T_dc_max = T_max; end

K = prm.K; P = prm.P; N = prm.N; M = prm.M; Nt = prm.N / prm.M;
W_init = cell(K,1);
for k = 1:K
    W_init{k} = eye(N) * (prm.Pmax / prm.K / prm.M);
end
b0 = ones(M,P) * (prm.N_req / M);
[W_sdr, ~, ~, b_sdr, ~, status0] = solve_p3_sca_t(prm, W_init, b0, 0, 0);
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
fast_max_iter = min(T_max, T_dc_max);
true_obj_trace = zeros(fast_max_iter,1);
rank_trace = zeros(fast_max_iter,1);
binary_trace = zeros(fast_max_iter,1);
dc_converged = false;
last_status = status0;
for t = 1:fast_max_iter
    [W_new, Z_new, ~, b_new, ~, status] = ...
        solve_p3_sca_t(prm, W_prev, b_prev, eta_rank, eta_b);
    if ~contains(status, 'Solved'), break; end
    last_status = status;
    rank_trace(t) = sum(cellfun(@(W) max(0, real(trace(W)) - max(real(eig(W,'vector')))), W_new));
    binary_trace(t) = sum(b_new(:) .* (1 - b_new(:)));
    base_power = sum(cellfun(@(W) real(trace(W)), W_new)) + real(trace(Z_new));
    true_obj_trace(t) = base_power + eta_rank * rank_trace(t) + eta_b * binary_trace(t);
    if verbose
        fprintf('  fixed-DCP %d: F=%.6f, power=%.6f, rank=%.2e, bin=%.2e\n', ...
            t, true_obj_trace(t), base_power, rank_trace(t), binary_trace(t));
    end
    W_prev = W_new;
    b_prev = b_new;
    if t > 1 && rank_trace(t) <= eps && binary_trace(t) <= eps && ...
            abs(true_obj_trace(t)-true_obj_trace(t-1)) <= eps
        dc_converged = true;
        break;
    end
end
iters = find(true_obj_trace ~= 0, 1, 'last');
if isempty(iters)
    res = struct('status', last_status, 'dc_converged', false, 'dc_iterations', 0);
    return;
end
true_obj_trace = true_obj_trace(1:iters);
rank_trace = rank_trace(1:iters);
binary_trace = binary_trace(1:iters);

% Cardinality-preserving projection followed by deterministic local repair.
% Besides the relaxed-b projection, use a small ISAC-aware candidate set:
% the per-AP sensing-to-user coupling efficiency and two normalized mixtures
% of that efficiency with the relaxed score.  These candidates account for
% sensing strength and communication interference jointly, rather than using
% a distance/nearest-AP rule; every one is judged by its fixed-assignment
% physical solution and true transmit power.
candidates = recovery_assignments(b_prev, prm, 21);
res = [];
for c = 1:numel(candidates)
    candidate_res = solve_p3_with_fixed_b(prm, candidates{c}, ...
        T_max, eps, eta_rank, 0, 1.0);
    if isfield(candidate_res, 'is_physical_feasible') && ...
            candidate_res.is_physical_feasible
        if isempty(res) || candidate_res.final_obj < res.final_obj
            res = candidate_res;
            res.recovery_candidate_index = c;
        end
    end
end
if isempty(res)
    res = struct('status', 'infeasible_after_recovery', ...
        'is_physical_feasible', false, 'b', candidates{1});
end
res.true_obj_trace = true_obj_trace;
res.rank_residual_trace = rank_trace;
res.binary_residual_trace = binary_trace;
res.obj_trace_dc = true_obj_trace; % legacy plotting name
res.dc_converged = dc_converged;
res.dc_iterations = iters;
res.dc_rank_deficiency = rank_trace(end);
res.dc_binary_distance = binary_trace(end);
res.init_label = 'unpenalized_sdr';
res.rounding_label = 'relaxed_channel_aware_local_swap_fixed_b_recovery';
end

function b = round_assignment(b_relaxed, prm)
b = zeros(prm.M,prm.P);
for p = prm.active_targets
    [~,idx] = sort(b_relaxed(:,p),'descend');
    b(idx(1:prm.N_req),p) = 1;
end
end

function candidates = recovery_assignments(b_relaxed, prm, max_candidates)
base = round_assignment(b_relaxed, prm);
candidates = {base};

% Add ISAC-aware Top-N projections before spending the bounded budget on
% local swaps.  The score is target-channel energy divided by aggregate
% communication-channel coupling at an AP: large values favor sensing gain
% while reducing likely sensing-waveform interference at UEs.  The two
% mixtures retain the SCA association information.
efficiency = sensing_to_user_efficiency(prm);
score_sets = {efficiency, ...
    0.75 * normalize_columns(b_relaxed) + 0.25 * normalize_columns(efficiency), ...
    0.50 * normalize_columns(b_relaxed) + 0.50 * normalize_columns(efficiency)};
for s = 1:numel(score_sets)
    candidates = append_unique_candidate(candidates, round_assignment(score_sets{s}, prm));
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

function efficiency = sensing_to_user_efficiency(prm)
efficiency = zeros(prm.M, prm.P);
user_coupling = zeros(prm.M, 1);
for m = 1:prm.M
    ap_idx = (m-1)*prm.Nt + (1:prm.Nt);
    user_coupling(m) = real(trace(prm.H(ap_idx,:) * prm.H(ap_idx,:)'));
end
coupling_floor = max(1e-12, 1e-6 * max(user_coupling));
for p = prm.active_targets
    for m = 1:prm.M
        ap_idx = (m-1)*prm.Nt + (1:prm.Nt);
        sensing_gain = real(prm.G(ap_idx,p)' * prm.G(ap_idx,p));
        efficiency(m,p) = sensing_gain / (user_coupling(m) + coupling_floor);
    end
end
end

function x_norm = normalize_columns(x)
x_norm = zeros(size(x));
for p = 1:size(x,2)
    col = real(x(:,p));
    lo = min(col);
    hi = max(col);
    if hi > lo
        x_norm(:,p) = (col - lo) / (hi - lo);
    else
        x_norm(:,p) = 0;
    end
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
