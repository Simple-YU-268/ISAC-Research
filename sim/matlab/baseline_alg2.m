function res = baseline_alg2(prm, T_max, eps, eta_rank, eta_b, eta_growth, verbose)
%BASELINE_ALG2  Fast DC-SCA + multi-candidate greedy rounding for large-scale MC
%
%   Phase 1: convex warm-start (eta_rank=eta_b=0) from uniform and distance
%            heuristic binary patterns.
%   Phase 2: short double-DC loop (max 5 iters).  If binary convergence is not
%            reached, the current relaxed b is frozen.
%   Phase 3: greedy rounding with multiple fallback candidates
%            (relaxed top-N_req, relaxed top-(N_req+1), distance heuristic) and
%            a final fixed-b re-optimization.  The first candidate that yields a
%            Solved status is returned; if all fail, the trial is marked
%            infeasible.

if nargin < 2, T_max = 80; end
if nargin < 3, eps = 1e-5; end
if nargin < 4, eta_rank = 1.0; end
if nargin < 5, eta_b = 1.0; end
if nargin < 6, eta_growth = 1.3; end
if nargin < 7, verbose = false; end

K = prm.K; P = prm.P; N = prm.N; M = prm.M; Nt = prm.N / prm.M;

% Warm start: eta_rank = eta_b = 0
W0 = cell(K,1);
for k = 1:K
    W0{k} = eye(N) * (prm.Pmax / prm.K / prm.M);
end
b0 = ones(M, P) * (prm.N_req / M);

[sol0.W, sol0.Z, sol0.mu, sol0.b, sol0.M_p, sol0.status] = ...
    solve_p3_sca_t(prm, W0, b0, 0.0, 0.0);

if ~contains(sol0.status, 'Solved')
    res.status = 'initial_infeasible';
    return;
end

% Distance-based heuristic binary warm start
b_heur = zeros(M, P);
for p = 1:P
    dists = sqrt(sum((prm.AP_pos - prm.Target_pos(p,:)).^2, 2));
    [~, idx] = sort(dists, 'ascend');
    b_heur(idx(1:prm.N_req), p) = 1;
end

b_inits = {sol0.b, b_heur};
labels = {'relaxed', 'heuristic'};
fast_max_iter = 5;

for init_idx = 1:length(b_inits)
    b_prev = b_inits{init_idx};
    W_prev = sol0.W;
    Z_prev = sol0.Z;
    mu_prev = sol0.mu;
    M_p_prev = sol0.M_p;
    cur_eta_rank = eta_rank;
    cur_eta_b = eta_b;
    obj_trace = [];

    if verbose && length(b_inits) > 1
        fprintf('  -- warm start %s --\n', labels{init_idx});
    end

    % Phase 2: short DC loop (at most 5 iterations)
    for t = 1:min(T_max, fast_max_iter)
        [W_new, Z_new, mu_new, b_new, M_p_new, status] = ...
            solve_p3_sca_t(prm, W_prev, b_prev, cur_eta_rank, cur_eta_b);

        if ~contains(status, 'Solved')
            if verbose, fprintf('    iter %d: solver status = %s\n', t, status); end
            break;
        end

        true_obj = 0;
        for k = 1:K
            true_obj = true_obj + real(trace(W_new{k}));
        end
        true_obj = true_obj + real(trace(Z_new));
        obj_trace = [obj_trace, true_obj];

        rank_def = zeros(K,1);
        for k = 1:K
            d = eig(W_new{k}, 'vector');
            rank_def(k) = real(trace(W_new{k})) - max(d);
        end
        binary_dist = max( min(b_new(:), 1 - b_new(:)) );

        if verbose
            fprintf('  iter %d: true_obj=%.4f | max_rank_def=%.2e | bin_dist=%.2e\n', ...
                t, true_obj, max(rank_def), binary_dist);
        end

        W_prev = W_new;
        Z_prev = Z_new;
        b_prev = b_new;
        mu_prev = mu_new;
        M_p_prev = M_p_new;

        rank_ok = max(rank_def) < eps;
        bin_ok = binary_dist < eps;
        obj_ok = (length(obj_trace) > 1) && abs(obj_trace(end)-obj_trace(end-1)) < eps;
        if rank_ok && bin_ok && obj_ok
            % Converged: return immediately
            res = finalize_res(prm, W_prev, Z_prev, b_prev, mu_prev, M_p_prev, status, obj_trace, labels{init_idx});
            return;
        end

        cur_eta_rank = min(cur_eta_rank * eta_growth, 5.0);
        cur_eta_b = min(cur_eta_b * eta_growth, 1000.0);
    end

    % Phase 3: multi-candidate greedy rounding + single final convex solve
    if verbose
        fprintf('  DC not converged, trying multi-candidate rounding\n');
    end

    candidates = {};
    cand_labels = {};

    % Candidate 1: greedy top-N_req from relaxed b
    candidates{end+1} = greedy_round_b(b_prev, prm.N_req);
    cand_labels{end+1} = 'relaxed_topN';

    % Candidate 2: greedy top-(N_req+1) to add redundancy
    if prm.N_req + 1 <= M
        candidates{end+1} = greedy_round_b(b_prev, prm.N_req + 1);
        cand_labels{end+1} = 'relaxed_topN+1';
    end

    % Candidate 3: distance heuristic (already binary)
    candidates{end+1} = b_heur;
    cand_labels{end+1} = 'distance_heur';

    best_obj = inf;
    best_res = [];

    for c = 1:length(candidates)
        b_cand = candidates{c};
        try
            % Single final convex optimization with fixed binary b_cand and
            % no DC penalties (fast, no iterations)
            [W_fix, Z_fix, mu_fix, ~, M_p_fix, status_fix] = ...
                solve_p3_sca_t(prm, sol0.W, b_cand, 0.0, 0.0, b_cand);
            if contains(status_fix, 'Solved')
                res_cand = finalize_res(prm, W_fix, Z_fix, b_cand, mu_fix, M_p_fix, status_fix, [], cand_labels{c});
                if verbose
                    fprintf('    candidate %s: obj=%.4f, status=%s\n', cand_labels{c}, res_cand.final_obj, res_cand.status);
                end
                if res_cand.final_obj < best_obj || isempty(best_res)
                    best_res = res_cand;
                    best_res.status = [status_fix, ' (rounded_' cand_labels{c} ')'];
                    best_res.init_label = labels{init_idx};
                    best_obj = res_cand.final_obj;
                end
            end
        catch ME
            if verbose
                fprintf('    candidate %s failed: %s\n', cand_labels{c}, ME.message);
            end
        end
    end

    if ~isempty(best_res)
        res = best_res;
        return;
    end

    if verbose
        fprintf('  all rounding candidates infeasible for %s warm start\n', labels{init_idx});
    end
end

% All warm starts and all rounding candidates failed
res.status = 'infeasible_after_rounding';
res.final_obj = inf;
res.sum_rate = 0;
res.max_violation = inf;
res.W = [];
res.Z = [];
res.b = [];
res.mu = [];
res.M_p = [];
res.iters = 0;
res.obj_trace = [];
res.binary_converged = false;
res.init_label = 'none';

end

%% Local helpers
function b_rounded = greedy_round_b(b_relaxed, N_req)
[M, P] = size(b_relaxed);
b_rounded = zeros(M, P);
for p = 1:P
    [~, idx] = sort(b_relaxed(:, p), 'descend');
    b_rounded(idx(1:N_req), p) = 1;
end
end

function res = finalize_res(prm, W_prev, Z_prev, b_prev, mu_prev, M_p_prev, status, obj_trace, label)
K = prm.K;
% Extract physical rank-1 beams
w_star = cell(K,1);
W_phys = cell(K,1);
for k = 1:K
    [V, D] = eig(W_prev{k}, 'vector');
    [max_eig, idx] = max(D);
    w_star{k} = sqrt(max_eig) * V(:, idx);
    W_phys{k} = w_star{k} * w_star{k}';
end

[sum_rate, sens_sinr_db, pcrb] = evaluate(W_phys, Z_prev, b_prev, prm);
[max_viol, ~] = validate_solution(prm, W_phys, Z_prev, mu_prev, b_prev, M_p_prev, 1e-6);

final_obj = 0;
for k = 1:K
    final_obj = final_obj + real(trace(W_phys{k}));
end
final_obj = final_obj + real(trace(Z_prev));

res.status = status;
res.iters = length(obj_trace);
res.obj_trace = obj_trace;
res.binary_converged = true;
res.final_obj = final_obj;
res.sum_rate = sum_rate;
res.sens_sinr_db = sens_sinr_db;
res.pcrb = pcrb;
res.W = W_phys;
res.w_star = w_star;
res.Z = Z_prev;
res.b = b_prev;
res.mu = mu_prev;
res.M_p = M_p_prev;
res.max_violation = max_viol;
res.init_label = label;
end
