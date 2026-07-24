function res = baseline_alg2(prm, T_max, eps, eta_rank, eta_b, eta_growth, verbose)
%BASELINE_ALG2  Fast DC-SCA with valid binary topology recovery.
%   A short double-DC phase generates candidate associations.  Every
%   candidate satisfies sum_m b_mp=N_req exactly and is accepted only after
%   fixed-b re-optimization and rank-one physical feasibility validation.

if nargin < 2, T_max = 80; end
if nargin < 3, eps = 1e-5; end
if nargin < 4, eta_rank = 1.0; end
if nargin < 5, eta_b = 1.0; end
if nargin < 6, eta_growth = 1.3; end
if nargin < 7, verbose = false; end

K = prm.K; P = prm.P; N = prm.N; M = prm.M;
W_init = cell(K,1);
for k = 1:K
    W_init{k} = eye(N) * (prm.Pmax / prm.K / prm.M);
end
b0 = ones(M,P) * (prm.N_req / M);
[W_sdr, ~, ~, b_sdr, ~, status0] = solve_p3_sca_t(prm, W_init, b0, 0, 0);
if ~contains(status0, 'Solved')
    res = struct('status', 'initial_infeasible');
    return;
end

b_heur = nearest_assignment(prm);
b_inits = {b_sdr, b_heur};
labels = {'relaxed', 'distance_heur'};
fast_max_iter = min(T_max, 5);
best_res = [];
best_obj = inf;
failed_candidates = {};

for init_idx = 1:numel(b_inits)
    W_prev = W_sdr;
    b_prev = b_inits{init_idx};
    cur_eta_rank = eta_rank;
    cur_eta_b = eta_b;
    obj_trace = [];

    for t = 1:fast_max_iter
        [W_new, Z_new, ~, b_new, ~, status] = ...
            solve_p3_sca_t(prm, W_prev, b_prev, cur_eta_rank, cur_eta_b);
        if ~contains(status, 'Solved'), break; end
        obj_trace(end+1,1) = sum(cellfun(@(W) real(trace(W)), W_new)) + real(trace(Z_new));
        rank_def = max(cellfun(@(W) real(trace(W)) - max(eig(W,'vector')), W_new));
        bin_dist = max(min(b_new(:), 1-b_new(:)));
        if verbose
            fprintf('  %s %d: obj=%.4f, rank=%.2e, bin=%.2e\n', ...
                labels{init_idx}, t, obj_trace(end), rank_def, bin_dist);
        end
        W_prev = W_new;
        b_prev = b_new;
        if rank_def < eps && bin_dist < eps && numel(obj_trace)>1 ...
                && abs(obj_trace(end)-obj_trace(end-1)) < eps
            break;
        end
        cur_eta_rank = min(cur_eta_rank * eta_growth, 5);
        cur_eta_b = min(cur_eta_b * eta_growth, 1000);
    end

    % Both candidates are valid M-by-P binary matrices with exactly N_req
    % selected APs per active target.  Do not use top-(N_req+1): it violates C6.
    candidates = {round_assignment(b_prev, prm), b_heur};
    cand_labels = {'dc_topN', 'distance_heur'};
    for c = 1:numel(candidates)
        b_fixed = candidates{c};
        fixed_res = solve_p3_with_fixed_b(prm, b_fixed, ...
            min(T_max, 10), eps, eta_rank, 0, eta_growth);
        if isfield(fixed_res,'is_physical_feasible') && fixed_res.is_physical_feasible
            fixed_res.obj_trace_dc = obj_trace;
            fixed_res.binary_converged = true;
            fixed_res.init_label = labels{init_idx};
            fixed_res.rounding_label = cand_labels{c};
            if fixed_res.final_obj < best_obj
                best_res = fixed_res;
                best_obj = fixed_res.final_obj;
            end
        else
            diag.b = b_fixed;
            diag.init_label = labels{init_idx};
            diag.rounding_label = cand_labels{c};
            diag.full_status = fixed_res.status;
            diag.pcrb_only = check_fixed_b_pcrb(prm, b_fixed, false);
            diag.pcrb_sensing = check_fixed_b_pcrb(prm, b_fixed, true);
            failed_candidates{end+1,1} = diag;
        end
    end
end

if isempty(best_res)
    res = struct('status','infeasible_after_rounding', ...
        'failed_candidates',{failed_candidates});
else
    res = best_res;
end
end

function b = nearest_assignment(prm)
b = zeros(prm.M, prm.P);
for p = prm.active_targets
    d = sqrt(sum((prm.AP_pos-prm.Target_pos(p,:)).^2,2));
    [~,idx] = sort(d,'ascend');
    b(idx(1:prm.N_req),p) = 1;
end
end

function b = round_assignment(b_relaxed, prm)
b = zeros(prm.M,prm.P);
for p = prm.active_targets
    [~,idx] = sort(b_relaxed(:,p),'descend');
    b(idx(1:prm.N_req),p) = 1;
end
end
