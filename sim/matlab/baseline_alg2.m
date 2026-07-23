function res = baseline_alg2(prm, T_max, eps, eta_rank, eta_b, eta_growth, verbose)
%BASELINE_ALG2  Algorithm 2: double-DC SCA with AP selection + rank-1 recovery
%
%   To avoid the symmetric saddle point where all APs are equally selected,
%   the relaxed AP pattern b is warm-started from both the uniform relaxed
%   solution and a distance-based heuristic. DC penalty iterations then
%   refine the binary pattern. No heuristic rounding is performed after the
%   DC loop.

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
best_res = [];
best_obj = inf;

for init_idx = 1:length(b_inits)
    b_prev = b_inits{init_idx};
    W_prev = sol0.W;
    Z_prev = sol0.Z;
    mu_prev = sol0.mu;
    M_p_prev = sol0.M_p;
    last_status = sol0.status;
    cur_eta_rank = eta_rank;
    cur_eta_b = eta_b;
    obj_trace = [];
    binary_converged = false;

    if verbose && length(b_inits) > 1
        fprintf('  -- warm start %s --\n', labels{init_idx});
    end

    for t = 1:T_max
        [W_new, Z_new, mu_new, b_new, M_p_new, status] = ...
            solve_p3_sca_t(prm, W_prev, b_prev, cur_eta_rank, cur_eta_b);

        if ~contains(status, 'Solved')
            if verbose, fprintf('    iter %d: solver status = %s\n', t, status); end
            break;
        end
        last_status = status;

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
        bin_def = sum(b_new(:) - b_new(:).^2);
        binary_dist = max( min(b_new(:), 1 - b_new(:)) );

        if verbose
            fprintf('  iter %d: true_obj=%.4f | max_rank_def=%.2e | bin_def=%.2e | bin_dist=%.2e\n', ...
                t, true_obj, max(rank_def), bin_def, binary_dist);
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
            binary_converged = true;
            break;
        end

        cur_eta_rank = min(cur_eta_rank * eta_growth, 5.0);
        cur_eta_b = min(cur_eta_b * eta_growth, 100.0);
    end

    if ~binary_converged
        last_status = [last_status, ' (binary_not_converged)'];
    end

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

    tmp_res.status = last_status;
    tmp_res.iters = length(obj_trace);
    tmp_res.obj_trace = obj_trace;
    tmp_res.binary_converged = binary_converged;
    tmp_res.final_obj = final_obj;
    tmp_res.sum_rate = sum_rate;
    tmp_res.sens_sinr_db = sens_sinr_db;
    tmp_res.pcrb = pcrb;
    tmp_res.W = W_phys;
    tmp_res.w_star = w_star;
    tmp_res.Z = Z_prev;
    tmp_res.b = b_prev;
    tmp_res.mu = mu_prev;
    tmp_res.M_p = M_p_prev;
    tmp_res.max_violation = max_viol;
    tmp_res.init_label = labels{init_idx};

    if binary_converged || final_obj < best_obj
        best_res = tmp_res;
        best_obj = final_obj;
    end
    if binary_converged
        break;  % no need to try other warm starts
    end
end

res = best_res;

end
