function res = solve_p3_with_fixed_b(prm, b_fixed, T_max, eps, eta_rank, eta_b, eta_growth)
%SOLVE_P3_WITH_FIXED_B  Fixed-assignment recovery with a fixed rank penalty.
%   This implements Baseline 1 (heuristic AP selection): b is fixed and only
%   the communication covariances W, target-specific sensing covariances S_p,
%   S-Procedure multipliers, and PCRB auxiliary matrices are optimized.

if nargin < 3, T_max = 80; end
if nargin < 4, eps = 1e-5; end
if nargin < 5, eta_rank = 1.0; end
if nargin < 6, eta_b = 1.0; end
if nargin < 7, eta_growth = 1.0; end %#ok<NASGU>

K = prm.K; P = prm.P; N = prm.N; M = prm.M; Nt = prm.N / prm.M;
assert(isequal(size(b_fixed), [M, P]), 'b_fixed must be M-by-P.');
assert(all(abs(b_fixed(:) - round(b_fixed(:))) <= 1e-10) && ...
       all(b_fixed(:) >= -1e-10) && all(b_fixed(:) <= 1 + 1e-10), ...
       'b_fixed must be binary.');

% Warm start
W0 = cell(K,1);
for k = 1:K
    W0{k} = eye(N) * (prm.Pmax / prm.K / M);  % match baseline_alg2 warm-start scaling
end

% b_fixed is passed as the optional sixth argument to solve_p3_sca_t.
prm_fixed = prm;
if isfield(prm, 'recovery_mosek_max_time')
    prm_fixed.mosek_max_time = prm.recovery_mosek_max_time;
end
[sol0.W, sol0.Z, sol0.mu, sol0.b, sol0.M_p, sol0.status, sol0.S_p] = ...
    solve_p3_sca_t(prm_fixed, W0, b_fixed, 0.0, 0.0, b_fixed);

if ~contains(sol0.status, 'Solved')
    res.status = 'initial_infeasible';
    return;
end

W_prev = sol0.W;
S_prev = sol0.S_p;
b_prev = sol0.b;          % should equal b_fixed
mu_prev = sol0.mu;
M_p_prev = sol0.M_p;
last_status = sol0.status;

true_obj_trace = [];
rank_residual_trace = [];
binary_converged = true;  % b is fixed, so binary convergence is trivial

for t = 1:T_max
    [W_new, ~, mu_new, b_new, M_p_new, status, S_new] = ...
        solve_p3_sca_t(prm_fixed, W_prev, b_fixed, eta_rank, eta_b, b_fixed);
    
    if ~contains(status, 'Solved')
        last_status = status;
        break;
    end
    last_status = status;
    
    base_power = 0;
    for k = 1:K
        base_power = base_power + real(trace(W_new{k}));
    end
    base_power = base_power + real(trace(sum(S_new, 3)));
    
    rank_def = zeros(K,1);
    for k = 1:K
        d = eig(W_new{k}, 'vector');
        rank_def(k) = real(trace(W_new{k})) - max(d);
    end
    
    rank_residual = sum(rank_def);
    true_obj_trace(end+1) = base_power + eta_rank * rank_residual;
    rank_residual_trace(end+1) = rank_residual;
    W_prev = W_new;
    S_prev = S_new;
    mu_prev = mu_new;
    M_p_prev = M_p_new;
    
    rank_ok = rank_residual < eps;
    obj_ok = numel(true_obj_trace) > 1 && ...
        abs(true_obj_trace(end)-true_obj_trace(end-1)) < eps;
    if rank_ok && obj_ok
        break;
    end
    
end

% Extract rank-1 physical beams
w_star = cell(K,1);
W_phys = cell(K,1);
for k = 1:K
    [V, D] = eig(W_prev{k}, 'vector');
    [max_eig, idx] = max(D);
    w_star{k} = sqrt(max_eig) * V(:, idx);
    W_phys{k} = w_star{k} * w_star{k}';
end

[sum_rate, sens_sinr_db, pcrb] = evaluate(W_phys, S_prev, b_prev, prm);
feas_tol = 1e-5;
[max_viol, viol_report] = validate_solution(prm, W_phys, S_prev, mu_prev, b_prev, M_p_prev, feas_tol);

final_obj = 0;
for k = 1:K
    final_obj = final_obj + real(trace(W_phys{k}));
end
for p = 1:P
    final_obj = final_obj + real(trace(S_prev(:,:,p)));
end

res.is_physical_feasible = max_viol <= feas_tol;
res.solver_status = last_status;
if res.is_physical_feasible
    res.status = last_status;
else
    res.status = sprintf('physical_solution_infeasible (max violation: %.3e)', max_viol);
end
res.iters = numel(true_obj_trace);
res.obj_trace = true_obj_trace;
res.true_obj_trace = true_obj_trace;
res.rank_residual_trace = rank_residual_trace;
res.binary_converged = binary_converged;
res.final_obj = final_obj;
res.sum_rate = sum_rate;
res.sens_sinr_db = sens_sinr_db;
res.pcrb = pcrb;
res.W = W_phys;
res.w_star = w_star;
res.S_p = S_prev;
res.Z = sum(S_prev, 3);  % aggregate sensing covariance for legacy consumers
res.b = b_prev;
res.mu = mu_prev;
res.M_p = M_p_prev;
res.max_violation = max_viol;
res.violation_report = viol_report;

end
