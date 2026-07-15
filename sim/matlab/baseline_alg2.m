function res = baseline_alg2(prm, T_max, eps, eta_rank, eta_b, eta_growth, verbose)
%BASELINE_ALG2  Algorithm 2: double-DC SCA with AP selection + rank-1 recovery
%
%   Strictly no heuristic rounding: binary variables b must be driven to
%   {0,1} by the DC penalty alone.  If convergence to binary is not
%   achieved within T_max, the routine returns the relaxed solution and
%   a non-binary status.

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

W_prev = sol0.W;
Z_prev = sol0.Z;
b_prev = sol0.b;
mu_prev = sol0.mu;
M_p_prev = sol0.M_p;
last_status = sol0.status;

cur_eta_rank = eta_rank;
cur_eta_b = eta_b;
obj_trace = [];
binary_converged = false;
for t = 1:T_max
    [W_new, Z_new, mu_new, b_new, M_p_new, status] = ...
        solve_p3_sca_t(prm, W_prev, b_prev, cur_eta_rank, cur_eta_b);

    if ~contains(status, 'Solved')
        if verbose, fprintf('  iter %d: solver status = %s\n', t, status); end
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
    binary_dist = max( min(b_new(:), 1 - b_new(:)) );  % distance to nearest binary

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

% No heuristic rounding: b is the relaxed solution from the DC-SCA loop.
% If not binary-converged, append a warning to status.
if ~binary_converged
    last_status = [last_status, ' (binary_not_converged)'];
end

% Evaluate and validate
[sum_rate, sens_sinr_db, pcrb] = evaluate(W_prev, Z_prev, b_prev, prm);
[max_viol, ~] = validate_solution(prm, W_prev, Z_prev, mu_prev, b_prev, M_p_prev, 1e-6);

final_obj = 0;
for k = 1:K
    final_obj = final_obj + real(trace(W_prev{k}));
end
final_obj = final_obj + real(trace(Z_prev));

res.status = last_status;
res.iters = length(obj_trace);
res.obj_trace = obj_trace;
res.binary_converged = binary_converged;
res.final_obj = final_obj;
res.sum_rate = sum_rate;
res.sens_sinr_db = sens_sinr_db;
res.pcrb = pcrb;
res.W = W_prev;
res.Z = Z_prev;
res.b = b_prev;
res.mu = mu_prev;
res.M_p = M_p_prev;
res.max_violation = max_viol;

end

function [sum_rate, sens_sinr_db, pcrb] = evaluate(W, Z, b, prm)
K = prm.K; P = prm.P; N = prm.N;
R = zeros(N, N);
for k = 1:K
    R = R + W{k};
end
R = R + Z;

sinr = zeros(K, 1);
for k = 1:K
    hk = prm.H(:, k);
    sig = real(hk' * W{k} * hk);
    interf = 0;
    for j = setdiff(1:K, k)
        interf = interf + real(hk' * W{j} * hk);
    end
    sinr(k) = sig / (interf + prm.sigma_c2);
end
sum_rate = sum(log2(1 + sinr));

sens_sinr_db = zeros(P, 1);
for p = 1:P
    gp = prm.G(:, p);
    sens_sinr_db(p) = 10 * log10(max(real(gp' * Z * gp), 1e-30) / prm.sigma_s2);
end

pcrb = zeros(P, 1);
for p = 1:P
    gp = prm.G(:, p);
    Jp = real(gp' * R * gp) / prm.sigma_s2;
    if Jp > 1e-9
        pcrb(p) = 1 / Jp;
    else
        pcrb(p) = inf;
    end
end
end
