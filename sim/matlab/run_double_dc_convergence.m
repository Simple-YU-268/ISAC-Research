function result = run_double_dc_convergence(varargin)
%RUN_DOUBLE_DC_CONVERGENCE  Dedicated 30-iteration double-DC convergence run.
%   This diagnostic is intentionally separate from Monte Carlo trials, whose
%   binary-recovery phase is shortened for throughput. It records total power,
%   rank deficiency, and binary distance at every SCA iteration.

p = inputParser;
addParameter(p, 'T_max', 30, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Seed', 2026, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'N_req', 3, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'M', 8, @(x) isnumeric(x) && isscalar(x) && x >= 2);
addParameter(p, 'Nt', 4, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'K', 4, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'P', 2, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'AreaSize', 400, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Pmax_dBm', 20, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'Gamma_alpha', 3, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Solver', 'sdpt3', @(x) ischar(x) || isstring(x));
addParameter(p, 'Eta_rank0', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Eta_b0', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Eta_rank_growth', 1.3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'Eta_b_growth', 1.3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'Eta_rank_max', 5, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Eta_b_max', 1000, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Output_tag', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'Output_dir', fullfile(pwd, 'figures'), @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opt = p.Results;

prm = generate_scenario(opt.M, opt.Nt, opt.K, opt.P, 2, opt.Pmax_dBm, 'auto', ...
    'AreaSize', opt.AreaSize, 'N_req', opt.N_req, 'eps_h', 0.05, 'seed', opt.Seed, ...
    'Gamma_alpha', opt.Gamma_alpha);
prm.solver = char(opt.Solver);
K = prm.K; M = prm.M; P = prm.P; N = prm.N;
W_prev = cell(K, 1);
for k = 1:K
    W_prev{k} = eye(N) * (prm.Pmax / (K * M));
end
b_prev = ones(M, P) * (prm.N_req / M);

[W_prev, ~, ~, b_prev, ~, gate_status] = ...
    solve_p3_sca_t(prm, W_prev, b_prev, 0, 0);
if ~contains(gate_status, 'Solved')
    error('run_double_dc_convergence:InitialInfeasible', ...
        'Continuous relaxation is infeasible: %s', gate_status);
end

eta_rank = opt.Eta_rank0;
eta_b = opt.Eta_b0;
result.power = NaN(opt.T_max, 1);
result.rank_deficiency = NaN(opt.T_max, 1);
result.binary_distance = NaN(opt.T_max, 1);
result.eta_rank = NaN(opt.T_max, 1);
result.eta_b = NaN(opt.T_max, 1);
result.status = strings(opt.T_max, 1);

for t = 1:opt.T_max
    [W_new, Z_new, ~, b_new, ~, status] = ...
        solve_p3_sca_t(prm, W_prev, b_prev, eta_rank, eta_b);
    result.status(t) = string(status);
    if ~contains(status, 'Solved')
        break;
    end
    result.power(t) = sum(cellfun(@(W) real(trace(W)), W_new)) + real(trace(Z_new));
    result.rank_deficiency(t) = max(cellfun(@(W) ...
        max(0, real(trace(W)) - max(real(eig(W, 'vector')))), W_new));
    result.binary_distance(t) = max(min(b_new(:), 1 - b_new(:)));
    result.eta_rank(t) = eta_rank;
    result.eta_b(t) = eta_b;
    fprintf('iter=%d, power=%.6g, rank=%.3e, binary=%.3e\n', ...
        t, result.power(t), result.rank_deficiency(t), result.binary_distance(t));
    W_prev = W_new;
    b_prev = b_new;
    eta_rank = min(eta_rank * opt.Eta_rank_growth, opt.Eta_rank_max);
    eta_b = min(eta_b * opt.Eta_b_growth, opt.Eta_b_max);
end

valid = find(isfinite(result.power));
result.iterations = numel(valid);
result.prm = prm;
if ~exist(opt.Output_dir, 'dir'), mkdir(opt.Output_dir); end
tag = sprintf('double_dc_convergence_seed%d_nreq%d', opt.Seed, opt.N_req);
if strlength(string(opt.Output_tag)) > 0
    tag = sprintf('%s_%s', tag, char(opt.Output_tag));
end
plot_double_dc_convergence(result, fullfile(opt.Output_dir, tag));
save(fullfile(opt.Output_dir, [tag, '.mat']), 'result');
end
