function experiments_paper()
%EXPERIMENTS_PAPER  Small-scale Monte Carlo experiments for the Cell-Free ISAC paper
%   Generates four core figures: convergence, Pareto frontier, robustness, and
%   AP assignment effectiveness. Uses generate_scenario + baseline_alg2.
%
%   Run time for the default small-scale setting (N_mc=20) is ~10-30 minutes
%   on a single core. Increase N_workers and N_mc for final paper figures.

%% 0. Common settings ------------------------------------------------------
M = 12; Nt = 4; K = 4; P = 2; N_theta = 2;          % memory-safe large scale (16 GB M4)
Pmax_dBm = 20;                                      % per-AP power (dBm)
Gamma_track = 'auto';                                % physical, per-target PCRB reference
N_mc = 100;                                          % large-scale Monte Carlo trials
N_workers = 2;                                       % conservative parallel workers to avoid OOM

save_tag = sprintf('M%d_Nt%d_K%d_P%d_Nth%d_mc%d', M, Nt, K, P, N_theta, N_mc);
out_dir = fullfile(pwd, 'figures');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% 1. Figure 1: Convergence of a single representative channel realization
fprintf('=== Figure 1: Convergence ===\n');
prm = generate_scenario(M, Nt, K, P, N_theta, Pmax_dBm, Gamma_track, ...
    'AreaSize', 400, 'N_req', 2, 'eps_h', 0.05, 'seed', 2026);
res = baseline_alg2(prm, 30, 1e-5, 1.0, 1.0, 1.3, true);

figure('Name', 'Fig1_Convergence');
plot(1:length(res.obj_trace), res.obj_trace, 'b-o', 'LineWidth', 1.5);
grid on;
xlabel('SCA iteration $t$', 'Interpreter', 'latex');
ylabel('Total transmit power', 'Interpreter', 'latex');
title('Convergence of double-DC SCA');
saveas(gcf, fullfile(out_dir, [save_tag, '_fig1_convergence.png']));

%% 2. Figure 2: Pareto frontier (sum rate vs total power for varying Gamma_track)
fprintf('=== Figure 2: Pareto Frontier ===\n');
Gamma_scale_list = [1, 1.25, 1.5, 2, 3, 5];
prm_ref = generate_scenario(M, Nt, K, P, N_theta, Pmax_dBm, 'auto', ...
    'AreaSize', 400, 'N_req', 2, 'eps_h', 0.05, 'seed', 2026);
Gamma_list = Gamma_scale_list;  % multipliers of the physical reference threshold
pareto_power = nan(length(Gamma_list), 1);
pareto_rate  = nan(length(Gamma_list), 1);
for i = 1:length(Gamma_list)
    fprintf('  Gamma scale = %.2f ...\n', Gamma_scale_list(i));
    prm = generate_scenario(M, Nt, K, P, N_theta, Pmax_dBm, ...
        Gamma_scale_list(i) * prm_ref.Gamma_track, ...
        'AreaSize', 400, 'N_req', 2, 'eps_h', 0.05, 'seed', 2026);
    res = baseline_alg2(prm, 30, 1e-5, 1.0, 1.0, 1.3, false);
    if contains(res.status, 'Solved')
        pareto_power(i) = res.final_obj;
        pareto_rate(i)  = res.sum_rate;
    end
end

figure('Name', 'Fig2_Pareto');
plot(pareto_power, pareto_rate, 'rs-', 'LineWidth', 1.5, 'MarkerSize', 8);
grid on;
xlabel('Total transmit power', 'Interpreter', 'latex');
ylabel('Sum rate [bit/s/Hz]', 'Interpreter', 'latex');
title('Communication-sensing trade-off');
saveas(gcf, fullfile(out_dir, [save_tag, '_fig2_pareto.png']));

%% 3. Figure 3: Robustness vs CSI error (Monte Carlo)
fprintf('=== Figure 3: Robustness Monte Carlo ===\n');
eps_list = [0, 0.02, 0.04, 0.06, 0.08, 0.10];
N_eps = length(eps_list);
rob_power = nan(N_eps, 1);
rob_power_std = nan(N_eps, 1);
rob_outage = nan(N_eps, 1);
base_outage = nan(N_eps, 1);
base_power = nan(N_eps, 1);

for i = 1:N_eps
    fprintf('  eps_h = %.2f ...\n', eps_list(i));
    prm_nom = generate_scenario(M, Nt, K, P, N_theta, Pmax_dBm, Gamma_track, ...
        'AreaSize', 400, 'N_req', 2, 'eps_h', eps_list(i), 'seed', 2026);

    [rob_power(i), rob_power_std(i), rob_outage(i)] = ...
        mc_run(prm_nom, N_mc, N_workers, 'proposed', 'eval_eps', eps_list(i));

    % Non-robust baseline: design with eps=0, evaluate with true random errors
    prm_nonrob = generate_scenario(M, Nt, K, P, N_theta, Pmax_dBm, Gamma_track, ...
        'AreaSize', 400, 'N_req', 2, 'eps_h', 0, 'seed', 2026);
    [base_power(i), ~, base_outage(i)] = mc_run(prm_nonrob, N_mc, N_workers, ...
        'nonrobust', 'eval_eps', eps_list(i));
end

figure('Name', 'Fig3_Robustness');
subplot(1,2,1);
errorbar(eps_list, rob_power, rob_power_std, 'b-o', 'LineWidth', 1.5); hold on;
plot(eps_list, base_power, 'r--s', 'LineWidth', 1.5);
grid on;
xlabel('Normalized CSI error $\epsilon_h$', 'Interpreter', 'latex');
ylabel('Total transmit power', 'Interpreter', 'latex');
legend('Proposed', 'Non-robust', 'Location', 'northwest');
title('Power vs CSI error');

subplot(1,2,2);
plot(eps_list, rob_outage, 'b-o', 'LineWidth', 1.5); hold on;
plot(eps_list, base_outage, 'r--s', 'LineWidth', 1.5);
grid on;
xlabel('Normalized CSI error $\epsilon_h$', 'Interpreter', 'latex');
ylabel('Outage probability', 'Interpreter', 'latex');
legend('Proposed', 'Non-robust', 'Location', 'northwest');
title('SINR outage vs CSI error');
saveas(gcf, fullfile(out_dir, [save_tag, '_fig3_robustness.png']));

%% 4. Figure 4: AP assignment effectiveness vs N_req
fprintf('=== Figure 4: AP Assignment ===\n');
N_req_list = 1:min(6, M);
assign_power_proposed = nan(length(N_req_list), 1);
assign_power_heuristic = nan(length(N_req_list), 1);
for i = 1:length(N_req_list)
    fprintf('  N_req = %d ...\n', N_req_list(i));
    prm = generate_scenario(M, Nt, K, P, N_theta, Pmax_dBm, Gamma_track, ...
        'AreaSize', 400, 'N_req', N_req_list(i), 'eps_h', 0.05, 'seed', 2026);

    [assign_power_proposed(i), ~, ~] = mc_run(prm, N_mc, N_workers, 'proposed');
    [assign_power_heuristic(i), ~, ~] = mc_run(prm, N_mc, N_workers, 'heuristic_b');
end

figure('Name', 'Fig4_AP_Assignment');
plot(N_req_list, assign_power_proposed, 'b-o', 'LineWidth', 1.5); hold on;
plot(N_req_list, assign_power_heuristic, 'r--s', 'LineWidth', 1.5);
grid on;
xlabel('Required APs per target $N_{\\rm req}$', 'Interpreter', 'latex');
ylabel('Total transmit power', 'Interpreter', 'latex');
legend('Proposed (joint)', 'Heuristic nearest-AP', 'Location', 'best');
title('AP assignment effectiveness');
saveas(gcf, fullfile(out_dir, [save_tag, '_fig4_ap_assignment.png']));

%% Save numeric results
save(fullfile(out_dir, [save_tag, '_results.mat']), ...
    'Gamma_list', 'pareto_power', 'pareto_rate', ...
    'eps_list', 'rob_power', 'rob_power_std', 'rob_outage', 'base_power', 'base_outage', ...
    'N_req_list', 'assign_power_proposed', 'assign_power_heuristic');

fprintf('All figures and results saved to: %s\n', out_dir);

end

%% -------------------------------------------------------------------------
function [avg_power, std_power, outage] = mc_run(prm, N_mc, N_workers, mode, varargin)
%MC_RUN  Monte Carlo wrapper for a given mode
%   mode: 'proposed', 'heuristic_b', 'nonrobust', 'comm_only', 'sensing_only'
%   Optional 'eval_eps' gives the true CSI error used to evaluate every design.

p = inputParser;
addParameter(p, 'eval_eps', 0, @isnumeric);
parse(p, varargin{:});
eval_eps = p.Results.eval_eps;

results = cell(N_mc, 1);

if N_workers > 0 && isempty(gcp('nocreate'))
    parpool('local', N_workers);
end

if N_workers > 0
    parfor n = 1:N_mc
        results{n} = run_single_trial(prm, mode, eval_eps, n);
    end
else
    for n = 1:N_mc
        results{n} = run_single_trial(prm, mode, eval_eps, n);
    end
end

powers = cellfun(@(r) r.power, results);
outages = cellfun(@(r) r.outage, results);
valid = ~isnan(powers);
if sum(valid) == 0
    avg_power = NaN; std_power = NaN; outage = NaN;
else
    avg_power = mean(powers(valid));
    std_power = std(powers(valid));
    outage = mean(outages(valid));
end
end

function rec = run_single_trial(prm0, mode, eval_eps, n)
%RUN_SINGLE_TRIAL  One MC trial; draw a fresh channel realization

seed = prm0.seed + n;

% Re-draw scenario with same structural parameters but different random seed
prm = generate_scenario(prm0.M, prm0.Nt, prm0.K, prm0.P, prm0.N_theta, ...
    10*log10(prm0.Pmax) + 30, gamma_input(prm0), ...
    'AreaSize', 400, 'N_req', prm0.N_req, 'eps_h', prm0.eps_h, ...
    'sigma_c2', prm0.sigma_c2, 'sigma_s2', prm0.sigma_s2, 'seed', seed);

rec.power = NaN;
rec.outage = 0;

switch mode
    case 'proposed'
        res = baseline_alg2(prm, 30, 1e-5, 1.0, 1.0, 1.3, false);

    case 'heuristic_b'
        b = heuristic_b(prm);
        res = solve_p3_with_fixed_b(prm, b, 30, 1e-5, 1.0, 1.0, 1.3);

    case 'nonrobust'
        res = baseline_alg2(prm, 20, 1e-5, 1.0, 1.0, 1.3, false);

    case 'comm_only'
        prm.Gamma_track = 1e6 * ones(prm.P, 1);
        prm.enable_sensing_sinr = false;
        prm.enable_pcrb = false;
        b = heuristic_b(prm);
        res = solve_p3_with_fixed_b(prm, b, 30, 1e-5, 1.0, 1.0, 1.3);

    case 'sensing_only'
        prm.gamma_k = 1e-6 * ones(prm.K, 1);
        b = heuristic_b(prm);
        res = solve_p3_with_fixed_b(prm, b, 30, 1e-5, 1.0, 1.0, 1.3);

    otherwise
        error('Unknown mode: %s', mode);
end

if ~contains(res.status, 'Solved')
    return;
end

rec.power = res.final_obj;

% Evaluate every design under the same true random channel error model.
% This is necessary for a meaningful robust-vs-non-robust outage comparison.
if eval_eps > 0
    N_err_samples = 1000;  % per-UE samples for outage probability
    outage_count = 0;
    total_count = 0;
    for k = 1:prm.K
        hk = prm.H(:, k);
        radius = eval_eps * norm(hk);
        N_dim = length(hk);
        for s = 1:N_err_samples
            % Uniform sample inside complex hypersphere of radius = eval_eps * norm(hk)
            delta = (randn(N_dim, 1) + 1j * randn(N_dim, 1)) / sqrt(2);
            delta_dir = delta / norm(delta);
            u = rand();
            r = radius * (u ^ (1 / (2 * N_dim)));
            delta_h = r * delta_dir;
            hk_true = hk + delta_h;

            sig = real(hk_true' * res.W{k} * hk_true);
            interf = 0;
            for j = setdiff(1:prm.K, k)
                interf = interf + real(hk_true' * res.W{j} * hk_true);
            end
            sinr_true = sig / (interf + prm.sigma_c2);
            if sinr_true < prm.gamma_k(k)
                outage_count = outage_count + 1;
            end
            total_count = total_count + 1;
        end
    end
    rec.outage = outage_count / total_count;
else
    rec.outage = 0;
end
end

function gamma = gamma_input(prm)
if isfield(prm, 'gamma_track_auto') && prm.gamma_track_auto
    gamma = 'auto';
else
    gamma = prm.Gamma_track;
end
end

function b = heuristic_b(prm)
%HEURISTIC_B  Nearest-AP binary assignment for each target
M = prm.M; P = prm.P;
b = zeros(M, P);
for p = 1:P
    dists = sqrt(sum((prm.AP_pos - prm.Target_pos(p,:)).^2, 2));
    [~, idx] = sort(dists, 'ascend');
    b(idx(1:prm.N_req), p) = 1;
end
end
