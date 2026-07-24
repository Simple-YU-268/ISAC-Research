function experiments_paper(varargin)
%EXPERIMENTS_PAPER  Feasibility-aware Monte Carlo study for Cell-Free ISAC.
%   The binary AP-target association controls a network-wide AP activation
%   gate.  Consequently an infeasible topology is a physical outcome, not a
%   missing power sample.  This script therefore reports feasibility first and
%   reports power, communication, sensing, and activation cost only
%   conditionally on feasible binary solutions.
%
%   Examples:
%     experiments_paper('Quick', true)       % 3 paired seeds, smoke study
%     experiments_paper('N_mc', 100)         % final Monte Carlo campaign
%
%   All N_req values share the same seed set, channel model, power budget, and
%   auto-calibrated Gamma reference.  The latter is independent of N_req.

p = inputParser;
addParameter(p, 'Quick', false, @islogical);
addParameter(p, 'N_mc', 30, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'N_workers', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'Base_seed', 2026, @(x) isnumeric(x) && isscalar(x));
parse(p, varargin{:});
opt = p.Results;
if opt.Quick, opt.N_mc = 3; end

%% Common configuration ----------------------------------------------------
cfg.M = 8; cfg.Nt = 4; cfg.K = 4; cfg.P = 2; cfg.N_theta = 2;
cfg.Pmax_dBm = 20;
cfg.AreaSize = 400;
cfg.eps_h = 0.05;
cfg.Gamma_track = 'auto';
cfg.N_req_main = 3;                 % validated representative operating point
cfg.N_req_list = 2:min(5, cfg.M);  % study the physical feasibility boundary
cfg.T_max = 30;
cfg.eps = 1e-5;
cfg.eta_rank = 1.0;
cfg.eta_b = 1.0;
cfg.eta_growth = 1.3;
cfg.N_mc = opt.N_mc;
cfg.N_workers = opt.N_workers;
cfg.Base_seed = opt.Base_seed;

out_dir = fullfile(pwd, 'figures');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save_tag = sprintf('feasibility_M%d_K%d_P%d_mc%d', cfg.M, cfg.K, cfg.P, cfg.N_mc);

%% Figure 1: representative convergence at a feasible operating point -----
fprintf('=== Figure 1: representative convergence (N_req=%d) ===\n', cfg.N_req_main);
prm = make_scenario(cfg, cfg.N_req_main, cfg.eps_h, cfg.Base_seed);
res = baseline_alg2(prm, cfg.T_max, cfg.eps, cfg.eta_rank, ...
    cfg.eta_b, cfg.eta_growth, true);
if contains(res.status, 'Solved') && isfield(res, 'obj_trace_dc')
    figure('Name', 'Fig1_Convergence');
    plot(1:numel(res.obj_trace_dc), res.obj_trace_dc, 'b-o', 'LineWidth', 1.5);
    grid on;
    xlabel('DC-SCA iteration'); ylabel('Total transmit power [W]');
    title(sprintf('Representative convergence (N_{req}=%d)', cfg.N_req_main));
    saveas(gcf, fullfile(out_dir, [save_tag, '_fig1_convergence.png']));
else
    warning('Representative N_req=%d realization was infeasible; no convergence plot.', ...
        cfg.N_req_main);
end

%% Figures 2--4: paired feasibility and conditional performance vs N_req ---
fprintf('=== Figures 2--4: paired N_req feasibility study ===\n');
nreq_result = struct();
for i = 1:numel(cfg.N_req_list)
    nreq = cfg.N_req_list(i);
    fprintf('  N_req=%d: proposed ...\n', nreq);
    nreq_result.proposed(i) = mc_run(cfg, nreq, 'proposed', cfg.eps_h, cfg.eps_h);
    fprintf('  N_req=%d: nearest-AP baseline ...\n', nreq);
    nreq_result.nearest(i) = mc_run(cfg, nreq, 'nearest', cfg.eps_h, cfg.eps_h);
end

prop_feas = [nreq_result.proposed.feasibility];
near_feas = [nreq_result.nearest.feasibility];
prop_power = [nreq_result.proposed.conditional_power];
near_power = [nreq_result.nearest.conditional_power];
prop_std = [nreq_result.proposed.conditional_power_std];
near_std = [nreq_result.nearest.conditional_power_std];

figure('Name', 'Fig2_FeasibilityVsNreq');
plot(cfg.N_req_list, prop_feas, 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, near_feas, 'r--s', 'LineWidth', 1.5);
grid on; ylim([0 1]);
xlabel('Required APs per target N_{req}'); ylabel('Binary physical feasibility rate');
legend('DC-SCA + UE-aware fixed-b recovery', 'Nearest-AP fixed-b', 'Location', 'southeast');
title('Feasibility versus AP-association cardinality');
saveas(gcf, fullfile(out_dir, [save_tag, '_fig2_feasibility_vs_nreq.png']));

figure('Name', 'Fig3_ConditionalPowerVsNreq');
errorbar(cfg.N_req_list, prop_power, prop_std, 'b-o', 'LineWidth', 1.5); hold on;
errorbar(cfg.N_req_list, near_power, near_std, 'r--s', 'LineWidth', 1.5);
grid on;
xlabel('Required APs per target N_{req}');
ylabel('Transmit power [W] | feasible');
legend('DC-SCA + UE-aware fixed-b recovery', 'Nearest-AP fixed-b', 'Location', 'best');
title('Conditional power: infeasible trials are excluded explicitly');
saveas(gcf, fullfile(out_dir, [save_tag, '_fig3_conditional_power_vs_nreq.png']));

% Communication, sensing, and cooperation outcomes must be read together
% with Fig. 2. They are conditional metrics, not substitutes for feasibility.
figure('Name', 'Fig4_ConditionalPerformanceVsNreq');
subplot(2,2,1);
plot(cfg.N_req_list, [nreq_result.proposed.conditional_rate], 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, [nreq_result.nearest.conditional_rate], 'r--s', 'LineWidth', 1.5);
grid on; xlabel('N_{req}'); ylabel('Sum rate [bit/s/Hz] | feasible');
legend('UE-aware recovery', 'Nearest-AP', 'Location', 'best');

subplot(2,2,2);
plot(cfg.N_req_list, [nreq_result.proposed.conditional_pcrb], 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, [nreq_result.nearest.conditional_pcrb], 'r--s', 'LineWidth', 1.5);
grid on; xlabel('N_{req}'); ylabel('Mean PCRB trace | feasible');

subplot(2,2,3);
plot(cfg.N_req_list, [nreq_result.proposed.conditional_sensing_sinr_db], 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, [nreq_result.nearest.conditional_sensing_sinr_db], 'r--s', 'LineWidth', 1.5);
grid on; xlabel('N_{req}'); ylabel('Mean sensing SINR [dB] | feasible');

subplot(2,2,4);
plot(cfg.N_req_list, [nreq_result.proposed.conditional_active_aps], 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, [nreq_result.nearest.conditional_active_aps], 'r--s', 'LineWidth', 1.5);
grid on; xlabel('N_{req}'); ylabel('Activated APs | feasible');
saveas(gcf, fullfile(out_dir, [save_tag, '_fig4_conditional_performance_vs_nreq.png']));

%% Figure 5: robustness at the selected feasible operating point -----------
fprintf('=== Figure 5: robustness at N_req=%d ===\n', cfg.N_req_main);
eps_list = [0, 0.02, 0.05, 0.08];
robust = struct();
for i = 1:numel(eps_list)
    fprintf('  design/evaluation eps_h=%.2f: robust ...\n', eps_list(i));
    robust.proposed(i) = mc_run(cfg, cfg.N_req_main, 'proposed', eps_list(i), eps_list(i));
    fprintf('  design eps_h=0, evaluation eps_h=%.2f: non-robust ...\n', eps_list(i));
    robust.nonrobust(i) = mc_run(cfg, cfg.N_req_main, 'nonrobust', 0, eps_list(i));
end

figure('Name', 'Fig5_Robustness');
subplot(1,2,1);
plot(eps_list, [robust.proposed.feasibility], 'b-o', 'LineWidth', 1.5); hold on;
plot(eps_list, [robust.nonrobust.feasibility], 'r--s', 'LineWidth', 1.5);
grid on; ylim([0 1]);
xlabel('CSI uncertainty radius \epsilon_h'); ylabel('Design feasibility rate');
legend('Robust design', 'Non-robust design', 'Location', 'southwest');

subplot(1,2,2);
plot(eps_list, [robust.proposed.conditional_outage], 'b-o', 'LineWidth', 1.5); hold on;
plot(eps_list, [robust.nonrobust.conditional_outage], 'r--s', 'LineWidth', 1.5);
grid on;
xlabel('True CSI uncertainty radius \epsilon_h'); ylabel('Outage probability | feasible');
legend('Robust design', 'Non-robust design', 'Location', 'northwest');
saveas(gcf, fullfile(out_dir, [save_tag, '_fig5_robustness.png']));

%% Save all statistics, including denominators and infeasible trials -------
save(fullfile(out_dir, [save_tag, '_results.mat']), 'cfg', 'nreq_result', 'robust', 'eps_list');
summary = make_summary_table(cfg.N_req_list, nreq_result);
writetable(summary, fullfile(out_dir, [save_tag, '_nreq_summary.csv']));
fprintf('Results saved to: %s\n', out_dir);
end

function stats = mc_run(cfg, nreq, mode, design_eps, eval_eps)
records = cell(cfg.N_mc, 1);
if cfg.N_workers > 0 && isempty(gcp('nocreate'))
    parpool('local', cfg.N_workers);
end
if cfg.N_workers > 0
    parfor n = 1:cfg.N_mc
        records{n} = run_trial(cfg, nreq, mode, design_eps, eval_eps, n);
    end
else
    for n = 1:cfg.N_mc
        records{n} = run_trial(cfg, nreq, mode, design_eps, eval_eps, n);
    end
end

is_feasible = cellfun(@(r) r.feasible, records);
stats.N_mc = cfg.N_mc;
stats.n_feasible = sum(is_feasible);
stats.feasibility = mean(is_feasible);
stats.records = records;
if ~any(is_feasible)
    stats.conditional_power = NaN;
    stats.conditional_power_std = NaN;
    stats.conditional_rate = NaN;
    stats.conditional_pcrb = NaN;
    stats.conditional_sensing_sinr_db = NaN;
    stats.conditional_outage = NaN;
    stats.conditional_active_aps = NaN;
    return;
end
valid = records(is_feasible);
stats.conditional_power = mean(cellfun(@(r) r.power, valid));
stats.conditional_power_std = std(cellfun(@(r) r.power, valid));
stats.conditional_rate = mean(cellfun(@(r) r.sum_rate, valid));
stats.conditional_pcrb = mean(cellfun(@(r) mean(r.pcrb), valid));
stats.conditional_sensing_sinr_db = mean(cellfun(@(r) mean(r.sens_sinr_db), valid));
stats.conditional_outage = mean(cellfun(@(r) r.outage, valid));
stats.conditional_active_aps = mean(cellfun(@(r) r.active_aps, valid));
end

function rec = run_trial(cfg, nreq, mode, design_eps, eval_eps, index)
prm = make_scenario(cfg, nreq, design_eps, cfg.Base_seed + index);
rec.feasible = false;
rec.status = '';
rec.power = NaN; rec.sum_rate = NaN; rec.pcrb = NaN; rec.sens_sinr_db = NaN;
rec.outage = NaN; rec.active_aps = NaN;

switch mode
    case 'proposed'
        res = baseline_alg2(prm, cfg.T_max, cfg.eps, cfg.eta_rank, ...
            cfg.eta_b, cfg.eta_growth, false);
    case 'nearest'
        res = solve_p3_with_fixed_b(prm, nearest_assignment(prm), ...
            cfg.T_max, cfg.eps, cfg.eta_rank, cfg.eta_b, cfg.eta_growth);
    case 'nonrobust'
        % design_eps is zero for this mode; evaluation still uses eval_eps.
        res = baseline_alg2(prm, cfg.T_max, cfg.eps, cfg.eta_rank, ...
            cfg.eta_b, cfg.eta_growth, false);
    otherwise
        error('Unknown mode: %s', mode);
end
rec.status = res.status;
if ~contains(res.status, 'Solved') || ~isfield(res, 'is_physical_feasible') ...
        || ~res.is_physical_feasible
    return;
end

rec.feasible = true;
rec.power = res.final_obj;
rec.sum_rate = res.sum_rate;
rec.pcrb = res.pcrb;
rec.sens_sinr_db = res.sens_sinr_db;
rec.active_aps = sum(any(res.b > 0.5, 2));
rec.outage = evaluate_outage(res.W, prm, eval_eps, 200);
end

function prm = make_scenario(cfg, nreq, eps_h, seed)
prm = generate_scenario(cfg.M, cfg.Nt, cfg.K, cfg.P, cfg.N_theta, ...
    cfg.Pmax_dBm, cfg.Gamma_track, 'AreaSize', cfg.AreaSize, ...
    'N_req', nreq, 'eps_h', eps_h, 'seed', seed);
end

function outage = evaluate_outage(W, prm, eps_h, n_samples)
if eps_h == 0
    outage = 0;
    return;
end
outages = 0;
total = 0;
for k = 1:prm.K
    h_nom = prm.H(:,k);
    radius = eps_h * norm(h_nom);
    for s = 1:n_samples
        direction = randn(prm.N,1) + 1j * randn(prm.N,1);
        direction = direction / norm(direction);
        radius_sample = radius * rand()^(1 / (2 * prm.N));
        h = h_nom + radius_sample * direction;
        signal = real(h' * W{k} * h);
        interference = 0;
        for j = 1:prm.K
            if j ~= k, interference = interference + real(h' * W{j} * h); end
        end
        outages = outages + (signal / (interference + prm.sigma_c2) < prm.gamma_k(k));
        total = total + 1;
    end
end
outage = outages / total;
end

function b = nearest_assignment(prm)
b = zeros(prm.M, prm.P);
for p = prm.active_targets
    distance = sqrt(sum((prm.AP_pos - prm.Target_pos(p,:)).^2, 2));
    [~, order] = sort(distance, 'ascend');
    b(order(1:prm.N_req), p) = 1;
end
end

function summary = make_summary_table(nreq_list, result)
summary = table(nreq_list(:), [result.proposed.feasibility]', ...
    [result.nearest.feasibility]', [result.proposed.conditional_power]', ...
    [result.nearest.conditional_power]', [result.proposed.conditional_active_aps]', ...
    [result.nearest.conditional_active_aps]', [result.proposed.conditional_rate]', ...
    [result.nearest.conditional_rate]', [result.proposed.conditional_pcrb]', ...
    [result.nearest.conditional_pcrb]', [result.proposed.conditional_sensing_sinr_db]', ...
    [result.nearest.conditional_sensing_sinr_db]', [result.proposed.conditional_outage]', ...
    [result.nearest.conditional_outage]', ...
    'VariableNames', {'N_req', 'proposed_feasibility', 'nearest_feasibility', ...
    'proposed_power_given_feasible_W', 'nearest_power_given_feasible_W', ...
    'proposed_active_APs_given_feasible', 'nearest_active_APs_given_feasible', ...
    'proposed_rate_given_feasible', 'nearest_rate_given_feasible', ...
    'proposed_mean_PCRB_given_feasible', 'nearest_mean_PCRB_given_feasible', ...
    'proposed_sensing_SINR_dB_given_feasible', 'nearest_sensing_SINR_dB_given_feasible', ...
    'proposed_outage_given_feasible', 'nearest_outage_given_feasible'});
end
