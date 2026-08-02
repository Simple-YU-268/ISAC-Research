function [nreq_result, robust, cfg] = experiments_paper(varargin)
%EXPERIMENTS_PAPER  Feasibility-aware Monte Carlo study for Cell-Free ISAC.
%   The binary AP-target association determines the dedicated sensing waveform
%   resources, while communication transmission remains globally cooperative.
%   Consequently an infeasible sensing topology is a physical outcome, not a
%   missing power sample. This script reports feasibility first and reports
%   power, communication, sensing, and cluster cost conditionally on feasible
%   binary solutions.
%
%   Examples:
%     experiments_paper('Quick', true)       % 3 paired seeds, smoke study
%     experiments_paper('N_mc', 100)         % final Monte Carlo campaign
%     experiments_paper('N_req_list', 1:6, 'Run_robustness', false)
%
%   All N_req values share the same seed set, channel model, power budget, and
%   auto-calibrated Gamma reference.  The latter is independent of N_req.

p = inputParser;
addParameter(p, 'Quick', false, @islogical);
addParameter(p, 'N_mc', 30, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'N_workers', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'Base_seed', 2026, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'N_req_list', [], @(x) isnumeric(x) && isvector(x));
addParameter(p, 'Run_robustness', true, @islogical);
addParameter(p, 'T_max', 30, @(x) isnumeric(x) && isscalar(x) && x >= 1 && x == round(x));
addParameter(p, 'Solver', 'mosek', @(x) ischar(x) || isstring(x));
addParameter(p, 'Recovery_max_candidates', 21, @(x) isnumeric(x) && isscalar(x) && x >= 1 && x == round(x));
addParameter(p, 'Recovery_stop_first_feasible', false, @islogical);
addParameter(p, 'Output_dir', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'Output_tag', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'Progress_file', '', @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opt = p.Results;
if opt.Quick, opt.N_mc = 3; end

%% Common configuration ----------------------------------------------------
cfg.M = 8; cfg.Nt = 4; cfg.K = 4; cfg.P = 2; cfg.N_theta = 2;
cfg.Pmax_dBm = 20;
cfg.AreaSize = 400;
cfg.eps_h = 0.05;
cfg.Gamma_track = 'auto';      % per-target physical isotropic-reference calibration
cfg.N_req_main = 3;            % representative sensing-cluster size
cfg.N_req_list = 1:cfg.M;
cfg.T_max = opt.T_max;
cfg.eps = 1e-5;
cfg.eta_rank = 1.0;
cfg.eta_b = 1.0;
cfg.eta_growth = 1.0;  % retained in calls; paper algorithm uses fixed penalties
cfg.N_mc = opt.N_mc;
cfg.N_workers = opt.N_workers;
cfg.Base_seed = opt.Base_seed;
cfg.sensing_power_threshold = 1e-8;  % W: realized AP-target sensing activity
cfg.outage_samples = 200;
if opt.Quick
    cfg.N_req_list = 1:min(4, cfg.M);
    cfg.outage_samples = 40;
end
if ~isempty(opt.N_req_list)
    cfg.N_req_list = unique(opt.N_req_list(:).');
    assert(all(cfg.N_req_list >= 1 & cfg.N_req_list <= cfg.M & ...
        cfg.N_req_list == round(cfg.N_req_list)), ...
        'N_req_list must contain integer values in 1:M.');
end
cfg.run_robustness = opt.Run_robustness;
cfg.solver = char(opt.Solver);
cfg.nearest_baseline = 'oracle_true_position';
cfg.recovery_max_candidates = opt.Recovery_max_candidates;
cfg.recovery_stop_first_feasible = opt.Recovery_stop_first_feasible;

if strlength(string(opt.Output_dir)) == 0
    out_dir = fullfile(pwd, 'figures');
else
    out_dir = char(opt.Output_dir);
end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save_tag = sprintf('feasibility_M%d_K%d_P%d_mc%d', cfg.M, cfg.K, cfg.P, cfg.N_mc);
if strlength(string(opt.Output_tag)) > 0
    save_tag = [save_tag, '_', char(opt.Output_tag)];
end
if strlength(string(opt.Progress_file)) == 0
    cfg.progress_file = fullfile(out_dir, [save_tag, '_progress.log']);
else
    cfg.progress_file = char(opt.Progress_file);
end
fid = fopen(cfg.progress_file, 'a');
if fid >= 0
    fprintf(fid, '[%s] Started: N_mc=%d, N_req=%s, solver=%s\\n', ...
        datestr(now, 'yyyy-mm-dd HH:MM:SS'), cfg.N_mc, mat2str(cfg.N_req_list), cfg.solver);
    fclose(fid);
end

%% Figure 1: representative convergence at a feasible operating point -----
fprintf('=== Figure 1: representative convergence (N_req=%d) ===\n', cfg.N_req_main);
prm = make_scenario(cfg, cfg.N_req_main, cfg.eps_h, cfg.Base_seed);
res = baseline_alg2(prm, cfg.T_max, cfg.eps, cfg.eta_rank, ...
    cfg.eta_b, cfg.eta_growth, true);
if isfield(res, 'true_obj_trace') && ~isempty(res.true_obj_trace)
    figure('Name', 'Fig1_Convergence');
    tiledlayout(2,1, 'TileSpacing', 'compact');
    nexttile;
    plot(1:numel(res.true_obj_trace), res.true_obj_trace, 'b-o', 'LineWidth', 1.5);
    grid on; ylabel('True fixed-penalty objective');
    title(sprintf('Representative fixed-penalty DC-SCA (N_{req}=%d)', cfg.N_req_main));
    nexttile;
    semilogy(1:numel(res.rank_residual_trace), max(res.rank_residual_trace, eps), ...
        'r-o', 1:numel(res.binary_residual_trace), max(res.binary_residual_trace, eps), ...
        'k-s', 'LineWidth', 1.5);
    grid on; xlabel('DC-SCA iteration'); ylabel('Summed residual');
    legend('rank residual', 'binary residual', 'Location', 'best');
    saveas(gcf, fullfile(out_dir, [save_tag, '_fig1_convergence.png']));
else
    warning('Representative N_req=%d realization did not complete a DCP iteration; no convergence plot.', ...
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
    fprintf('  N_req=%d: random-assignment baseline ...\n', nreq);
    nreq_result.random(i) = mc_run(cfg, nreq, 'random', cfg.eps_h, cfg.eps_h);
    nreq_result.paired(i) = paired_stats(nreq_result.proposed(i), ...
        nreq_result.nearest(i));
    nreq_result.paired_random(i) = paired_stats(nreq_result.proposed(i), ...
        nreq_result.random(i));
end

prop_feas = [nreq_result.proposed.feasibility];
near_feas = [nreq_result.nearest.feasibility];
rand_feas = [nreq_result.random.feasibility];
prop_power = [nreq_result.proposed.conditional_power];
near_power = [nreq_result.nearest.conditional_power];
rand_power = [nreq_result.random.conditional_power];
prop_std = [nreq_result.proposed.conditional_power_std];
near_std = [nreq_result.nearest.conditional_power_std];
rand_std = [nreq_result.random.conditional_power_std];

figure('Name', 'Fig2_FeasibilityVsNreq');
plot(cfg.N_req_list, prop_feas, 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, near_feas, 'r--s', 'LineWidth', 1.5);
plot(cfg.N_req_list, rand_feas, 'k:^', 'LineWidth', 1.5);
grid on; xlim([min(cfg.N_req_list)-0.5, max(cfg.N_req_list)+0.5]); ylim([0 1]);
xlabel('Required APs per target N_{req}'); ylabel('Binary physical feasibility rate');
legend('Proposed association + recovery', 'Oracle nearest-AP fixed-b', 'Random fixed-b', 'Location', 'southeast');
title('Feasibility versus AP-association cardinality');
saveas(gcf, fullfile(out_dir, [save_tag, '_fig2_feasibility_vs_nreq.png']));

figure('Name', 'Fig3_ConditionalPowerVsNreq');
errorbar(cfg.N_req_list, prop_power, prop_std, 'b-o', 'LineWidth', 1.5); hold on;
errorbar(cfg.N_req_list, near_power, near_std, 'r--s', 'LineWidth', 1.5);
errorbar(cfg.N_req_list, rand_power, rand_std, 'k:^', 'LineWidth', 1.5);
grid on; xlim([min(cfg.N_req_list)-0.5, max(cfg.N_req_list)+0.5]);
xlabel('Required APs per target N_{req}');
ylabel('Transmit power [W] | feasible');
legend('Proposed association + recovery', 'Oracle nearest-AP fixed-b', 'Random fixed-b', 'Location', 'best');
title('Conditional power: infeasible trials are excluded explicitly');
saveas(gcf, fullfile(out_dir, [save_tag, '_fig3_conditional_power_vs_nreq.png']));

% Communication, sensing, and cooperation outcomes must be read together
% with Fig. 2. They are conditional metrics, not substitutes for feasibility.
figure('Name', 'Fig4_ConditionalPerformanceVsNreq');
subplot(2,2,1);
plot(cfg.N_req_list, [nreq_result.proposed.conditional_rate], 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, [nreq_result.nearest.conditional_rate], 'r--s', 'LineWidth', 1.5);
plot(cfg.N_req_list, [nreq_result.random.conditional_rate], 'k:^', 'LineWidth', 1.5);
grid on; xlim([min(cfg.N_req_list)-0.5, max(cfg.N_req_list)+0.5]); xlabel('N_{req}'); ylabel('Sum rate [bit/s/Hz] | feasible');
legend('Optimized association', 'Nearest-AP', 'Random assignment', 'Location', 'best');

subplot(2,2,2);
plot(cfg.N_req_list, [nreq_result.proposed.conditional_pcrb], 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, [nreq_result.nearest.conditional_pcrb], 'r--s', 'LineWidth', 1.5);
plot(cfg.N_req_list, [nreq_result.random.conditional_pcrb], 'k:^', 'LineWidth', 1.5);
grid on; xlim([min(cfg.N_req_list)-0.5, max(cfg.N_req_list)+0.5]); xlabel('N_{req}'); ylabel('Mean PCRB trace | feasible');

subplot(2,2,3);
plot(cfg.N_req_list, [nreq_result.proposed.conditional_sensing_sinr_db], 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, [nreq_result.nearest.conditional_sensing_sinr_db], 'r--s', 'LineWidth', 1.5);
plot(cfg.N_req_list, [nreq_result.random.conditional_sensing_sinr_db], 'k:^', 'LineWidth', 1.5);
grid on; xlim([min(cfg.N_req_list)-0.5, max(cfg.N_req_list)+0.5]); xlabel('N_{req}'); ylabel('Mean sensing SINR [dB] | feasible');

subplot(2,2,4);
plot(cfg.N_req_list, [nreq_result.proposed.conditional_realized_network_aps], 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, [nreq_result.nearest.conditional_realized_network_aps], 'r--s', 'LineWidth', 1.5);
plot(cfg.N_req_list, [nreq_result.random.conditional_realized_network_aps], 'k:^', 'LineWidth', 1.5);
grid on; xlim([min(cfg.N_req_list)-0.5, max(cfg.N_req_list)+0.5]); xlabel('N_{req}'); ylabel('APs with nonzero sensing power | feasible');
saveas(gcf, fullfile(out_dir, [save_tag, '_fig4_conditional_performance_vs_nreq.png']));

% Direct algorithm comparison must use only paired realizations where both
% methods return a binary physical solution.
figure('Name', 'Fig5_PairedPerformanceVsNreq');
subplot(2,2,1);
plot(cfg.N_req_list, [nreq_result.paired.proposed_rate], 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, [nreq_result.paired.nearest_rate], 'r--s', 'LineWidth', 1.5);
grid on; xlim([min(cfg.N_req_list)-0.5, max(cfg.N_req_list)+0.5]); xlabel('N_{req}'); ylabel('Sum rate [bit/s/Hz] | jointly feasible');
legend('Proposed', 'Nearest-AP', 'Location', 'best');

subplot(2,2,2);
plot(cfg.N_req_list, [nreq_result.paired.proposed_pcrb_ratio], 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, [nreq_result.paired.nearest_pcrb_ratio], 'r--s', 'LineWidth', 1.5);
grid on; xlim([min(cfg.N_req_list)-0.5, max(cfg.N_req_list)+0.5]); xlabel('N_{req}'); ylabel('Mean PCRB / threshold | jointly feasible');

subplot(2,2,3);
plot(cfg.N_req_list, [nreq_result.paired.proposed_power], 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, [nreq_result.paired.nearest_power], 'r--s', 'LineWidth', 1.5);
grid on; xlim([min(cfg.N_req_list)-0.5, max(cfg.N_req_list)+0.5]); xlabel('N_{req}'); ylabel('Transmit power [W] | jointly feasible');

subplot(2,2,4);
plot(cfg.N_req_list, [nreq_result.paired.proposed_realized_per_target], 'b-o', 'LineWidth', 1.5); hold on;
plot(cfg.N_req_list, [nreq_result.paired.nearest_realized_per_target], 'r--s', 'LineWidth', 1.5);
grid on; xlim([min(cfg.N_req_list)-0.5, max(cfg.N_req_list)+0.5]); xlabel('N_{req}'); ylabel('Realized sensing APs per target | jointly feasible');
saveas(gcf, fullfile(out_dir, [save_tag, '_fig5_paired_performance_vs_nreq.png']));

%% Figure 6: robustness at the selected feasible operating point -----------
eps_list = [];
robust = struct();
if cfg.run_robustness
    fprintf('=== Figure 6: robustness at N_req=%d ===\n', cfg.N_req_main);
    eps_list = [0, 0.02, 0.05, 0.08];
    for i = 1:numel(eps_list)
        fprintf('  design/evaluation eps_h=%.2f: robust ...\n', eps_list(i));
        robust.proposed(i) = mc_run(cfg, cfg.N_req_main, 'proposed', eps_list(i), eps_list(i));
        fprintf('  design eps_h=0, evaluation eps_h=%.2f: non-robust ...\n', eps_list(i));
        robust.nonrobust(i) = mc_run(cfg, cfg.N_req_main, 'nonrobust', 0, eps_list(i));
    end

    figure('Name', 'Fig6_Robustness');
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
    saveas(gcf, fullfile(out_dir, [save_tag, '_fig6_robustness.png']));
end

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
        write_trial_progress(cfg, nreq, mode, n, records{n});
    end
end

is_feasible = cellfun(@(r) r.feasible, records);
stats.N_mc = cfg.N_mc;
stats.n_feasible = sum(is_feasible);
stats.feasibility = mean(is_feasible);
stats.records = records;
status = cellfun(@(r) r.status, records, 'UniformOutput', false);
stats.n_relaxed_infeasible = sum(contains(status, 'infeasible_relaxed'));
stats.n_binary_infeasible = sum(contains(status, 'infeasible_after_rounding') | ...
    contains(status, 'initial_infeasible') | contains(status, 'physical_solution_infeasible'));
stats.n_other_failure = cfg.N_mc - stats.n_feasible - stats.n_relaxed_infeasible - ...
    stats.n_binary_infeasible;
if ~any(is_feasible)
    stats.conditional_power = NaN;
    stats.conditional_power_std = NaN;
    stats.conditional_rate = NaN;
    stats.conditional_pcrb = NaN;
    stats.conditional_sensing_sinr_db = NaN;
    stats.conditional_outage = NaN;
    stats.conditional_active_aps = NaN;
    stats.conditional_realized_per_target = NaN;
    stats.conditional_realized_network_aps = NaN;
    stats.conditional_sensing_power = NaN;
    stats.conditional_communication_power = NaN;
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
stats.conditional_realized_per_target = mean(cellfun(@(r) mean(r.realized_per_target), valid));
stats.conditional_realized_network_aps = mean(cellfun(@(r) r.realized_network_aps, valid));
stats.conditional_sensing_power = mean(cellfun(@(r) r.sensing_power, valid));
stats.conditional_communication_power = mean(cellfun(@(r) r.communication_power, valid));
end

function rec = run_trial(cfg, nreq, mode, design_eps, eval_eps, index)
prm = make_scenario(cfg, nreq, design_eps, cfg.Base_seed + index);
rec.feasible = false;
rec.status = '';
rec.power = NaN; rec.sum_rate = NaN; rec.pcrb = NaN; rec.sens_sinr_db = NaN;
rec.outage = NaN; rec.active_aps = NaN;
rec.sensing_power = NaN; rec.communication_power = NaN;
rec.realized_per_target = NaN; rec.realized_network_aps = NaN;

switch mode
    case 'proposed'
        res = baseline_alg2(prm, cfg.T_max, cfg.eps, cfg.eta_rank, ...
            cfg.eta_b, cfg.eta_growth, false);
    case 'nearest'
        res = solve_p3_with_fixed_b(prm, nearest_assignment(prm), ...
            cfg.T_max, cfg.eps, cfg.eta_rank, cfg.eta_b, cfg.eta_growth);
    case 'random'
        % Same cardinality and continuous recovery as the optimized method;
        % only the AP--target sensing assignment is random.
        assignment_seed = cfg.Base_seed + 700000*nreq + 1000*index;
        res = solve_p3_with_fixed_b(prm, random_assignment(prm, assignment_seed), ...
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
rec.gamma_track = prm.Gamma_track;
rec.active_aps = sum(any(res.b > 0.5, 2));
rec.sensing_power = sum(arrayfun(@(p) real(trace(res.S_p(:,:,p))), 1:prm.P));
rec.communication_power = res.final_obj - rec.sensing_power;
[rec.realized_per_target, rec.realized_network_aps] = ...
    realized_sensing_activity(res.S_p, prm, cfg.sensing_power_threshold);
eval_seed = cfg.Base_seed + 100000*nreq + 1000*index + round(1e4*eval_eps);
rec.outage = evaluate_outage(res.W, res.S_p, prm, eval_eps, ...
    cfg.outage_samples, eval_seed);
end

function prm = make_scenario(cfg, nreq, eps_h, seed)
prm = generate_scenario(cfg.M, cfg.Nt, cfg.K, cfg.P, cfg.N_theta, ...
    cfg.Pmax_dBm, cfg.Gamma_track, 'AreaSize', cfg.AreaSize, ...
    'N_req', nreq, 'eps_h', eps_h, 'seed', seed);
prm.solver = cfg.solver;
prm.recovery_max_candidates = cfg.recovery_max_candidates;
prm.recovery_stop_first_feasible = cfg.recovery_stop_first_feasible;
end

function write_trial_progress(cfg, nreq, mode, index, rec)
message = sprintf('[%s] mode=%s N_req=%d trial=%d/%d status=%s feasible=%d\\n', ...
    datestr(now, 'yyyy-mm-dd HH:MM:SS'), mode, nreq, index, cfg.N_mc, ...
    rec.status, rec.feasible);
fprintf('%s', message);
fid = fopen(cfg.progress_file, 'a');
if fid >= 0
    fprintf(fid, '%s', message);
    fclose(fid);
end
end

function outage = evaluate_outage(W, S_p, prm, eps_h, n_samples, rng_seed)
if eps_h == 0
    outage = 0;
    return;
end
prior_rng = rng;
rng(rng_seed, 'twister');
cleanup_rng = onCleanup(@() rng(prior_rng));
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
        if ~isfield(prm, 'sensing_waveform_cancelled_at_ue') || ...
                ~prm.sensing_waveform_cancelled_at_ue
            interference = interference + real(h' * sum(S_p, 3) * h);
        end
        outages = outages + (signal / (interference + prm.sigma_c2) < prm.gamma_k(k));
        total = total + 1;
    end
end
outage = outages / total;
end

function [per_target, network_aps] = realized_sensing_activity(S_p, prm, threshold)
M = prm.M; P = prm.P; Nt = prm.N / prm.M;
active = false(M, P);
for p = 1:P
    for m = 1:M
        block = (m-1)*Nt + (1:Nt);
        active(m,p) = real(trace(S_p(block,block,p))) > threshold;
    end
end
per_target = sum(active, 1);
network_aps = sum(any(active, 2));
end

function stats = paired_stats(proposed, nearest)
mask = cellfun(@(rp,rn) rp.feasible && rn.feasible, ...
    proposed.records, nearest.records);
stats.n_joint_feasible = sum(mask);
stats.joint_feasibility = mean(mask);
if ~any(mask)
    stats.proposed_rate = NaN; stats.nearest_rate = NaN;
    stats.proposed_pcrb_ratio = NaN; stats.nearest_pcrb_ratio = NaN;
    stats.proposed_power = NaN; stats.nearest_power = NaN;
    stats.proposed_realized_per_target = NaN;
    stats.nearest_realized_per_target = NaN;
    return;
end
rp = proposed.records(mask);
rn = nearest.records(mask);
stats.proposed_rate = mean(cellfun(@(r) r.sum_rate, rp));
stats.nearest_rate = mean(cellfun(@(r) r.sum_rate, rn));
stats.proposed_pcrb_ratio = mean(cellfun(@(r) mean(r.pcrb ./ ...
    r.gamma_track), rp));
stats.nearest_pcrb_ratio = mean(cellfun(@(r) mean(r.pcrb ./ ...
    r.gamma_track), rn));
stats.proposed_power = mean(cellfun(@(r) r.power, rp));
stats.nearest_power = mean(cellfun(@(r) r.power, rn));
stats.proposed_realized_per_target = mean(cellfun(@(r) ...
    mean(r.realized_per_target), rp));
stats.nearest_realized_per_target = mean(cellfun(@(r) ...
    mean(r.realized_per_target), rn));
end

function b = nearest_assignment(prm)
b = zeros(prm.M, prm.P);
% Oracle benchmark: true target positions are unavailable to an online AP
% association policy. This method is retained only as a theoretical geometric
% reference and must be labelled as such in all reported results.
target_reference = prm.Target_pos;
for p = prm.active_targets
    distance = sqrt(sum((prm.AP_pos - target_reference(p,:)).^2, 2));
    [~, order] = sort(distance, 'ascend');
    b(order(1:prm.N_req), p) = 1;
end
end

function b = random_assignment(prm, assignment_seed)
%RANDOM_ASSIGNMENT  Fair random baseline with exactly N_req APs per target.
prior_rng = rng;
cleanup_rng = onCleanup(@() rng(prior_rng));
rng(assignment_seed, 'twister');
b = zeros(prm.M, prm.P);
for p = prm.active_targets
    order = randperm(prm.M, prm.N_req);
    b(order,p) = 1;
end
end

function summary = make_summary_table(nreq_list, result)
summary = table(nreq_list(:), [result.proposed.feasibility]', ...
    [result.nearest.feasibility]', [result.random.feasibility]', [result.proposed.conditional_power]', ...
    [result.nearest.conditional_power]', [result.random.conditional_power]', [result.proposed.conditional_active_aps]', ...
    [result.nearest.conditional_active_aps]', [result.proposed.conditional_rate]', ...
    [result.nearest.conditional_rate]', [result.proposed.conditional_pcrb]', ...
    [result.nearest.conditional_pcrb]', [result.proposed.conditional_sensing_sinr_db]', ...
    [result.nearest.conditional_sensing_sinr_db]', [result.proposed.conditional_outage]', ...
    [result.nearest.conditional_outage]', [result.proposed.conditional_realized_per_target]', ...
    [result.nearest.conditional_realized_per_target]', [result.paired.joint_feasibility]', ...
    [result.paired.proposed_power]', [result.paired.nearest_power]', ...
    'VariableNames', {'N_req', 'proposed_feasibility', 'nearest_feasibility', 'random_feasibility', ...
    'proposed_power_given_feasible_W', 'nearest_power_given_feasible_W', 'random_power_given_feasible_W', ...
    'proposed_active_APs_given_feasible', 'nearest_active_APs_given_feasible', ...
    'proposed_rate_given_feasible', 'nearest_rate_given_feasible', ...
    'proposed_mean_PCRB_given_feasible', 'nearest_mean_PCRB_given_feasible', ...
    'proposed_sensing_SINR_dB_given_feasible', 'nearest_sensing_SINR_dB_given_feasible', ...
    'proposed_outage_given_feasible', 'nearest_outage_given_feasible', ...
    'proposed_realized_sensing_APs_per_target', 'nearest_realized_sensing_APs_per_target', ...
    'joint_feasibility', 'proposed_power_jointly_feasible_W', ...
    'nearest_power_jointly_feasible_W'});
end
