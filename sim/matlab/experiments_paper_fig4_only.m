function experiments_paper_fig4_only()
%EXPERIMENTS_PAPER_FIG4_ONLY  Continue from existing Figure 1-3 results and run only Figure 4.
%   This avoids re-running the slow Pareto and Robustness Monte Carlo sections.

M = 8; Nt = 4; K = 4; P = 2; N_theta = 2;
Pmax_dBm = 20;
Gamma_track = 10;
N_mc = 3;
N_workers = 0;  % sequential to avoid parpool/CVX issues

save_tag = sprintf('M%d_Nt%d_K%d_P%d_Nth%d_mc%d', M, Nt, K, P, N_theta, N_mc);
out_dir = fullfile(pwd, 'figures');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

mat_path = fullfile(out_dir, [save_tag, '_results.mat']);
if ~exist(mat_path, 'file')
    error('Results file not found: %s. Run experiments_paper first to generate Figures 1-3.', mat_path);
end

load(mat_path);
fprintf('Loaded existing results from %s\n', mat_path);

%% Figure 4: AP assignment effectiveness vs N_req
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

%% Save numeric results (overwrite with Figure 4 fields added)
save(mat_path, ...
    'Gamma_list', 'pareto_power', 'pareto_rate', ...
    'eps_list', 'rob_power', 'rob_power_std', 'rob_outage', 'base_power', 'base_outage', ...
    'N_req_list', 'assign_power_proposed', 'assign_power_heuristic');

fprintf('Figure 4 and updated results saved to: %s\n', out_dir);
end
