function plot_paper_experiment_figures(output_dir)
%PLOT_PAPER_EXPERIMENT_FIGURES  Create paper figures from saved MAT results.
if nargin < 1
    output_dir = fullfile(pwd, '..', '..', 'experiment_packages', 'v1.0', 'figures');
end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
set(groot, 'defaultFigureVisible', 'off');

root = fullfile(pwd, '..', '..', 'experiment_packages', 'v1.0', 'results');
nreq_paths = {fullfile(root,'nreq_sweep','nreq2','pilot_final.mat'), ...
    fullfile(root,'main_config_mc_100seeds','pilot_final.mat'), ...
    fullfile(root,'nreq_sweep','nreq4','pilot_final.mat'), ...
    fullfile(root,'nreq_sweep','nreq5','pilot_final.mat'), ...
    fullfile(root,'nreq_sweep','nreq6','pilot_final.mat')};
nreq = 2:6;
feas = zeros(1,5); power_med = zeros(1,5); power_lo = zeros(1,5); power_hi = zeros(1,5);
time_med = zeros(1,5); time_p90 = zeros(1,5);
for q = 1:5
    data = load(nreq_paths{q}, 'records'); records = data.records;
    if q == 2, records = records(1:50); end
    f = [records.proposed_feasible]; p = 1e3*[records.proposed_power_W]; t = [records.proposed_time_s];
    feas(q) = mean(f); power_med(q) = median(p(f)); power_lo(q) = prctile(p(f),10); power_hi(q) = prctile(p(f),90);
    time_med(q) = median(t); time_p90(q) = prctile(t,90);
end
fig = figure('Position',[100 100 1200 330]);
subplot(1,3,1); plot(nreq,100*feas,'-o','LineWidth',1.8); ylim([0 105]); grid on; xlabel('N_{req}'); ylabel('Feasibility (%)');
subplot(1,3,2); errorbar(nreq,power_med,power_med-power_lo,power_hi-power_med,'-o','LineWidth',1.8); grid on; xlabel('N_{req}'); ylabel('Total transmit power (mW)');
subplot(1,3,3); errorbar(nreq,time_med,zeros(size(nreq)),time_p90-time_med,'-o','LineWidth',1.8); grid on; xlabel('N_{req}'); ylabel('Runtime (s)');
exportgraphics(fig, fullfile(output_dir,'fig3_cluster_size_tradeoff.png'),'Resolution',300); close(fig);

data = load(fullfile(root,'main_config_mc_100seeds','pilot_final.mat'),'records');
g = [data.records.power_penalty_pct];
fig = figure('Position',[100 100 500 360]); [f,x] = ecdf(g); stairs(x,f,'LineWidth',2); grid on; xlabel('Power penalty over SDR (%)'); ylabel('Empirical CDF');
exportgraphics(fig, fullfile(output_dir,'fig5_power_gap_cdf.png'),'Resolution',300); close(fig);

data = load(fullfile(root,'recovery_ablation_30seeds','ablation_final.mat'),'summary'); s = data.summary;
ablation_data = load(fullfile(root,'recovery_ablation_30seeds','ablation_final.mat'),'records');
sample_count = zeros(1,3);
for m=1:3, sample_count(m) = nnz(arrayfun(@(r) r.methods(m).feasible, ablation_data.records)); end
fig = figure('Position',[100 100 1200 330]);
subplot(1,3,1); bar(100*s.feasibility_rate); ylim([0 105]); grid on; ylabel('Feasibility (%)');
subplot(1,3,2); bar(s.power_gap_median_pct); grid on; ylabel('Median power penalty (%)');
subplot(1,3,3); bar(s.time_median_s); hold on; errorbar(1:3,s.time_median_s,s.time_p90_s-s.time_median_s,'.k','LineWidth',1.5); grid on; ylabel('Runtime (s)');
method_labels = {sprintf('FIM\\n(n=%d)',sample_count(1)), ...
    sprintf('DC Top-N\\n(n=%d)',sample_count(2)), sprintf('Full\\n(n=%d)',sample_count(3))};
for a=1:3, subplot(1,3,a); set(gca,'XTickLabel',method_labels); end
exportgraphics(fig, fullfile(output_dir,'fig6_recovery_ablation.png'),'Resolution',300); close(fig);

dim_paths = {fullfile(root,'dimension_sweep','m4','pilot_final.mat'), ...
    fullfile(root,'main_config_mc_100seeds','pilot_final.mat'), ...
    fullfile(root,'dimension_sweep','m8','pilot_final.mat')};
N = [8 12 16]; tm = zeros(1,3); tp = zeros(1,3); ff = zeros(1,3);
for q=1:3
    data = load(dim_paths{q},'records'); records = data.records; if q==2, records=records(1:30); end
    t=[records.proposed_time_s]; tm(q)=median(t); tp(q)=prctile(t,90); ff(q)=mean([records.proposed_feasible]);
end
fig = figure('Position',[100 100 800 330]);
subplot(1,2,1); errorbar(N,tm,zeros(size(N)),tp-tm,'-o','LineWidth',1.8); grid on; xlabel('Total antennas N'); ylabel('Runtime (s)');
subplot(1,2,2); plot(N,100*ff,'-o','LineWidth',1.8); ylim([0 105]); grid on; xlabel('Total antennas N'); ylabel('Feasibility (%)');
exportgraphics(fig, fullfile(output_dir,'fig7_dimension_sensitivity.png'),'Resolution',300); close(fig);
end
