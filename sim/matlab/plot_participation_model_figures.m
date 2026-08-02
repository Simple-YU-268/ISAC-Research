function plot_participation_model_figures(results_root, figure_dir)
%PLOT_PARTICIPATION_MODEL_FIGURES  Paper plots from current-model raw data.
if nargin < 1 || isempty(results_root)
    results_root = fullfile(pwd,'experiment_packages','v1.0','results');
end
if nargin < 2 || isempty(figure_dir)
    figure_dir = fullfile(fileparts(results_root),'figures');
end
if ~exist(figure_dir,'dir'), mkdir(figure_dir); end
set(groot,'defaultFigureVisible','off');

%% Fig. 3: cluster-size feasibility, power, and runtime.
L=load(fullfile(results_root,'participation_nreq_sweep_30seeds','nreq_qos_final.mat'));
R=L.output.records; x=L.output.nreq_list(:).';
feas=zeros(size(x)); medP=NaN(size(x)); loP=medP; hiP=medP; medT=medP; p90T=medP;
for q=1:numel(x)
    rr=R(q,:); f=[rr.feasible]; metrics=[rr(f).metrics];
    power=1e3*arrayfun(@(m)m.total_power_W,metrics);
    t=[rr.time_s]; feas(q)=mean(f); medP(q)=median(power); loP(q)=prctile(power,10); hiP(q)=prctile(power,90);
    medT(q)=median(t); p90T(q)=prctile(t,90);
end
fig=figure('Position',[100 100 1120 300]); tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
ax=nexttile; plot(x,100*feas,'-o','LineWidth',1.8); grid on; ylim([0 105]); xlabel('N_{req}'); ylabel('Physical feasibility (%)');
ax=nexttile; errorbar(x,medP,medP-loP,hiP-medP,'-o','LineWidth',1.8); grid on; xlabel('N_{req}'); ylabel('Total transmit power (mW)');
ax=nexttile; errorbar(x,medT,zeros(size(x)),p90T-medT,'-o','LineWidth',1.8); grid on; xlabel('N_{req}'); ylabel('End-to-end runtime (s)');
exportgraphics(fig,fullfile(figure_dir,'fig3_cluster_size_tradeoff.png'),'Resolution',300); close(fig);

%% Fig. 4: QoS tightness from current Nreq records.
plot_nreq_qos_statistics(fullfile(results_root,'participation_nreq_sweep_30seeds','nreq_qos_final.mat'),figure_dir);

%% Fig. 7: current-model scaling.
L=load(fullfile(results_root,'participation_network_scaling_10seeds','network_scaling_final.mat'));
R=L.output.records; M=L.output.M_list(:).'; N=zeros(size(M)); medT=NaN(size(M)); p90T=medT; feas=zeros(size(M)); medP=medT;
for q=1:numel(M)
    rr=R(q,:); N(q)=rr(1).N; f=[rr.feasible]; t=[rr.time_s]; p=1e3*[rr(f).power_W];
    medT(q)=median(t); p90T(q)=prctile(t,90); feas(q)=mean(f); medP(q)=median(p);
end
fig=figure('Position',[100 100 1120 300]); tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
ax=nexttile; errorbar(N,medT,zeros(size(N)),p90T-medT,'-o','LineWidth',1.8); grid on; xlabel('Total transmit antennas, N'); ylabel('Runtime (s)');
ax=nexttile; plot(N,100*feas,'-o','LineWidth',1.8); grid on; ylim([0 105]); xlabel('Total transmit antennas, N'); ylabel('Physical feasibility (%)');
ax=nexttile; plot(N,medP,'-o','LineWidth',1.8); grid on; xlabel('Total transmit antennas, N'); ylabel('Median transmit power (mW)');
exportgraphics(fig,fullfile(figure_dir,'fig7_dimension_sensitivity.png'),'Resolution',300); close(fig);

%% Fig. 9: merge 10- and 20-seed robust records, then plot once.
A=load(fullfile(results_root,'participation_csi_robustness_pilot10','csi_robustness_final.mat'));
B=load(fullfile(results_root,'participation_csi_robustness_seeds11to30','csi_robustness_final.mat'));
output=A.output; output.records=[A.output.records B.output.records]; output.seeds=[A.output.seeds B.output.seeds];
combined_file=fullfile(results_root,'participation_csi_robustness_30seeds.mat'); save(combined_file,'output');
plot_csi_robustness(combined_file,figure_dir);
end
