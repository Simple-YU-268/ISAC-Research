function stats = plot_nreq_qos_statistics(result_file, output_dir)
%PLOT_NREQ_QOS_STATISTICS  Figure 4: QoS tightness versus cluster size.

if nargin < 1 || isempty(result_file)
    result_file = fullfile(pwd,'..','..','experiment_packages','v1.0','results', ...
        'nreq_qos_sweep','nreq_qos_final.mat');
end
if nargin < 2 || isempty(output_dir), output_dir = fullfile(pwd,'..','..', ...
        'experiment_packages','v1.0','figures'); end
if ~exist(output_dir,'dir'), mkdir(output_dir); end
loaded = load(result_file,'output'); output = loaded.output;
nreq = output.nreq_list(:).'; records = output.records;
stats.nreq = nreq;
stats.pcrb = NaN(numel(nreq),3); stats.comm = stats.pcrb; stats.sensing = stats.pcrb;
stats.count = zeros(numel(nreq),1);
for q = 1:numel(nreq)
    feasible = [records(q,:).feasible];
    rec = records(q,feasible); stats.count(q) = numel(rec);
    if isempty(rec), continue; end
    pcrb = arrayfun(@(r) max(r.metrics.pcrb_ratio), rec);
    comm = arrayfun(@(r) min(r.metrics.nominal_sinr_margin_dB), rec);
    sensing = arrayfun(@(r) min(r.metrics.sensing_sinr_margin_dB), rec);
    stats.pcrb(q,:) = quantile(pcrb,[.1 .5 .9]);
    stats.comm(q,:) = quantile(comm,[.1 .5 .9]);
    stats.sensing(q,:) = quantile(sensing,[.1 .5 .9]);
end

fig = figure('Visible','off','Position',[100 100 1200 350]);
tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
draw_band(nexttile, nreq, stats.pcrb, 'Worst-target PCRB ratio', ...
    'max target PCRB / tracking threshold', [0.995 1.005], true);
draw_band(nexttile, nreq, stats.comm, 'Worst-user nominal SINR margin', ...
    'min_k SINR margin (dB)', [], false);
draw_band(nexttile, nreq, stats.sensing, 'Worst-target sensing SINR margin', ...
    'min_p sensing-SINR margin (dB)', [], false);
exportgraphics(fig,fullfile(output_dir,'fig4_qos_vs_cluster_size.png'),'Resolution',300);
close(fig);
save(fullfile(output_dir,'fig4_qos_statistics.mat'),'stats');
end

function draw_band(ax,x,values,title_text,ylabel_text,ylims,draw_one)
hold(ax,'on'); grid(ax,'on');
fill(ax,[x fliplr(x)],[values(:,1).' fliplr(values(:,3).')], ...
    [0.75 0.85 1],'EdgeColor','none','FaceAlpha',0.65,'DisplayName','10th--90th percentile');
plot(ax,x,values(:,2),'-o','Color',[0 0.25 0.7],'LineWidth',1.8, ...
    'MarkerFaceColor',[0 0.25 0.7],'DisplayName','Median');
if draw_one
    boundary = yline(ax,1,'--','Color',[0.75 0 0]);
    boundary.DisplayName = 'Constraint boundary';
end
xlabel(ax,'Required sensing APs per target, N_{req}'); ylabel(ax,ylabel_text);
title(ax,title_text); xticks(ax,x); xlim(ax,[min(x)-.25,max(x)+.25]);
if ~isempty(ylims), ylim(ax,ylims); end
if draw_one, legend(ax,'Location','southwest'); end
end
