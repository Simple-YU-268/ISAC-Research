function stats = plot_csi_robustness(result_file, output_dir)
%PLOT_CSI_ROBUSTNESS  Robust and nominal CSI performance statistics.

if nargin < 1 || isempty(result_file)
    result_file=fullfile(pwd,'..','..','experiment_packages','v1.0','results', ...
        'csi_robustness','csi_robustness_final.mat');
end
if nargin < 2 || isempty(output_dir)
    output_dir=fullfile(pwd,'..','..','experiment_packages','v1.0','figures');
end
if ~exist(output_dir,'dir'), mkdir(output_dir); end
loaded=load(result_file,'output'); output=loaded.output;
eps_list=output.eps_list(:).'; records=output.records;
methods={'robust','nominal'}; stats.eps=eps_list;
for m=1:2
    stats.(methods{m}).feasibility=NaN(numel(eps_list),1);
    stats.(methods{m}).outage=NaN(numel(eps_list),3);
    stats.(methods{m}).power=NaN(numel(eps_list),3);
end
for q=1:numel(eps_list)
    for m=1:2
        method=methods{m}; values=[records(q,:).(method)];
        feasible=[values.feasible];
        stats.(method).feasibility(q)=mean(feasible);
        if any(feasible)
            outage=[values(feasible).outage_probability];
            power=[values(feasible).power_W];
            stats.(method).outage(q,:)=quantile(outage,[.1 .5 .9]);
            stats.(method).power(q,:)=quantile(power,[.1 .5 .9]);
        end
    end
end

fig=figure('Visible','off','Position',[100 100 1200 350]);
tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
ax=nexttile; hold(ax,'on'); grid(ax,'on');
plot(ax,eps_list,100*stats.robust.feasibility,'-o','LineWidth',1.7,'DisplayName','Robust');
plot(ax,eps_list,100*stats.nominal.feasibility,'-s','LineWidth',1.7,'DisplayName','Nominal CSI');
xlabel(ax,'CSI uncertainty radius, \epsilon_h'); ylabel(ax,'Design feasibility (%)');
ylim(ax,[0 105]); title(ax,'Certified design feasibility'); legend(ax,'Location','southwest');

ax=nexttile; draw_two_bands(ax,eps_list,100*stats.robust.outage,100*stats.nominal.outage, ...
    'System outage probability (%)','Empirical outage under ball perturbations');
ax=nexttile; draw_two_bands(ax,eps_list,1e3*stats.robust.power,1e3*stats.nominal.power, ...
    'Total transmit power (mW)','Power cost of robustness');
exportgraphics(fig,fullfile(output_dir,'fig9_csi_robustness.png'),'Resolution',300);
close(fig); save(fullfile(output_dir,'fig9_csi_robustness_statistics.mat'),'stats');
end

function draw_two_bands(ax,x,robust,nominal,ylabel_text,title_text)
hold(ax,'on'); grid(ax,'on');
fill(ax,[x fliplr(x)],[robust(:,1).' fliplr(robust(:,3).')], ...
    [0.65 0.80 1],'EdgeColor','none','FaceAlpha',0.5,'HandleVisibility','off');
fill(ax,[x fliplr(x)],[nominal(:,1).' fliplr(nominal(:,3).')], ...
    [1 0.75 0.62],'EdgeColor','none','FaceAlpha',0.5,'HandleVisibility','off');
plot(ax,x,robust(:,2),'-o','Color',[0 .28 .75],'LineWidth',1.8, ...
    'MarkerFaceColor',[0 .28 .75],'DisplayName','Robust');
plot(ax,x,nominal(:,2),'-s','Color',[.85 .25 .05],'LineWidth',1.8, ...
    'MarkerFaceColor',[.85 .25 .05],'DisplayName','Nominal CSI');
xlabel(ax,'CSI uncertainty radius, \epsilon_h'); ylabel(ax,ylabel_text);
title(ax,title_text); legend(ax,'Location','best');
end
