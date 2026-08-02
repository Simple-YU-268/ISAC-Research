function summary = plot_extended_physical_mc(results_file)
%PLOT_EXTENDED_PHYSICAL_MC  Plot final extended physical-setting campaign.

if nargin<1
    results_file=fullfile(pwd,'experiment_packages','v1.0','results', ...
        'extended_physical_mc','extended_physical_mc_final.mat');
end
raw=load(results_file,'campaign'); campaign=raw.campaign;
summary=struct2table(campaign.summary);
summary.calibration_rejected=false(height(summary),1);
for c=1:numel(campaign.configurations)
    records=campaign.records{c};
    rejected=~isempty(records) && all(strlength(string({records.error_id}))>0);
    if rejected
        summary.calibration_rejected(summary.config_id==string(campaign.configurations(c).id))=true;
    end
end
out_dir=fileparts(results_file);
writetable(summary,fullfile(out_dir,'extended_physical_mc_summary.csv'));

families=["ap_count","antennas_per_ap","ue_load","target_load", ...
    "area_side_m","power_budget_dBm"];
labels={"AP count M","Antennas per AP N_t","UE load K", ...
    "Target load P","Area side length (m)","Per-AP budget (dBm)"};
fig=figure('Visible','off','Position',[100 100 1250 700]);
layout=tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
for q=1:numel(families)
    ax=nexttile(layout); plot_factor(ax,summary,families(q),labels{q});
end
exportgraphics(fig,fullfile(out_dir,'fig11_extended_physical_factors.png'),'Resolution',300);
close(fig);

stress=summary(summary.family=="stress",:);
if ~isempty(stress)
    fig=figure('Visible','off','Position',[100 100 980 380]);
    layout=tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    nexttile(layout); grouped_metric(stress,'feasibility_rate','Physical feasibility');
    nexttile(layout); grouped_metric(stress,'mean_power_mW','Conditional mean power (mW)');
    exportgraphics(fig,fullfile(out_dir,'fig12_pressure_geometries.png'),'Resolution',300);
    close(fig);
end
end

function plot_factor(ax,T,family,xlabel_text)
rows=T(T.family==family,:); methods=unique(rows.method,'stable');
levels=unique(str2double(rows.level)); levels=sort(levels(isfinite(levels)));
colors=[0.000 0.447 0.741; 0.850 0.325 0.098; 0.929 0.694 0.125];
markers={'o','s','d'};
yyaxis(ax,'left'); hold(ax,'on'); grid(ax,'on');
for j=1:numel(methods)
    y=NaN(size(levels));
    for i=1:numel(levels)
        r=rows(rows.method==methods(j) & str2double(rows.level)==levels(i),:);
        if ~isempty(r) && ~r.calibration_rejected(1), y(i)=r.mean_power_mW(1); end
    end
    plot(ax,levels,y,['-' markers{j}],'Color',colors(j,:), ...
        'LineWidth',1.5,'MarkerFaceColor','w', ...
        'DisplayName',char(methods(j)));
end
ylabel(ax,'Conditional mean power (mW)');
yyaxis(ax,'right'); hold(ax,'on');
for j=1:numel(methods)
    y=NaN(size(levels));
    for i=1:numel(levels)
        r=rows(rows.method==methods(j) & str2double(rows.level)==levels(i),:);
        if ~isempty(r) && ~r.calibration_rejected(1), y(i)=100*r.feasibility_rate(1); end
    end
    plot(ax,levels,y,['--' markers{j}],'Color',colors(j,:), ...
        'LineWidth',1.1,'MarkerFaceColor',colors(j,:), ...
        'HandleVisibility','off');
end
ylim(ax,[0 105]); ylabel(ax,'Feasibility (%)'); xlabel(ax,xlabel_text);
title(ax,replace(char(family),'_',' '),'Interpreter','none');
if family=="ap_count", legend(ax,'Location','best'); end
end

function grouped_metric(T,metric,ylabel_text)
stress=unique(T.level,'stable'); methods=unique(T.method,'stable');
Y=NaN(numel(stress),numel(methods));
for i=1:numel(stress)
    for j=1:numel(methods)
        r=T(T.level==stress(i)&T.method==methods(j),:);
        if ~isempty(r), Y(i,j)=r.(metric)(1); end
    end
end
bar(Y); grid on; set(gca,'XTickLabel',replace(stress,"_"," "));
ylabel(ylabel_text); legend(methods,'Location','best');
end
