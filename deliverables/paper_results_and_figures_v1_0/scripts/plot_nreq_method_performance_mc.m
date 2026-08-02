function summary = plot_nreq_method_performance_mc(result_file, figure_dir)
%PLOT_NREQ_METHOD_PERFORMANCE_MC  Aggregate physical method comparison.
%   Means and percentile bands are conditional on physical feasibility.  The
%   corresponding feasible count is saved in the summary table so random or
%   other weak baselines cannot be misrepresented through missing values.
if nargin < 1 || isempty(result_file)
    result_file = fullfile(pwd,'experiment_packages','v1.0','results', ...
        'nreq_method_performance_mc','nreq_method_performance_final.mat');
end
if nargin < 2 || isempty(figure_dir)
    figure_dir = fullfile(fileparts(fileparts(fileparts(result_file))),'figures');
end
if ~exist(figure_dir,'dir'), mkdir(figure_dir); end

L = load(result_file,'output'); output = L.output;
nreq = output.nreq_list(:).'; labels = string(output.labels(:).');
R = output.records; Q = numel(nreq); J = numel(labels);
fields = {'power_W','mean_pcrb_ratio','sum_rate_bpsHz','mean_sensing_sinr_dB', ...
    'num_nonzero_sensing_aps','num_nonzero_sensing_pairs','time_s'};
S = struct(); S.nreq = nreq; S.labels = labels;
for f = 1:numel(fields)
    S.([fields{f} '_mean']) = NaN(Q,J);
    S.([fields{f} '_p10']) = NaN(Q,J);
    S.([fields{f} '_p90']) = NaN(Q,J);
end
S.feasible_count = zeros(Q,J); S.total_count = zeros(Q,J);
S.active_pair_match_rate = NaN(Q,J); S.sdr_power_mean_W = NaN(Q,1);

for q = 1:Q
    rec_q = R(q,:); S.sdr_power_mean_W(q) = mean([rec_q.sdr_lower_bound_W],'omitnan');
    all_methods = [rec_q.methods];
    for j = 1:J
        methods = all_methods(j:J:end);
        feasible = [methods.feasible]; S.total_count(q,j) = numel(methods);
        S.feasible_count(q,j) = nnz(feasible);
        methods = methods(feasible);
        if isempty(methods), continue; end
        pair_match = false(1,numel(methods));
        for u=1:numel(methods)
            pair_match(u) = methods(u).metrics.num_nonzero_sensing_pairs == ...
                output.records(q,1).N_req * output.configuration.P;
        end
        S.active_pair_match_rate(q,j) = mean(pair_match);
        for f = 1:numel(fields)
            name = fields{f}; values = extract_field(methods,name);
            S.([name '_mean'])(q,j) = mean(values,'omitnan');
            S.([name '_p10'])(q,j) = prctile(values,10);
            S.([name '_p90'])(q,j) = prctile(values,90);
        end
    end
end

colors = lines(J);
fig = figure('Visible','off','Position',[100 100 1250 700]);
tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
ax=nexttile; draw_metric(ax,nreq,100*S.feasible_count./S.total_count,[],labels,colors, ...
    'Physical feasibility (%)',[0 105]);
legend(ax,'Location','southwest','FontSize',8);
draw_metric(nexttile,nreq,1e3*S.power_W_mean,1e3*[S.power_W_p10 S.power_W_p90],labels,colors, ...
    'Conditional mean power (mW)',[]);
draw_metric(nexttile,nreq,S.mean_pcrb_ratio_mean,[S.mean_pcrb_ratio_p10 S.mean_pcrb_ratio_p90],labels,colors, ...
    'Mean normalized PCRB',[]);
draw_metric(nexttile,nreq,S.sum_rate_bpsHz_mean,[S.sum_rate_bpsHz_p10 S.sum_rate_bpsHz_p90],labels,colors, ...
    'Conditional mean sum rate (bit/s/Hz)',[]);
draw_metric(nexttile,nreq,S.mean_sensing_sinr_dB_mean,[S.mean_sensing_sinr_dB_p10 S.mean_sensing_sinr_dB_p90],labels,colors, ...
    'Mean sensing SINR (dB)',[]);
draw_metric(nexttile,nreq,S.num_nonzero_sensing_aps_mean,[S.num_nonzero_sensing_aps_p10 S.num_nonzero_sensing_aps_p90],labels,colors, ...
    'Mean nonzero sensing APs',[]);
exportgraphics(fig,fullfile(figure_dir,'fig10_method_comparison_vs_nreq.png'),'Resolution',300); close(fig);

rows = table();
for q=1:Q
    for j=1:J
        rows = [rows; table(nreq(q),labels(j),S.feasible_count(q,j),S.total_count(q,j), ...
            1e3*S.power_W_mean(q,j),S.mean_pcrb_ratio_mean(q,j), ...
            S.sum_rate_bpsHz_mean(q,j),S.mean_sensing_sinr_dB_mean(q,j), ...
            S.num_nonzero_sensing_aps_mean(q,j),S.active_pair_match_rate(q,j), ...
            'VariableNames',{'N_req','Method','Feasible','Total','MeanPower_mW', ...
            'MeanPCRBRatio','MeanSumRate_bpsHz','MeanSensingSINR_dB', ...
            'MeanNonzeroSensingAPs','ActivePairInvariantRate'})]; %#ok<AGROW>
    end
end
writetable(rows,fullfile(figure_dir,'table_method_comparison_vs_nreq.csv'));
save(fullfile(figure_dir,'method_comparison_summary.mat'),'S','rows');
summary = S;
end

function values = extract_field(methods,name)
if strcmp(name,'time_s') || strcmp(name,'power_W')
    values = [methods.(name)]; return;
end
values = arrayfun(@(x)x.metrics.(name),methods);
end

function draw_metric(ax,x,mean_values,bands,labels,colors,ylabel_text,ylims)
hold(ax,'on'); grid(ax,'on');
for j=1:numel(labels)
    y=mean_values(:,j).';
    if ~isempty(bands)
        lo=bands(:,j).'; hi=bands(:,j+numel(labels)).';
        fill(ax,[x fliplr(x)],[lo fliplr(hi)],colors(j,:),'FaceAlpha',.12, ...
            'EdgeColor','none','HandleVisibility','off');
    end
    plot(ax,x,y,'-o','LineWidth',1.6,'Color',colors(j,:), ...
        'MarkerFaceColor',colors(j,:),'DisplayName',labels(j));
end
xlabel(ax,'N_{req}'); ylabel(ax,ylabel_text); xticks(ax,x); xlim(ax,[min(x)-.2,max(x)+.2]);
if ~isempty(ylims), ylim(ax,ylims); end
end
