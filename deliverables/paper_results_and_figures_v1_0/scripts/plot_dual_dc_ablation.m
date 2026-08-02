function summary = plot_dual_dc_ablation(result_file, figure_dir)
%PLOT_DUAL_DC_ABLATION Plot the current-model rank/binary penalty ablation.
%   The four modes are SDR relaxation, rank-DC only, binary-DC only, and
%   dual DC.  All quantities come from the saved common-seed raw results.

if nargin < 1 || isempty(result_file)
    result_file = fullfile(pwd,'experiment_packages','v1.0','results', ...
        'participation_dual_dc_ablation_seeds6to30','final.mat');
end
if nargin < 2 || isempty(figure_dir)
    figure_dir = fullfile(pwd,'experiment_packages','v1.0','figures');
end
if ~exist(figure_dir,'dir'), mkdir(figure_dir); end

raw = load(result_file, 'summary', 'modes', 'opt');
summary = raw.summary;
labels = categorical(string({raw.modes.label}), string({raw.modes.label}));
n_seeds = numel(raw.opt.Seeds);

fig = figure('Color','w','Position',[100 100 1280 320]);
tiledlayout(1,4,'TileSpacing','compact','Padding','compact');

nexttile;
bar(labels,100*summary.feasibility_rate,0.65,'FaceColor',[0.20 0.45 0.75]);
grid on; ylim([0 105]); ylabel('Physical feasibility (%)');
title(sprintf('%d common scenarios',n_seeds));

nexttile;
bar(labels,max(summary.binary_distance_median,1e-12),0.65,'FaceColor',[0.90 0.55 0.15]);
set(gca,'YScale','log'); grid on; ylabel('Median binary distance');
yline(1e-4,'--','10^{-4} reference','Color',[0.75 0 0]);
title('Binary recovery');

nexttile;
bar(labels,max(summary.rank_residual_median,1e-12),0.65,'FaceColor',[0.35 0.65 0.40]);
set(gca,'YScale','log'); grid on; ylabel('Median rank residual');
title('Rank-one residual');

nexttile;
yyaxis left
bar(labels,summary.power_gap_median_pct,0.65,'FaceColor',[0.55 0.35 0.70]);
ylabel('Median gap to SDR (%)'); grid on;
yyaxis right
plot(labels,summary.runtime_median_s,'-o','Color',[0.1 0.1 0.1], ...
    'LineWidth',1.5,'MarkerFaceColor',[0.1 0.1 0.1]);
ylabel('Median runtime (s)');
title('Energy and runtime');

exportgraphics(fig,fullfile(figure_dir,'fig2_dual_dc_ablation.png'),'Resolution',300);
save(fullfile(figure_dir,'fig2_dual_dc_ablation_summary.mat'),'summary','labels','n_seeds');
close(fig);
end
