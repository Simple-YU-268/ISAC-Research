function summary = plot_large_scale_fixed_nreq_validation(result_file, output_dir)
%PLOT_LARGE_SCALE_FIXED_NREQ_VALIDATION Summarize a fixed-Nreq MC campaign.
%   Produces a paper-ready three-panel figure and CSV table containing only
%   feasibility, conditional total power, and runtime.  Conditional power is
%   explicitly labelled because infeasible heuristic samples must not be
%   silently treated as zero-power observations.

if nargin < 2 || isempty(output_dir)
    output_dir = fileparts(result_file);
end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
raw = load(result_file, 'output');
output = raw.output;
assert(numel(output.nreq_list) == 1, ...
    'This helper is for a single fixed N_req experiment.');

labels = string(output.labels);
records = output.records;
n_seed = numel(output.seeds);
n_method = numel(labels);
summary = struct();
summary.N_req = output.nreq_list;
summary.num_seeds = n_seed;
summary.labels = labels;
summary.feasible_count = zeros(n_method,1);
summary.feasibility_rate = zeros(n_method,1);
summary.mean_power_mW = NaN(n_method,1);
summary.mean_runtime_s = NaN(n_method,1);
summary.mean_pcrb_ratio = NaN(n_method,1);

for m = 1:n_method
    methods = [records.methods];
    methods = methods(m:n_method:end);
    feasible = [methods.feasible];
    summary.feasible_count(m) = nnz(feasible);
    summary.feasibility_rate(m) = mean(feasible);
    summary.mean_runtime_s(m) = mean([methods.time_s], 'omitnan');
    if any(feasible)
        summary.mean_power_mW(m) = 1e3 * mean([methods(feasible).power_W], 'omitnan');
        metrics = [methods(feasible).metrics];
        summary.mean_pcrb_ratio(m) = mean([metrics.mean_pcrb_ratio], 'omitnan');
    end
end

tbl = table(labels(:), summary.feasible_count, repmat(n_seed,n_method,1), ...
    summary.feasibility_rate, summary.mean_power_mW, summary.mean_runtime_s, ...
    summary.mean_pcrb_ratio, 'VariableNames', {'Method','Feasible','Total', ...
    'FeasibilityRate','ConditionalMeanPower_mW','MeanRuntime_s','MeanPCRBRatio'});
writetable(tbl, fullfile(output_dir, 'table_m12_fixed_nreq_validation.csv'));
save(fullfile(output_dir, 'm12_fixed_nreq_validation_summary.mat'), 'summary', 'tbl');

method_names = categorical(labels, labels);
fig = figure('Color','w','Position',[100,100,1250,360]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile;
bar(method_names, 100*summary.feasibility_rate, 0.65, 'FaceColor',[0.2 0.45 0.75]);
ylim([0 105]); ylabel('Physical feasibility (%)'); grid on;
title(sprintf('M=%d, K=%d, P=%d, N_{req}=%d', output.configuration.M, ...
    output.configuration.K, output.configuration.P, output.nreq_list));

nexttile;
bar(method_names, summary.mean_power_mW, 0.65, 'FaceColor',[0.85 0.4 0.2]);
ylabel('Mean power over feasible cases (mW)'); grid on;
title('Conditional energy cost');

nexttile;
bar(method_names, summary.mean_runtime_s, 0.65, 'FaceColor',[0.35 0.65 0.4]);
ylabel('Mean wall-clock time per scenario (s)'); grid on;
title('Computation cost');

exportgraphics(fig, fullfile(output_dir, 'fig11_m12_scalability_validation.png'), ...
    'Resolution', 300);
close(fig);
end
