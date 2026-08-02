function summary = plot_m12_nreq_scalability(nreq3_file, sweep_file, figure_dir)
%PLOT_M12_NREQ_SCALABILITY Plot a common-seed M12 cluster-size comparison.
%   Nreq=3 is read from the eight-seed validation but restricted to the seeds
%   present in SWEEP_FILE.  This preserves common-seed comparisons across all
%   displayed Nreq values.

if nargin < 1 || isempty(nreq3_file)
    nreq3_file = fullfile(pwd,'experiment_packages','v1.0','results', ...
        'large_scale_algorithm_validation','M12_K6_P3_Nreq3_seed01to08_workers4_t60', ...
        'nreq_method_performance_final.mat');
end
if nargin < 2 || isempty(sweep_file)
    sweep_file = fullfile(pwd,'experiment_packages','v1.0','results', ...
        'large_scale_algorithm_validation','M12_K6_P3_nreq2_4_5_seed01to05_workers4_t60', ...
        'nreq_method_performance_final.mat');
end
if nargin < 3 || isempty(figure_dir)
    figure_dir = fileparts(sweep_file);
end
if ~exist(figure_dir,'dir'), mkdir(figure_dir); end

A = load(nreq3_file,'output'); B = load(sweep_file,'output');
assert(isequal(A.output.labels,B.output.labels),'Method order mismatch.');
common_seeds = B.output.seeds(:).';
idx3 = ismember(A.output.seeds,common_seeds);
assert(nnz(idx3)==numel(common_seeds),'Nreq=3 output misses a sweep seed.');

nreq = sort([B.output.nreq_list(:); A.output.nreq_list(:)]).';
nreq = unique(nreq,'stable');
labels = string(B.output.labels); nm = numel(labels); nq = numel(nreq);
summary = struct('nreq',nreq,'seeds',common_seeds,'labels',labels, ...
    'feasibility_rate',zeros(nm,nq),'mean_power_mW',NaN(nm,nq), ...
    'mean_runtime_s',NaN(nm,nq),'mean_pcrb_ratio',NaN(nm,nq));

for q=1:nq
    if nreq(q)==A.output.nreq_list
        records = A.output.records(1,idx3);
    else
        row = find(B.output.nreq_list==nreq(q),1);
        records = B.output.records(row,:);
    end
    for m=1:nm
        methods = [records.methods]; methods = methods(m:nm:end);
        feasible = [methods.feasible];
        summary.feasibility_rate(m,q) = mean(feasible);
        summary.mean_runtime_s(m,q) = mean([methods.time_s],'omitnan');
        if any(feasible)
            summary.mean_power_mW(m,q) = 1e3*mean([methods(feasible).power_W],'omitnan');
            metrics = [methods(feasible).metrics];
            summary.mean_pcrb_ratio(m,q) = mean([metrics.mean_pcrb_ratio],'omitnan');
        end
    end
end

fig=figure('Color','w','Position',[100 100 1200 330]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
colors=lines(nm);
nexttile; hold on; grid on;
for m=1:nm, plot(nreq,100*summary.feasibility_rate(m,:),'-o','LineWidth',1.6,'Color',colors(m,:)); end
xlabel('N_{req}'); ylabel('Physical feasibility (%)'); ylim([0 105]); legend(labels,'Location','southwest');
nexttile; hold on; grid on;
for m=1:nm, plot(nreq,summary.mean_power_mW(m,:),'-o','LineWidth',1.6,'Color',colors(m,:)); end
xlabel('N_{req}'); ylabel('Mean power over feasible cases (mW)');
nexttile; hold on; grid on;
for m=1:nm, plot(nreq,summary.mean_runtime_s(m,:),'-o','LineWidth',1.6,'Color',colors(m,:)); end
xlabel('N_{req}'); ylabel('Mean wall-clock time (s)');
sgtitle(sprintf('M=12, K=6, P=3; %d common seeds',numel(common_seeds)));
exportgraphics(fig,fullfile(figure_dir,'fig12_m12_nreq_scalability.png'),'Resolution',300);
close(fig);

rows = table();
for q=1:nq
    rows = [rows; table(repmat(nreq(q),nm,1),labels(:),summary.feasibility_rate(:,q), ...
        summary.mean_power_mW(:,q),summary.mean_runtime_s(:,q),summary.mean_pcrb_ratio(:,q), ...
        'VariableNames',{'N_req','Method','FeasibilityRate','ConditionalMeanPower_mW', ...
        'MeanRuntime_s','MeanPCRBRatio'})]; %#ok<AGROW>
end
writetable(rows,fullfile(figure_dir,'table_m12_nreq_scalability.csv'));
save(fullfile(figure_dir,'m12_nreq_scalability_summary.mat'),'summary','rows');
end
