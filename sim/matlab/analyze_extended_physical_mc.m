function report = analyze_extended_physical_mc(results_file)
%ANALYZE_EXTENDED_PHYSICAL_MC  Paired evidence for the physical MC campaign.
%   Produces reproducible tables that compare the proposed recovery against
%   the FIM-greedy and oracle nearest-AP associations on common realizations.
%   A positive paired power gap means that the proposed method uses less power.

if nargin < 1
    results_file = fullfile(pwd,'experiment_packages','v1.0','results', ...
        'extended_physical_mc','extended_physical_mc_final.mat');
end
raw = load(results_file,'campaign'); campaign = raw.campaign;
out_dir = fileparts(results_file);
methods = string(campaign.methods);
idx_proposed = find(methods == "Proposed",1);
comparators = ["FIM-greedy","Nearest-AP"];

pair_rows = repmat(empty_pair_row(),0,1);
qos_rows = repmat(empty_qos_row(),0,1);
failure_rows = repmat(empty_failure_row(),0,1);
for c = 1:numel(campaign.configurations)
    cfg = campaign.configurations(c);
    records = campaign.records{c};
    method_data = cell(1,numel(methods));
    for j = 1:numel(methods)
        method_data{j} = extract_method(records,j);
        qos_rows(end+1,1) = make_qos_row(cfg,methods(j),method_data{j}); %#ok<AGROW>
    end
    feasible_matrix = cell2mat(cellfun(@(x)x.feasible(:),method_data, ...
        'UniformOutput',false));
    failure_rows(end+1,1) = make_failure_row(cfg,feasible_matrix); %#ok<AGROW>
    for comp = comparators
        idx_comp = find(methods == comp,1);
        paired = method_data{idx_proposed}.feasible & method_data{idx_comp}.feasible;
        p_prop = 1e3 * method_data{idx_proposed}.power_W(paired);
        p_comp = 1e3 * method_data{idx_comp}.power_W(paired);
        pair_rows(end+1,1) = make_pair_row(cfg,comp,p_prop,p_comp); %#ok<AGROW>
    end
end

pairwise = struct2table(pair_rows);
qos = struct2table(qos_rows);
failures = struct2table(failure_rows);
writetable(pairwise,fullfile(out_dir,'extended_pairwise_power_comparison.csv'));
writetable(qos,fullfile(out_dir,'extended_qos_audit.csv'));
writetable(failures,fullfile(out_dir,'extended_failure_classification.csv'));
report = struct('pairwise',pairwise,'qos',qos,'failures',failures);
save(fullfile(out_dir,'extended_physical_mc_analysis.mat'),'report','-v7.3');
fprintf('Paired analysis complete: %d comparisons, %d QoS rows.\n', ...
    height(pairwise),height(qos));
end

function x = extract_method(records,j)
n = numel(records);
x = struct('feasible',false(n,1),'power_W',NaN(n,1), ...
    'mean_pcrb_ratio',NaN(n,1),'min_comm_margin_dB',NaN(n,1), ...
    'min_sensing_margin_dB',NaN(n,1),'time_s',NaN(n,1));
for i = 1:n
    y = records(i).methods(j);
    x.feasible(i) = y.feasible; x.power_W(i) = y.power_W;
    x.mean_pcrb_ratio(i) = y.mean_pcrb_ratio;
    x.min_comm_margin_dB(i) = y.min_comm_margin_dB;
    x.min_sensing_margin_dB(i) = y.min_sensing_margin_dB;
    x.time_s(i) = y.time_s;
end
end

function row = make_pair_row(cfg,comparator,p_prop,p_comp)
row = empty_pair_row(); row.config_id = string(cfg.id); row.family = string(cfg.family);
row.level = string(cfg.level); row.comparator = comparator; row.n_paired = numel(p_prop);
if isempty(p_prop), return; end
gap = p_comp - p_prop;
row.proposed_mean_mW = mean(p_prop); row.comparator_mean_mW = mean(p_comp);
row.mean_power_gap_mW = mean(gap);
row.ci95_halfwidth_mW = 1.96 * std(gap) / sqrt(numel(gap));
row.proposed_reduction_pct = 100 * mean(gap ./ p_comp);
end

function row = make_qos_row(cfg,method,x)
row = empty_qos_row(); row.config_id = string(cfg.id); row.family = string(cfg.family);
row.level = string(cfg.level); row.method = method; f = x.feasible;
row.n_feasible = nnz(f);
if any(f)
    row.max_mean_pcrb_ratio = max(x.mean_pcrb_ratio(f));
    row.min_comm_margin_dB = min(x.min_comm_margin_dB(f));
    row.min_sensing_margin_dB = min(x.min_sensing_margin_dB(f));
    row.median_runtime_s = median(x.time_s(f));
end
end

function row = make_failure_row(cfg,f)
row = empty_failure_row(); row.config_id = string(cfg.id); row.family = string(cfg.family);
row.level = string(cfg.level); row.total = size(f,1);
row.all_infeasible = nnz(~any(f,2)); row.proposed_only = nnz(f(:,1) & ~f(:,2) & ~f(:,3));
row.proposed_failed_others_feasible = nnz(~f(:,1) & any(f(:,2:3),2));
row.all_feasible = nnz(all(f,2));
end

function row = empty_pair_row()
row = struct('config_id',"",'family',"",'level',"",'comparator',"", ...
    'n_paired',0,'proposed_mean_mW',NaN,'comparator_mean_mW',NaN, ...
    'mean_power_gap_mW',NaN,'ci95_halfwidth_mW',NaN,'proposed_reduction_pct',NaN);
end

function row = empty_qos_row()
row = struct('config_id',"",'family',"",'level',"",'method',"", ...
    'n_feasible',0,'max_mean_pcrb_ratio',NaN,'min_comm_margin_dB',NaN, ...
    'min_sensing_margin_dB',NaN,'median_runtime_s',NaN);
end

function row = empty_failure_row()
row = struct('config_id',"",'family',"",'level',"",'total',0, ...
    'all_infeasible',0,'proposed_only',0,'proposed_failed_others_feasible',0, ...
    'all_feasible',0);
end
