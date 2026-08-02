function result = run_statistical_double_dc_convergence(varargin)
%RUN_STATISTICAL_DOUBLE_DC_CONVERGENCE  Formal DC-SCA stabilization study.
%   Reports common-seed statistics for the continuous phase only.  It does
%   not claim a physical solution at an iteration; physical feasibility is
%   always evaluated separately after fixed-b re-optimization.
%
%   The reported diagnostics are deliberately topology-oriented:
%     (i) median/max binary distance with interquartile interval;
%    (ii) fraction of valid runs whose top-N support is unchanged from the
%         preceding iteration;
%   (iii) fraction of valid runs that are recovery-ready, namely rank-small,
%         support-stable, and within the configured rounding distance.
%   Total power is shown only as a continuation diagnostic because increasing
%   DC penalties means it is not a monotone descent objective.

ip = inputParser;
addParameter(ip,'Seeds',1:25,@(x)isnumeric(x)&&isvector(x)&&all(x>=0));
addParameter(ip,'T_max',6,@(x)isnumeric(x)&&isscalar(x)&&x>=2);
addParameter(ip,'Output_dir',fullfile(pwd,'experiment_packages','v1.0', ...
    'results','statistical_double_dc_convergence'),@(x)ischar(x)||isstring(x));
addParameter(ip,'Figure_dir',fullfile(pwd,'experiment_packages','v1.0', ...
    'figures'),@(x)ischar(x)||isstring(x));
addParameter(ip,'Solver_time_limit_s',12,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(ip,'Rounding_distance_max',0.05,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(ip,'Resume',true,@(x)islogical(x)&&isscalar(x));
parse(ip,varargin{:}); opt=ip.Results;
out_dir=char(opt.Output_dir); fig_dir=char(opt.Figure_dir);
if ~exist(out_dir,'dir'), mkdir(out_dir); end
if ~exist(fig_dir,'dir'), mkdir(fig_dir); end

records = repmat(empty_record(opt.T_max),numel(opt.Seeds),1);
checkpoint_file=fullfile(out_dir,'checkpoint.mat');
if opt.Resume && exist(checkpoint_file,'file')
    checkpoint=load(checkpoint_file,'records','opt');
    if isfield(checkpoint,'records') && numel(checkpoint.records)==numel(records)
        old_seeds=[checkpoint.records.seed];
        if all(isfinite(old_seeds) | isnan(old_seeds))
            records=checkpoint.records;
            fprintf('[statistical convergence] resuming from checkpoint\n');
        end
    end
end
for s=1:numel(opt.Seeds)
    seed=opt.Seeds(s);
    if records(s).seed==seed && records(s).status~="not_run"
        fprintf('[statistical convergence] %d/%d, seed=%d already complete\n', ...
            s,numel(opt.Seeds),seed);
        continue;
    end
    fprintf('[statistical convergence] %d/%d, seed=%d\n',s,numel(opt.Seeds),seed);
    prm=generate_scenario(6,2,3,2,2,20,'auto','AreaSize',400, ...
        'N_req',3,'eps_h',0.05,'seed',seed);
    prm.solver='mosek';
    prm.mosek_max_time=opt.Solver_time_limit_s;
    prm.enable_topology_early_stop=false;
    prm.successive_fixing_enabled=false;
    prm.skip_recovery=true;
    prm.adaptive_binary_penalty=true;
    prm.eta_b_growth=2;
    prm.eta_b_max=50;
    timer=tic;
    res=baseline_alg2(prm,opt.T_max,1e-5,1,1,1,false,opt.T_max);
    records(s).seed=seed;
    records(s).runtime_s=toc(timer);
    records(s).status=string(get_field(res,'status','unknown'));
    records(s).iterations=get_field(res,'dc_iterations',0);
    records(s).sdr_power_W=get_field(res,'sdr_power',NaN);
    records(s).sdr_binary_distance=get_field(res,'sdr_binary_distance',NaN);
    records(s).sdr_rank_residual=get_field(res,'sdr_rank_residual',NaN);
    records(s).power_W=pad_trace(get_field(res,'power_trace',[]),opt.T_max);
    records(s).binary_distance=pad_trace(get_field(res,'binary_distance_trace',[]),opt.T_max);
    records(s).rank_residual=pad_trace(get_field(res,'rank_residual_trace',[]),opt.T_max);
    records(s).topology_changed=logical(pad_trace(double(get_field(res, ...
        'topology_changed_trace',[])),opt.T_max));
    records(s).eta_b=pad_trace(get_field(res,'eta_b_trace',[]),opt.T_max);
    save(checkpoint_file,'records','opt');
end

result=aggregate_records(records,opt);
save(fullfile(out_dir,'statistical_double_dc_convergence_final.mat'),'records','result','opt');
write_summary_csv(result,fullfile(out_dir,'statistical_double_dc_convergence_summary.csv'));
plot_result(result,fullfile(fig_dir,'fig5_statistical_double_dc_convergence.png'));
end

function r=empty_record(T)
r=struct('seed',NaN,'runtime_s',NaN,'status',"not_run",'iterations',0, ...
    'sdr_power_W',NaN,'sdr_binary_distance',NaN,'sdr_rank_residual',NaN, ...
    'power_W',NaN(1,T),'binary_distance',NaN(1,T), ...
    'rank_residual',NaN(1,T),'topology_changed',false(1,T), ...
    'eta_b',NaN(1,T));
end

function y=pad_trace(x,T)
y=NaN(1,T); x=x(:).'; y(1:min(T,numel(x)))=x(1:min(T,numel(x)));
end

function result=aggregate_records(records,opt)
T=opt.T_max; n=numel(records);
B=[vertcat(records.sdr_binary_distance), vertcat(records.binary_distance)];
R=[vertcat(records.sdr_rank_residual), vertcat(records.rank_residual)];
P=[vertcat(records.sdr_power_W), vertcat(records.power_W)];
C=vertcat(records.topology_changed);
E=[zeros(n,1), vertcat(records.eta_b)];
valid=isfinite(B);
stable=false(n,T+1);
stable(:,2:end)=valid(:,2:end) & ~C;
rank_ok=R<=1e-5;
ready=stable & rank_ok & B<=opt.Rounding_distance_max;
result.iteration=0:T;
result.n_total=n;
result.n_valid=sum(valid,1);
result.binary_median=median(B,1,'omitnan');
result.binary_q25=prctile(B,25,1);
result.binary_q75=prctile(B,75,1);
result.rank_median=median(R,1,'omitnan');
result.power_median_mW=1e3*median(P,1,'omitnan');
result.power_q25_mW=1e3*prctile(P,25,1);
result.power_q75_mW=1e3*prctile(P,75,1);
result.support_stable_pct=100*sum(stable,1)./max(sum(valid,1),1);
result.recovery_ready_pct=100*sum(ready,1)./max(sum(valid,1),1);
result.eta_b_median=median(E,1,'omitnan');
result.completed_runs=sum([records.iterations]>=T);
result.median_runtime_s=median([records.runtime_s],'omitnan');
end

function write_summary_csv(result,file)
T=table(result.iteration(:),result.n_valid(:),result.binary_median(:), ...
    result.binary_q25(:),result.binary_q75(:),result.rank_median(:), ...
    result.support_stable_pct(:),result.recovery_ready_pct(:), ...
    result.power_median_mW(:),result.power_q25_mW(:),result.power_q75_mW(:), ...
    result.eta_b_median(:),'VariableNames',{'Iteration','ValidRuns', ...
    'MedianBinaryDistance','BinaryQ25','BinaryQ75','MedianRankResidual', ...
    'SupportStablePct','RecoveryReadyPct','MedianPower_mW','PowerQ25_mW', ...
    'PowerQ75_mW','MedianEtaB'});
writetable(T,file);
end

function plot_result(r,file)
t=r.iteration;
fig=figure('Visible','off','Position',[100 100 1180 800]);
layout=tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');

ax=nexttile(layout); hold(ax,'on'); grid(ax,'on');
fill(ax,[t fliplr(t)],[max(r.binary_q25,1e-8) fliplr(max(r.binary_q75,1e-8))], ...
    [0.78 0.88 0.97],'EdgeColor','none','FaceAlpha',0.8);
semilogy(ax,t,max(r.binary_median,1e-8),'o-','Color',[0 0.35 0.65], ...
    'LineWidth',1.8,'MarkerFaceColor',[0 0.35 0.65]);
yline(ax,0.05,'--','recovery distance threshold','Color',[0.75 0 0]);
xlabel(ax,'DC-SCA iteration'); ylabel(ax,'Maximum binary distance');
title(ax,'Binary relaxation approaches a stable recovery basin');
legend(ax,{'25th–75th percentile','median'},'Location','southwest');

ax=nexttile(layout); grid(ax,'on'); hold(ax,'on');
plot(ax,t,r.support_stable_pct,'s-','Color',[0.12 0.47 0.25], ...
    'LineWidth',1.8,'MarkerFaceColor',[0.12 0.47 0.25]);
plot(ax,t,r.recovery_ready_pct,'d-','Color',[0.85 0.35 0.05], ...
    'LineWidth',1.8,'MarkerFaceColor',[0.85 0.35 0.05]);
ylim(ax,[0 105]); xlabel(ax,'DC-SCA iteration'); ylabel(ax,'Valid-run fraction (%)');
title(ax,'Topology stabilization and recovery readiness');
legend(ax,{'Top-N support unchanged','rank-small + stable + recovery-ready'}, ...
    'Location','southeast');

ax=nexttile(layout); grid(ax,'on');
fill(ax,[t fliplr(t)],[r.power_q25_mW fliplr(r.power_q75_mW)], ...
    [0.9 0.9 0.9],'EdgeColor','none'); hold(ax,'on');
plot(ax,t,r.power_median_mW,'o-','Color',[0.2 0.2 0.2], ...
    'LineWidth',1.8,'MarkerFaceColor',[0.2 0.2 0.2]);
xlabel(ax,'DC-SCA iteration'); ylabel(ax,'Continuous power (mW)');
title(ax,'Continuation diagnostic; not a descent claim');

ax=nexttile(layout); yyaxis(ax,'left');
semilogy(ax,t,max(r.rank_median,1e-12),'s-','Color',[0.7 0.1 0.1], ...
    'LineWidth',1.8,'MarkerFaceColor',[0.7 0.1 0.1]);
ylabel(ax,'Median rank residual'); grid(ax,'on');
yyaxis(ax,'right'); stairs(ax,t,r.eta_b_median,'Color',[0.2 0.2 0.2], ...
    'LineWidth',1.5); ylabel(ax,'Median binary penalty');
xlabel(ax,'DC-SCA iteration'); title(ax,'Penalty continuation and rank residual');

sgtitle(fig,sprintf('Statistical double-DC SCA stabilization: %d common seeds, %d complete traces', ...
    r.n_total,r.completed_runs));
exportgraphics(fig,file,'Resolution',300); close(fig);
end

function v=get_field(s,name,default)
if isfield(s,name), v=s.(name); else, v=default; end
end
