function campaign = run_extended_physical_mc(varargin)
%RUN_EXTENDED_PHYSICAL_MC  Parallel Monte-Carlo study beyond the main setting.
%   Sweeps AP count, antennas/AP, UE and target load, deployment scale, and
%   per-AP power.  Two controlled stress geometries are included.  Every trial
%   compares the certified proposed recovery with FIM-greedy and oracle
%   nearest-AP fixed topologies under the identical realization.
%
%   The campaign is deliberately factorized: only one physical factor changes
%   within a family.  This preserves causal interpretability of each curve.
%
% Example
%   run_extended_physical_mc('Seeds',1:30,'N_workers',6);

ip = inputParser;
addParameter(ip,'Seeds',1:30,@(x)isnumeric(x)&&isvector(x));
addParameter(ip,'N_workers',6,@(x)isnumeric(x)&&isscalar(x)&&x>=0);
addParameter(ip,'T_max',3,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(ip,'Mosek_max_time',15,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(ip,'Output_dir',fullfile(pwd,'experiment_packages','v1.0', ...
    'results','extended_physical_mc'),@(x)ischar(x)||isstring(x));
addParameter(ip,'Resume',true,@(x)islogical(x)&&isscalar(x));
addParameter(ip,'Configuration_ids',strings(0,1),@(x)ischar(x)||isstring(x)||iscellstr(x));
addParameter(ip,'Methods',["Proposed","FIM-greedy","Nearest-AP"], ...
    @(x)isstring(x)||iscellstr(x));
parse(ip,varargin{:}); opt=ip.Results;

seeds=opt.Seeds(:).'; methods=string(opt.Methods(:).');
assert(isequal(methods,["Proposed","FIM-greedy","Nearest-AP"]), ...
    'This campaign requires the three documented physical methods.');
out_dir=char(opt.Output_dir); if ~exist(out_dir,'dir'), mkdir(out_dir); end
cfgs=extended_configurations();
if ~isempty(opt.Configuration_ids)
    wanted=string(opt.Configuration_ids);
    keep=ismember(string({cfgs.id}),wanted);
    assert(any(keep),'No requested Configuration_ids are defined.');
    cfgs=cfgs(keep);
end
write_progress(out_dir,sprintf('CAMPAIGN START: %d configurations, %d seeds, workers=%d', ...
    numel(cfgs),numel(seeds),opt.N_workers));

pool=[];
if opt.N_workers>0
    pool=gcp('nocreate');
    if isempty(pool)
        pool=parpool('local',opt.N_workers);
    elseif pool.NumWorkers~=opt.N_workers
        warning('Using existing pool with %d workers (requested %d).',pool.NumWorkers,opt.N_workers);
    end
    % Local workers inherit the client MATLAB path when the pool is created.
end

campaign=struct(); campaign.configurations=cfgs; campaign.seeds=seeds;
campaign.methods=methods; campaign.records=cell(numel(cfgs),1); campaign.opt=opt;
for c=1:numel(cfgs)
    cfg=cfgs(c); file=fullfile(out_dir,[cfg.id,'.mat']); records=[];
    if opt.Resume && exist(file,'file')
        saved=load(file,'records','cfg_saved','seeds_saved','methods_saved');
        if isequal(saved.cfg_saved,cfg) && isequal(saved.seeds_saved,seeds) && ...
                isequal(string(saved.methods_saved),methods)
            records=saved.records;
        end
    end
    if isempty(records)
        write_progress(out_dir,sprintf('START %s (%d/%d)',cfg.id,c,numel(cfgs)));
        records=repmat(empty_record(methods),1,numel(seeds));
        if opt.N_workers>0
            q=parallel.pool.DataQueue;
            afterEach(q,@(msg)write_progress(out_dir,char(msg)));
            parfor s=1:numel(seeds)
                one=run_one(cfg,seeds(s),methods,opt);
                records(s)=one;
                send(q,sprintf('DONE %s seed=%d feasible=%s',cfg.id,seeds(s), ...
                    mat2str([one.methods.feasible])));
            end
        else
            for s=1:numel(seeds)
                records(s)=run_one(cfg,seeds(s),methods,opt);
                write_progress(out_dir,sprintf('DONE %s seed=%d feasible=%s',cfg.id,seeds(s), ...
                    mat2str([records(s).methods.feasible])));
            end
        end
        cfg_saved=cfg; seeds_saved=seeds; methods_saved=methods; %#ok<NASGU>
        save(file,'records','cfg_saved','seeds_saved','methods_saved','opt','-v7.3');
    else
        write_progress(out_dir,sprintf('RESUME %s: loaded completed configuration',cfg.id));
    end
    campaign.records{c}=records;
end
campaign.summary=summarize_campaign(campaign);
save(fullfile(out_dir,'extended_physical_mc_final.mat'),'campaign','-v7.3');
writetable(struct2table(campaign.summary),fullfile(out_dir,'extended_physical_mc_summary.csv'));
write_progress(out_dir,'CAMPAIGN COMPLETE');
end

function cfgs=extended_configurations()
% Baseline is repeated only as the reference point of each factorized family.
cfgs=struct('id',{},'family',{},'level',{},'M',{},'Nt',{},'K',{},'P',{}, ...
    'N_req',{},'AreaSize',{},'Pmax_dBm',{},'geometry',{});
base=struct('M',6,'Nt',2,'K',3,'P',2,'N_req',3,'AreaSize',400, ...
    'Pmax_dBm',20,'geometry',"random");
cfgs=append_family(cfgs,'ap_count','M',[4 6 8 10],base);
cfgs=append_family(cfgs,'antennas_per_ap','Nt',[1 2 4],base);
cfgs=append_family(cfgs,'ue_load','K',[2 3 4 5],base);
cfgs=append_family(cfgs,'target_load','P',[1 2 3],base);
cfgs=append_family(cfgs,'area_side_m','AreaSize',[200 400 600],base);
cfgs=append_family(cfgs,'power_budget_dBm','Pmax_dBm',[17 20 23],base);
for g=["edge_colocated","crowded_targets"]
    x=base; x.geometry=g; x.id=sprintf('stress_%s',g); x.family="stress";
    x.level=g; cfgs(end+1)=x; %#ok<AGROW>
end
end

function cfgs=append_family(cfgs,family,field,levels,base)
for value=levels
    x=base; x.(field)=value;
    if x.N_req>x.M, x.N_req=x.M; end
    x.family=string(family); x.level=value;
    x.id=sprintf('%s_%s%d',family,field,value);
    cfgs(end+1)=x; %#ok<AGROW>
end
end

function rec=run_one(cfg,seed,methods,opt)
rec=empty_record(methods); rec.seed=seed; rec.config_id=cfg.id;
try
    [AP_pos,UE_pos,Target_pos]=geometry_for_config(cfg,seed);
    prm=generate_scenario(cfg.M,cfg.Nt,cfg.K,cfg.P,2,cfg.Pmax_dBm,'auto', ...
        'AreaSize',cfg.AreaSize,'N_req',cfg.N_req,'eps_h',.05,'seed',seed, ...
        'AP_pos',AP_pos,'UE_pos',UE_pos,'Target_pos',Target_pos);
    prm.solver='mosek'; prm.mosek_max_time=opt.Mosek_max_time;
    prm.recovery_mosek_max_time=opt.Mosek_max_time;
    prm.recovery_max_candidates=3; prm.recovery_stop_first_feasible=false;
    rec.gamma_track=prm.Gamma_track;
    for j=1:numel(methods)
        timer=tic;
        switch methods(j)
            case "Proposed"
                res=baseline_alg2(prm,opt.T_max,1e-5,1,1,1,false);
            case "FIM-greedy"
                b=construct_greedy_fim_assignment(prm,'doptimal');
                res=solve_p3_with_fixed_b(prm,b,2,1e-5,1,0,1);
            case "Nearest-AP"
                res=solve_p3_with_fixed_b(prm,nearest_assignment(prm),2,1e-5,1,0,1);
        end
        rec.methods(j)=summarize_solution(res,prm,methods(j),toc(timer));
    end
catch ME
    rec.error_id=string(ME.identifier); rec.error_message=string(ME.message);
end
end

function method=summarize_solution(res,prm,label,elapsed)
method=struct('label',label,'feasible',false,'status',string(getfield_safe(res,'status','unknown')), ...
    'time_s',elapsed,'power_W',NaN,'sum_rate_bpsHz',NaN,'mean_pcrb_ratio',NaN, ...
    'mean_sensing_sinr_dB',NaN,'min_comm_margin_dB',NaN, ...
    'min_sensing_margin_dB',NaN,'nonzero_pairs',NaN,'nonzero_aps',NaN);
if isfield(res,'is_physical_feasible') && res.is_physical_feasible
    metrics=evaluate_isac_metrics(prm,res.W,res.S_p,res.mu,res.b,res.M_p);
    method.feasible=true; method.power_W=res.final_obj;
    method.sum_rate_bpsHz=metrics.sum_rate_bpsHz;
    method.mean_pcrb_ratio=metrics.mean_pcrb_ratio;
    method.mean_sensing_sinr_dB=metrics.mean_sensing_sinr_dB;
    method.min_comm_margin_dB=min(metrics.nominal_sinr_margin_dB);
    method.min_sensing_margin_dB=min(metrics.sensing_sinr_margin_dB);
    method.nonzero_pairs=metrics.num_nonzero_sensing_pairs;
    method.nonzero_aps=metrics.num_nonzero_sensing_aps;
end
end

function b=nearest_assignment(prm)
b=zeros(prm.M,prm.P);
for p=1:prm.P
    d=sqrt(sum((prm.AP_pos-prm.Target_pos(p,:)).^2,2)); [~,order]=sort(d);
    b(order(1:prm.N_req),p)=1;
end
end

function [AP,UE,T]=geometry_for_config(cfg,seed)
if cfg.geometry=="random", AP=[]; UE=[]; T=[]; return; end
state=rng; cleanup=onCleanup(@()rng(state)); %#ok<NASGU>
rng(800000+seed,'twister'); L=cfg.AreaSize;
AP=L*[.10 .12; .50 .10; .90 .12; .10 .88; .50 .90; .90 .88];
AP=AP+0.015*L*randn(size(AP)); AP=max(0,min(L,AP));
if cfg.geometry=="edge_colocated"
    UE=L*[.86 .50; .22 .22; .55 .78];
    T=L*[.84 .52; .18 .75];
elseif cfg.geometry=="crowded_targets"
    UE=L*[.48 .25; .25 .72; .78 .72];
    T=L*[.48 .53; .56 .53];
else
    error('Unknown stress geometry.');
end
UE=UE(1:cfg.K,:)+0.012*L*randn(cfg.K,2);
T=T(1:cfg.P,:)+0.008*L*randn(cfg.P,2);
UE=max(0,min(L,UE)); T=max(0,min(L,T));
end

function summary=summarize_campaign(campaign)
cfgs=campaign.configurations; methods=campaign.methods;
summary=repmat(struct('config_id',"",'family',"",'level',"",'method',"", ...
    'total',0,'feasible',0,'feasibility_rate',NaN,'mean_power_mW',NaN, ...
    'median_time_s',NaN,'mean_sum_rate_bpsHz',NaN,'mean_pcrb_ratio',NaN, ...
    'mean_sensing_sinr_dB',NaN),numel(cfgs)*numel(methods),1);
n=0;
for c=1:numel(cfgs)
    for j=1:numel(methods)
        n=n+1; x=[campaign.records{c}.methods]; x=x(j:numel(methods):end);
        f=[x.feasible]; summary(n).config_id=cfgs(c).id; summary(n).family=cfgs(c).family;
        summary(n).level=string(cfgs(c).level); summary(n).method=methods(j);
        summary(n).total=numel(x); summary(n).feasible=nnz(f); summary(n).feasibility_rate=mean(f);
        if any(f)
            summary(n).mean_power_mW=1e3*mean([x(f).power_W]);
            summary(n).median_time_s=median([x(f).time_s]);
            summary(n).mean_sum_rate_bpsHz=mean([x(f).sum_rate_bpsHz]);
            summary(n).mean_pcrb_ratio=mean([x(f).mean_pcrb_ratio]);
            summary(n).mean_sensing_sinr_dB=mean([x(f).mean_sensing_sinr_dB]);
        end
    end
end
end

function rec=empty_record(methods)
method=struct('label',"",'feasible',false,'status',"not_run",'time_s',NaN, ...
    'power_W',NaN,'sum_rate_bpsHz',NaN,'mean_pcrb_ratio',NaN, ...
    'mean_sensing_sinr_dB',NaN,'min_comm_margin_dB',NaN, ...
    'min_sensing_margin_dB',NaN,'nonzero_pairs',NaN,'nonzero_aps',NaN);
rec=struct('seed',NaN,'config_id',"",'gamma_track',NaN,'methods', ...
    repmat(method,1,numel(methods)),'error_id',"",'error_message',"");
end

function value=getfield_safe(s,name,default)
if isfield(s,name), value=s.(name); else, value=default; end
end

function write_progress(out_dir,message)
line=sprintf('[%s] %s\n',datestr(now,'yyyy-mm-dd HH:MM:SS'),message);
fprintf('%s',line); fid=fopen(fullfile(out_dir,'progress.log'),'a');
if fid>=0, fprintf(fid,'%s',line); fclose(fid); end
end
