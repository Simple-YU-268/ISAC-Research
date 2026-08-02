function output = run_csi_robustness_experiment(varargin)
%RUN_CSI_ROBUSTNESS_EXPERIMENT  Robust versus nominal CSI design comparison.
%   The nominal baseline removes the S-procedure rather than retaining a
%   degenerate epsilon_h=0 LMI.  Both designs are evaluated on identical
%   independently sampled complex uncertainty-ball channels.

ip = inputParser;
addParameter(ip,'Seeds',1:30,@(x)isnumeric(x)&&isvector(x));
addParameter(ip,'Eps_list',[0 .02 .05 .08],@(x)isnumeric(x)&&isvector(x)&&all(x>=0));
addParameter(ip,'Samples_per_seed',200,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(ip,'T_max',3,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(ip,'Mosek_max_time',10,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(ip,'Output_dir',fullfile(pwd,'..','..','experiment_packages','v1.0', ...
    'results','csi_robustness'),@(x)ischar(x)||isstring(x));
addParameter(ip,'Resume',true,@(x)islogical(x)&&isscalar(x));
parse(ip,varargin{:}); opt=ip.Results;
seeds=opt.Seeds(:).'; eps_list=opt.Eps_list(:).'; out_dir=char(opt.Output_dir);
if ~exist(out_dir,'dir'), mkdir(out_dir); end
checkpoint_file=fullfile(out_dir,'checkpoint.mat');
records=repmat(empty_record(),numel(eps_list),numel(seeds));
if opt.Resume && exist(checkpoint_file,'file')
    saved=load(checkpoint_file,'records','seeds_saved','eps_saved');
    if isequal(saved.seeds_saved,seeds) && isequal(saved.eps_saved,eps_list), records=saved.records; end
end

for q=1:numel(eps_list)
    for i=1:numel(seeds)
        if ~isnan(records(q,i).seed), continue; end
        eps_test=eps_list(q); seed=seeds(i);
        fprintf('CSI robustness eps=%.3f, seed=%d (%d/%d)\n',eps_test,seed, ...
            (q-1)*numel(seeds)+i,numel(eps_list)*numel(seeds));
        prm=generate_scenario(6,2,3,2,2,20,'auto','AreaSize',400,'N_req',3, ...
            'eps_h',eps_test,'seed',seed);
        prm_rob=prm;
        if eps_test == 0, prm_rob.use_s_procedure=false; end
        [robust, t_robust]=solve_design(prm_rob,opt);
        prm_nom=prm; prm_nom.eps_h=0; prm_nom.use_s_procedure=false;
        [nominal, t_nominal]=solve_design(prm_nom,opt);
        records(q,i).seed=seed; records(q,i).eps_test=eps_test;
        records(q,i).robust=assess_design(prm,prm_rob,robust,eps_test,opt.Samples_per_seed,seed);
        records(q,i).robust.time_s=t_robust;
        records(q,i).nominal=assess_design(prm,prm_nom,nominal,eps_test,opt.Samples_per_seed,seed);
        records(q,i).nominal.time_s=t_nominal;
        seeds_saved=seeds; eps_saved=eps_list;
        save(checkpoint_file,'records','seeds_saved','eps_saved','opt');
    end
end
output.records=records; output.seeds=seeds; output.eps_list=eps_list;
save(fullfile(out_dir,'csi_robustness_final.mat'),'output','opt');
end

function [res, elapsed]=solve_design(prm,opt)
prm.solver='mosek'; prm.mosek_max_time=opt.Mosek_max_time;
prm.recovery_mosek_max_time=opt.Mosek_max_time;
prm.recovery_max_candidates=3; prm.recovery_stop_first_feasible=false;
timer=tic; res=baseline_alg2(prm,opt.T_max,1e-5,1,1,1,false); elapsed=toc(timer);
end

function evaluation=assess_design(test_prm,design_prm,res,eps_test,num_samples,seed)
evaluation=struct('feasible',false,'power_W',NaN,'outage_probability',NaN, ...
    'worst_sample_margin_dB',NaN,'robust_lmi_min_eig',NaN);
if ~isfield(res,'is_physical_feasible') || ~res.is_physical_feasible, return; end
evaluation.feasible=true; evaluation.power_W=res.final_obj;
nominal_metrics=evaluate_isac_metrics(design_prm,res.W,res.S_p,res.mu,res.b,res.M_p);
evaluation.robust_lmi_min_eig=min(nominal_metrics.robust_lmi_min_eig);

N=test_prm.N; K=test_prm.K; S_total=sum(res.S_p,3);
rng(900000 + 1000*seed + round(10000*eps_test),'twister');
outage=false(num_samples,1); worst_margin=inf;
for r=1:num_samples
    user_margin=inf(K,1);
    for k=1:K
        delta=randn(N,1)+1i*randn(N,1); delta=delta/max(norm(delta),eps);
        radius=eps_test*norm(test_prm.H(:,k))*rand^(1/(2*N));
        hk=test_prm.H(:,k)+radius*delta;
        desired=real(hk'*res.W{k}*hk);
        interference=test_prm.sigma_c2+real(hk'*S_total*hk);
        for j=setdiff(1:K,k), interference=interference+real(hk'*res.W{j}*hk); end
        user_margin(k)=10*log10(max(desired/max(interference,eps),eps)/test_prm.gamma_k(k));
    end
    worst_margin=min(worst_margin,min(user_margin));
    outage(r)=any(user_margin<0);
end
evaluation.outage_probability=mean(outage);
evaluation.worst_sample_margin_dB=worst_margin;
end

function r=empty_record()
r=struct('seed',NaN,'eps_test',NaN,'robust',[],'nominal',[]);
end
