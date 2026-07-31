function manifest = run_large_scale_algorithm_validation(varargin)
%RUN_LARGE_SCALE_ALGORITHM_VALIDATION Reproducible large-scale paper campaign.
%   The campaign covers N_req=2:6 on three increasing physical networks, a
%   representative double-DC convergence trace, and a QoS trade-off surface.

ip = inputParser;
addParameter(ip,'Seeds',1:50,@(x)isnumeric(x)&&isvector(x));
addParameter(ip,'Tradeoff_seeds',1:10,@(x)isnumeric(x)&&isvector(x));
addParameter(ip,'T_max',3,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(ip,'Mosek_max_time',15,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(ip,'N_workers',0,@(x)isnumeric(x)&&isscalar(x)&&x>=0&&fix(x)==x);
addParameter(ip,'Output_dir',fullfile(pwd,'experiment_packages','v1.0', ...
    'results','large_scale_algorithm_validation'),@(x)ischar(x)||isstring(x));
addParameter(ip,'Resume',true,@(x)islogical(x)&&isscalar(x));
parse(ip,varargin{:}); opt=ip.Results;

out_dir=char(opt.Output_dir); if ~exist(out_dir,'dir'), mkdir(out_dir); end
cfg = [struct('id','M9_K4_P3','M',9,'Nt',2,'K',4,'P',3); ...
       struct('id','M12_K6_P3','M',12,'Nt',2,'K',6,'P',3); ...
       struct('id','M16_K8_P4','M',16,'Nt',2,'K',8,'P',4)];
manifest=struct('configuration',cfg,'nreq_list',2:6,'seeds',opt.Seeds, ...
    'tradeoff_seeds',opt.Tradeoff_seeds,'t_max',opt.T_max, ...
    'mosek_max_time',opt.Mosek_max_time,'n_workers',opt.N_workers);
save(fullfile(out_dir,'campaign_manifest.mat'),'manifest','opt');

if opt.N_workers > 0
    pool=gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers ~= opt.N_workers
        delete(pool);
        pool=[];
    end
    if isempty(pool)
        pool=parpool('local',opt.N_workers);
    end
    fprintf('Running campaign with %d parallel workers.\n',pool.NumWorkers);
    parfor task_id=1:(numel(cfg)+1)
        if task_id <= numel(cfg)
            run_scale_task(cfg(task_id),opt,out_dir);
        else
            run_auxiliary_task(opt,out_dir);
        end
    end
else
    for q=1:numel(cfg)
        run_scale_task(cfg(q),opt,out_dir);
    end
    run_auxiliary_task(opt,out_dir);
end
end

function run_scale_task(scale,opt,out_dir)
run_nreq_method_performance_mc('Seeds',opt.Seeds,'N_req_list',2:6, ...
    'M',scale.M,'Nt',scale.Nt,'K',scale.K,'P',scale.P, ...
    'T_max',opt.T_max,'Mosek_max_time',opt.Mosek_max_time, ...
    'Output_dir',fullfile(out_dir,scale.id),'Resume',opt.Resume);
end

function run_auxiliary_task(opt,out_dir)
fig_dir=fullfile(out_dir,'figures');
run_double_dc_convergence('M',12,'Nt',2,'K',6,'P',3,'N_req',3, ...
    'T_max',10,'Solver','mosek','Output_dir',fig_dir,'Output_tag','M12_K6_P3');
run_isac_tradeoff_surface_mc('Seeds',opt.Tradeoff_seeds,'M',12,'Nt',2, ...
    'K',6,'P',3,'N_req',3,'Gamma_alpha_list',[1.5 2 3 4], ...
    'Gamma_k_dB_list',[-3 0 3 6],'T_max',opt.T_max, ...
    'Mosek_max_time',opt.Mosek_max_time, ...
    'Output_dir',fullfile(out_dir,'tradeoff'),'Resume',opt.Resume);
end
