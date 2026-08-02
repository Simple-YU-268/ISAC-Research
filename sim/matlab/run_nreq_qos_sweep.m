function output = run_nreq_qos_sweep(varargin)
%RUN_NREQ_QOS_SWEEP  Certified QoS statistics versus sensing-cluster size.

ip = inputParser;
addParameter(ip, 'Seeds', 1:50, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'N_req_list', 2:6, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'T_max', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'Mosek_max_time', 10, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(ip, 'Output_dir', fullfile(pwd, '..', '..', 'experiment_packages', ...
    'v1.0', 'results', 'nreq_qos_sweep'), @(x) ischar(x) || isstring(x));
addParameter(ip, 'Resume', true, @(x) islogical(x) && isscalar(x));
parse(ip, varargin{:}); opt = ip.Results;
seeds = opt.Seeds(:).'; nreq_list = opt.N_req_list(:).'; out_dir = char(opt.Output_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
checkpoint_file = fullfile(out_dir, 'checkpoint.mat');
blank = empty_record();
records = repmat(blank, numel(nreq_list), numel(seeds));
if opt.Resume && exist(checkpoint_file, 'file')
    saved = load(checkpoint_file, 'records', 'seeds_saved', 'nreq_saved');
    if isequal(saved.seeds_saved,seeds) && isequal(saved.nreq_saved,nreq_list)
        records = saved.records;
    end
end

for q = 1:numel(nreq_list)
    for i = 1:numel(seeds)
        if ~isnan(records(q,i).seed), continue; end
        fprintf('QoS sweep Nreq=%d, seed=%d (%d/%d)\n', nreq_list(q), seeds(i), ...
            (q-1)*numel(seeds)+i, numel(nreq_list)*numel(seeds));
        prm = generate_scenario(6,2,3,2,2,20,'auto','AreaSize',400, ...
            'N_req',nreq_list(q),'eps_h',0.05,'seed',seeds(i));
        prm.solver = 'mosek'; prm.mosek_max_time = opt.Mosek_max_time;
        prm.recovery_mosek_max_time = opt.Mosek_max_time;
        prm.recovery_max_candidates = 3; prm.recovery_stop_first_feasible = false;
        timer = tic; res = baseline_alg2(prm,opt.T_max,1e-5,1,1,1,false);
        records(q,i).seed = seeds(i); records(q,i).N_req = nreq_list(q);
        records(q,i).time_s = toc(timer); records(q,i).status = string(res.status);
        records(q,i).feasible = isfield(res,'is_physical_feasible') && res.is_physical_feasible;
        if records(q,i).feasible
            records(q,i).metrics = evaluate_isac_metrics(prm,res.W,res.S_p,res.mu,res.b,res.M_p);
        end
        seeds_saved = seeds; nreq_saved = nreq_list;
        save(checkpoint_file,'records','seeds_saved','nreq_saved','opt');
    end
end
output.records = records; output.seeds = seeds; output.nreq_list = nreq_list;
save(fullfile(out_dir,'nreq_qos_final.mat'),'output','opt');
end

function r = empty_record()
r = struct('seed',NaN,'N_req',NaN,'time_s',NaN,'status',"not_run", ...
    'feasible',false,'metrics',[]);
end
