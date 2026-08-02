function output = run_nreq_method_performance_mc(varargin)
%RUN_NREQ_METHOD_PERFORMANCE_MC  Common-seed physical-method comparison.
%   Compares complete binary-DC recovery, FIM-greedy, nearest-AP, and random
%   fixed associations.  All physical methods use the same fixed-b continuous
%   beamforming/sensing-covariance optimizer after topology selection.  A
%   continuous SDR power lower bound is stored separately and is never used as
%   a physical QoS or topology result.

ip = inputParser;
addParameter(ip, 'Seeds', 1:30, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'N_req_list', 2:6, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'T_max', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'Mosek_max_time', 10, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(ip, 'Output_dir', fullfile(pwd, 'experiment_packages', 'v1.0', ...
    'results', 'nreq_method_performance_mc'), @(x) ischar(x) || isstring(x));
addParameter(ip, 'Resume', true, @(x) islogical(x) && isscalar(x));
parse(ip, varargin{:}); opt = ip.Results;

seeds = opt.Seeds(:).'; nreq_list = opt.N_req_list(:).';
labels = ["Proposed full recovery", "FIM-greedy topology", ...
    "Nearest-AP topology", "Random topology"];
out_dir = char(opt.Output_dir); if ~exist(out_dir,'dir'), mkdir(out_dir); end
checkpoint_file = fullfile(out_dir,'checkpoint.mat');
records = repmat(empty_record(), numel(nreq_list), numel(seeds));
if opt.Resume && exist(checkpoint_file,'file')
    saved = load(checkpoint_file,'records','seeds_saved','nreq_saved','labels_saved');
    if isequal(saved.seeds_saved,seeds) && isequal(saved.nreq_saved,nreq_list) && isequal(saved.labels_saved,labels)
        records = saved.records;
    end
end

for q = 1:numel(nreq_list)
    for i = 1:numel(seeds)
        if ~isnan(records(q,i).seed), continue; end
        nreq = nreq_list(q); seed = seeds(i);
        fprintf('Method MC Nreq=%d, seed=%d (%d/%d)\n', nreq, seed, ...
            (q-1)*numel(seeds)+i, numel(nreq_list)*numel(seeds));
        prm = generate_scenario(6,2,3,2,2,20,'auto','AreaSize',400, ...
            'N_req',nreq,'eps_h',.05,'seed',seed);
        prm.solver = 'mosek'; prm.mosek_max_time = opt.Mosek_max_time;
        prm.recovery_mosek_max_time = opt.Mosek_max_time;
        records(q,i).seed = seed; records(q,i).N_req = nreq;

        % The continuous lower bound is intentionally kept separate.
        records(q,i).sdr_lower_bound_W = solve_sdr_lower_bound(prm);

        prm_full = prm; prm_full.recovery_max_candidates = 3;
        prm_full.recovery_stop_first_feasible = false;
        records(q,i).methods(1) = solve_proposed(prm_full,opt.T_max,labels(1));

        [b_fim,~] = construct_greedy_fim_assignment(prm,'doptimal');
        records(q,i).methods(2) = solve_fixed(prm,b_fim,labels(2));
        records(q,i).methods(3) = solve_fixed(prm,nearest_assignment(prm),labels(3));
        records(q,i).methods(4) = solve_fixed(prm,random_assignment(prm,900000+nreq*1000+seed),labels(4));

        seeds_saved = seeds; nreq_saved = nreq_list; labels_saved = labels;
        save(checkpoint_file,'records','seeds_saved','nreq_saved','labels_saved','opt');
    end
end
output.records = records; output.seeds = seeds; output.nreq_list = nreq_list; output.labels = labels;
save(fullfile(out_dir,'nreq_method_performance_final.mat'),'output','opt');
end

function method = solve_proposed(prm, t_max, label)
timer = tic; res = baseline_alg2(prm,t_max,1e-5,1,1,1,false);
method = summarize(res,label,toc(timer),prm);
end

function method = solve_fixed(prm, b, label)
timer = tic; res = solve_p3_with_fixed_b(prm,b,2,1e-5,1,0,1);
method = summarize(res,label,toc(timer),prm);
end

function method = summarize(res, label, elapsed, prm)
method = struct('label',label,'feasible',false,'status',string(get_field(res,'status','unknown')), ...
    'time_s',elapsed,'power_W',NaN,'metrics',[]);
method.feasible = isfield(res,'is_physical_feasible') && res.is_physical_feasible;
method.power_W = get_field(res,'final_obj',NaN);
if method.feasible
    method.metrics = evaluate_isac_metrics(prm,res.W,res.S_p,res.mu,res.b,res.M_p);
end
end

function power = solve_sdr_lower_bound(prm)
W0 = cell(prm.K,1);
for k=1:prm.K, W0{k}=eye(prm.N)*prm.Pmax/(prm.K*prm.M); end
b0 = ones(prm.M,prm.P)*prm.N_req/prm.M;
[W,~,~,~,~,status,S] = solve_p3_sca_t(prm,W0,b0,0,0);
power = NaN;
if contains(status,'Solved')
    power = sum(cellfun(@(X)real(trace(X)),W)) + real(trace(sum(S,3)));
end
end

function b = nearest_assignment(prm)
b=zeros(prm.M,prm.P);
for p=prm.active_targets
    d=sqrt(sum((prm.AP_pos-prm.Target_pos(p,:)).^2,2)); [~,order]=sort(d);
    b(order(1:prm.N_req),p)=1;
end
end

function b = random_assignment(prm, seed)
state=rng; cleanup=onCleanup(@()rng(state)); %#ok<NASGU>
rng(seed,'twister'); b=zeros(prm.M,prm.P);
for p=prm.active_targets, b(randperm(prm.M,prm.N_req),p)=1; end
end

function value = get_field(s,name,default_value)
if isfield(s,name), value=s.(name); else, value=default_value; end
end

function r = empty_record()
method=struct('label',"",'feasible',false,'status',"not_run",'time_s',NaN,'power_W',NaN,'metrics',[]);
r=struct('seed',NaN,'N_req',NaN,'sdr_lower_bound_W',NaN,'methods',repmat(method,1,4));
end
