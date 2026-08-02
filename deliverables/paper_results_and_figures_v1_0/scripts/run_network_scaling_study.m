function output = run_network_scaling_study(varargin)
%RUN_NETWORK_SCALING_STUDY  End-to-end scaling under the participation model.
%   M varies while each AP has two transmit antennas; K, P, Ntheta and
%   Nreq remain fixed so the measured change is network size, not QoS.

ip = inputParser;
addParameter(ip, 'Seeds', 1:10, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'M_list', [4 6 8], @(x) isnumeric(x) && isvector(x) && all(x >= 3));
addParameter(ip, 'T_max', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'Mosek_max_time', 15, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(ip, 'Output_dir', fullfile(pwd, 'experiment_packages', 'v1.0', ...
    'results', 'network_scaling_study'), @(x) ischar(x) || isstring(x));
addParameter(ip, 'Resume', true, @(x) islogical(x) && isscalar(x));
parse(ip, varargin{:}); opt = ip.Results;

seeds = opt.Seeds(:).'; M_list = opt.M_list(:).';
out_dir = char(opt.Output_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
checkpoint_file = fullfile(out_dir, 'checkpoint.mat');
records = repmat(empty_record(), numel(M_list), numel(seeds));
if opt.Resume && exist(checkpoint_file, 'file')
    saved = load(checkpoint_file, 'records', 'seeds_saved', 'M_saved');
    if isequal(saved.seeds_saved,seeds) && isequal(saved.M_saved,M_list)
        records = saved.records;
    end
end

for q = 1:numel(M_list)
    for i = 1:numel(seeds)
        if ~isnan(records(q,i).seed), continue; end
        M = M_list(q); seed = seeds(i);
        fprintf('Scaling M=%d, seed=%d (%d/%d)\n', M, seed, ...
            (q-1)*numel(seeds)+i, numel(M_list)*numel(seeds));
        prm = generate_scenario(M,2,3,2,2,20,'auto','AreaSize',400, ...
            'N_req',3,'eps_h',.05,'seed',seed);
        prm.solver = 'mosek'; prm.mosek_max_time = opt.Mosek_max_time;
        prm.recovery_mosek_max_time = opt.Mosek_max_time;
        prm.recovery_max_candidates = 3; prm.recovery_stop_first_feasible = false;
        timer = tic; res = baseline_alg2(prm,opt.T_max,1e-5,1,1,1,false);
        records(q,i).M = M; records(q,i).N = prm.N; records(q,i).seed = seed;
        records(q,i).time_s = toc(timer); records(q,i).status = string(res.status);
        records(q,i).feasible = isfield(res,'is_physical_feasible') && res.is_physical_feasible;
        if records(q,i).feasible, records(q,i).power_W = res.final_obj; end
        seeds_saved = seeds; M_saved = M_list;
        save(checkpoint_file, 'records', 'seeds_saved', 'M_saved', 'opt');
    end
end
output.records = records; output.seeds = seeds; output.M_list = M_list;
save(fullfile(out_dir,'network_scaling_final.mat'),'output','opt');
end

function r = empty_record()
r = struct('M',NaN,'N',NaN,'seed',NaN,'time_s',NaN,'status',"not_run", ...
    'feasible',false,'power_W',NaN);
end
