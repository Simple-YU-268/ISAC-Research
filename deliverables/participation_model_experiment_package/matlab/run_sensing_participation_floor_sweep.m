function output = run_sensing_participation_floor_sweep(varargin)
%RUN_SENSING_PARTICIPATION_FLOOR_SWEEP  Sensitivity to Pmin^sen/Pmax.
%   Uses common seeds and the current dedicated-sensing model.  The zero
%   point is an authorization-only ablation; positive points enforce that a
%   selected AP-target pair radiates dedicated sensing energy.

ip = inputParser;
addParameter(ip, 'Seeds', 1:30, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'Floor_fraction_list', [0 .005 .01 .02 .05], ...
    @(x) isnumeric(x) && isvector(x) && all(x >= 0) && all(x <= 1));
addParameter(ip, 'T_max', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'Mosek_max_time', 10, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(ip, 'Output_dir', fullfile(pwd, 'experiment_packages', 'v1.0', ...
    'results', 'sensing_participation_floor_sweep'), @(x) ischar(x) || isstring(x));
addParameter(ip, 'Resume', true, @(x) islogical(x) && isscalar(x));
parse(ip, varargin{:}); opt = ip.Results;

seeds = opt.Seeds(:).'; floors = opt.Floor_fraction_list(:).';
out_dir = char(opt.Output_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
checkpoint_file = fullfile(out_dir, 'checkpoint.mat');
records = repmat(empty_record(), numel(floors), numel(seeds));
if opt.Resume && exist(checkpoint_file, 'file')
    saved = load(checkpoint_file, 'records', 'seeds_saved', 'floors_saved');
    if isequal(saved.seeds_saved, seeds) && isequal(saved.floors_saved, floors)
        records = saved.records;
    end
end

for q = 1:numel(floors)
    for i = 1:numel(seeds)
        if ~isnan(records(q,i).seed), continue; end
        fprintf('Participation floor %.3g Pmax, seed=%d (%d/%d)\n', ...
            floors(q), seeds(i), (q-1)*numel(seeds)+i, numel(floors)*numel(seeds));
        prm = generate_scenario(6,2,3,2,2,20,'auto','AreaSize',400, ...
            'N_req',3,'eps_h',.05,'seed',seeds(i), ...
            'sensing_min_power_fraction',floors(q));
        prm.solver = 'mosek'; prm.mosek_max_time = opt.Mosek_max_time;
        prm.recovery_mosek_max_time = opt.Mosek_max_time;
        prm.recovery_max_candidates = 3; prm.recovery_stop_first_feasible = false;
        timer = tic;
        res = baseline_alg2(prm,opt.T_max,1e-5,1,1,1,false);
        records(q,i).seed = seeds(i); records(q,i).floor_fraction = floors(q);
        records(q,i).time_s = toc(timer); records(q,i).status = string(res.status);
        records(q,i).feasible = isfield(res,'is_physical_feasible') && res.is_physical_feasible;
        if records(q,i).feasible
            met = evaluate_isac_metrics(prm,res.W,res.S_p,res.mu,res.b,res.M_p);
            records(q,i).total_power_W = met.total_power_W;
            records(q,i).max_pcrb_ratio = max(met.pcrb_ratio);
            records(q,i).min_selected_sensing_W = min_selected_power(prm,met.b,res.S_p);
            records(q,i).participation_violation_W = max(0, ...
                prm.sensing_min_power - records(q,i).min_selected_sensing_W);
        end
        seeds_saved = seeds; floors_saved = floors;
        save(checkpoint_file, 'records', 'seeds_saved', 'floors_saved', 'opt');
    end
end
output.records = records; output.seeds = seeds; output.floor_fraction_list = floors;
save(fullfile(out_dir,'participation_floor_final.mat'),'output','opt');
end

function value = min_selected_power(prm, b, S_p)
% Evaluate the same per-(AP,target) trace used by the participation LMI.
E = build_E_m(prm.M, prm.Nt);
power = zeros(prm.M, prm.P);
for p = prm.active_targets
    for m = 1:prm.M
        power(m,p) = real(trace(E{m} * S_p(:,:,p)));
    end
end
selected = b(:,prm.active_targets) > .5;
power = power(:,prm.active_targets);
if any(selected(:)), value = min(power(selected)); else, value = NaN; end
end

function r = empty_record()
r = struct('seed',NaN,'floor_fraction',NaN,'time_s',NaN,'status',"not_run", ...
    'feasible',false,'total_power_W',NaN,'max_pcrb_ratio',NaN, ...
    'min_selected_sensing_W',NaN,'participation_violation_W',NaN);
end
