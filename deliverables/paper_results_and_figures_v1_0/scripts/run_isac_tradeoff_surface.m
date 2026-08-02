function result = run_isac_tradeoff_surface(varargin)
%RUN_ISAC_TRADEOFF_SURFACE  Communication--sensing power trade-off sweep.
%   Gamma_alpha scales the auto-calibrated PCRB allowance: a smaller value
%   means stricter tracking. gamma_k_dB is the robust communication target.

ip = inputParser;
addParameter(ip, 'Seed', 1, @(x) isnumeric(x) && isscalar(x));
addParameter(ip, 'Gamma_alpha_list', linspace(1.5, 6, 7), @(x) isnumeric(x) && isvector(x) && all(x > 0));
addParameter(ip, 'Gamma_k_dB_list', -3:2:9, @(x) isnumeric(x) && isvector(x));
addParameter(ip, 'T_max', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'Mosek_max_time', 30, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(ip, 'Resume', true, @(x) islogical(x) && isscalar(x));
addParameter(ip, 'Output_dir', fullfile(pwd, '..', '..', 'experiment_packages', ...
    'v1.0', 'results', 'isac_tradeoff_surface'), @(x) ischar(x) || isstring(x));
parse(ip, varargin{:}); opt = ip.Results;
out_dir = char(opt.Output_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
diary(fullfile(out_dir, 'surface.log')); cleanup_diary = onCleanup(@() diary('off'));

alpha = opt.Gamma_alpha_list(:).';
gamma_db = opt.Gamma_k_dB_list(:).';
power_W = NaN(numel(alpha), numel(gamma_db));
runtime_s = NaN(size(power_W));
feasible = false(size(power_W));
topology = cell(size(power_W));
checkpoint_file = fullfile(out_dir, 'surface_checkpoint.mat');
if opt.Resume && exist(checkpoint_file, 'file')
    saved = load(checkpoint_file, 'checkpoint');
    if isequal(saved.checkpoint.alpha, alpha) && isequal(saved.checkpoint.gamma_db, gamma_db)
        power_W = saved.checkpoint.power_W;
        runtime_s = saved.checkpoint.runtime_s;
        feasible = saved.checkpoint.feasible;
        topology = saved.checkpoint.topology;
        fprintf('Resuming %d completed grid points from checkpoint.\n', nnz(~isnan(runtime_s)));
    else
        warning('Checkpoint grid differs from requested sweep; starting a new sweep.');
    end
end
for a = 1:numel(alpha)
    for g = 1:numel(gamma_db)
        if ~isnan(runtime_s(a,g))
            continue;
        end
        fprintf('Surface alpha=%.3g, gamma=%.1f dB (%d/%d)\n', alpha(a), gamma_db(g), ...
            (a-1)*numel(gamma_db)+g, numel(alpha)*numel(gamma_db));
        prm = generate_scenario(6, 2, 3, 2, 2, 20, 'auto', ...
            'AreaSize', 400, 'N_req', 3, 'eps_h', 0.05, 'seed', opt.Seed, ...
            'Gamma_alpha', alpha(a), 'gamma_k_dB', gamma_db(g));
        prm.solver = 'mosek';
        prm.mosek_max_time = opt.Mosek_max_time;
        prm.recovery_max_candidates = 3;
        prm.recovery_stop_first_feasible = false;
        prm.recovery_mosek_max_time = 30;
        prm.recovery_slack_diagnosis = true;
        prm.recovery_slack_guided_slots = 1;
        timer = tic;
        res = baseline_alg2(prm, opt.T_max, 1e-5, 1, 1, 1.0, false);
        runtime_s(a,g) = toc(timer);
        feasible(a,g) = isfield(res, 'is_physical_feasible') && res.is_physical_feasible;
        if feasible(a,g)
            power_W(a,g) = res.final_obj;
            topology{a,g} = extract_power_map(prm, res);
        end
        checkpoint.alpha = alpha; checkpoint.gamma_db = gamma_db;
        checkpoint.power_W = power_W; checkpoint.runtime_s = runtime_s;
        checkpoint.feasible = feasible; checkpoint.topology = topology;
        save(checkpoint_file, 'checkpoint', 'opt');
    end
end
result.alpha = alpha; result.gamma_db = gamma_db; result.power_W = power_W;
result.runtime_s = runtime_s; result.feasible = feasible; result.topology = topology;
result.seed = opt.Seed;
save(fullfile(out_dir, 'surface_final.mat'), 'result', 'opt');
plot_surface(result, out_dir);
end

function map = extract_power_map(prm, res)
E = build_E_m(prm.M, prm.Nt);
map.AP_pos = prm.AP_pos; map.UE_pos = prm.UE_pos; map.Target_pos = prm.Target_pos;
map.b = res.b;
map.communication_power_W = zeros(prm.M,1);
map.sensing_power_W = zeros(prm.M,prm.P);
for m = 1:prm.M
    for k = 1:prm.K
        map.communication_power_W(m) = map.communication_power_W(m) + real(trace(E{m}*res.W{k}));
    end
    for p = 1:prm.P
        map.sensing_power_W(m,p) = real(trace(E{m}*res.S_p(:,:,p)));
    end
end
end

function plot_surface(result, out_dir)
[G,A] = meshgrid(result.gamma_db, result.alpha);
fig = figure('Visible','off','Position',[100 100 700 500]);
surf(G, A, result.power_W, 'EdgeColor',[0.3 0.3 0.3]);
xlabel('Robust communication SINR target (dB)');
ylabel('PCRB allowance scale \alpha'); zlabel('Total transmit power (W)');
title('ISAC communication--sensing trade-off surface'); colorbar; grid on; view(45,30);
exportgraphics(fig, fullfile(out_dir, 'fig8_isac_tradeoff_surface.png'),'Resolution',300); close(fig);
end
