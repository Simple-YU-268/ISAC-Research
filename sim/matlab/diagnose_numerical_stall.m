function report = diagnose_numerical_stall(varargin)
%DIAGNOSE_NUMERICAL_STALL  Diagnose a difficult SDP instance without claiming infeasibility.

ip = inputParser;
addParameter(ip, 'Seed', 23, @(x) isnumeric(x) && isscalar(x));
addParameter(ip, 'N_req', 2, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(ip, 'Output_dir', fullfile(pwd, '..', '..', 'experiment_packages', ...
    'v1.0', 'results', 'nreq_qos_sweep', 'diagnostics'), @(x) ischar(x) || isstring(x));
addParameter(ip, 'Run_solver_probe', false, @(x) islogical(x) && isscalar(x));
parse(ip, varargin{:}); opt = ip.Results;
out_dir = char(opt.Output_dir); if ~exist(out_dir,'dir'), mkdir(out_dir); end

prm = generate_scenario(6,2,3,2,2,20,'auto','AreaSize',400, ...
    'N_req',opt.N_req,'eps_h',0.05,'seed',opt.Seed);
report.seed = opt.Seed; report.N_req = opt.N_req;
report.Gamma_track = prm.Gamma_track;
report.D_gram_eig = cell(prm.P,1);
report.D_gram_cond = NaN(prm.P,1);
for p = 1:prm.P
    G = real(prm.D(:,:,p)'*prm.D(:,:,p));
    ev = sort(real(eig(G,'vector')),'ascend');
    report.D_gram_eig{p} = ev;
    report.D_gram_cond(p) = max(ev) / max(min(ev), eps);
end
report.greedy_b = construct_greedy_fim_assignment(prm,'doptimal');
save(fullfile(out_dir,sprintf('seed_%d_geometry.mat',opt.Seed)),'report','prm','opt');

if opt.Run_solver_probe
    diary(fullfile(out_dir,sprintf('seed_%d_solver_probe.log',opt.Seed)));
    cleanup = onCleanup(@() diary('off')); %#ok<NASGU>
    prm.solver = 'mosek'; prm.mosek_max_time = 10; prm.cvx_quiet = false;
    prm.recovery_mosek_max_time = 10; prm.recovery_max_candidates = 1;
    prm.recovery_stop_first_feasible = true;
    timer = tic;
    report.probe = baseline_alg2(prm,1,1e-5,1,1,1,false);
    report.probe_elapsed_s = toc(timer);
    save(fullfile(out_dir,sprintf('seed_%d_solver_probe.mat',opt.Seed)),'report','opt');
end
end
