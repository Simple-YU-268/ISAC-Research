function summary = tune_double_dc_penalty(varargin)
%TUNE_DOUBLE_DC_PENALTY  Compare binary-penalty schedules on one fixed case.
%   This is a diagnostic only; it does not modify Monte Carlo defaults.

p = inputParser;
addParameter(p, 'T_max', 15, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Seed', 2026, @(x) isnumeric(x) && isscalar(x));
addParameter(p, 'N_req', 3, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Gamma_alpha', 4, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Solver', 'mosek', @(x) ischar(x) || isstring(x));
addParameter(p, 'Output_dir', fullfile(pwd, 'figures'), @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opt = p.Results;

% Moderate acceleration is tested before an intentionally stronger schedule.
schedules = struct('label', {'baseline','moderate','strong'}, ...
    'eta_b0', {1, 3, 5}, 'eta_b_growth', {1.3, 1.5, 1.6});
n = numel(schedules);
summary = table(strings(n,1), zeros(n,1), zeros(n,1), false(n,1), ...
    strings(n,1), zeros(n,1), zeros(n,1), ...
    'VariableNames', {'schedule','eta_b0','eta_b_growth','binary_converged', ...
    'status','iterations_to_binary','final_power_mW'});

for q = 1:n
    sch = schedules(q);
    tag = sprintf('penalty_%s', sch.label);
    result = run_double_dc_convergence('T_max', opt.T_max, 'Seed', opt.Seed, ...
        'N_req', opt.N_req, 'Gamma_alpha', opt.Gamma_alpha, ...
        'Solver', opt.Solver, 'Eta_b0', sch.eta_b0, ...
        'Eta_b_growth', sch.eta_b_growth, 'Output_tag', tag, ...
        'Output_dir', opt.Output_dir);
    valid = find(isfinite(result.power));
    hit = find(result.binary_distance(valid) <= 1e-5, 1, 'first');
    summary.schedule(q) = string(sch.label);
    summary.eta_b0(q) = sch.eta_b0;
    summary.eta_b_growth(q) = sch.eta_b_growth;
    summary.binary_converged(q) = ~isempty(hit);
    summary.status(q) = result.status(valid(end));
    if isempty(hit)
        summary.iterations_to_binary(q) = NaN;
    else
        summary.iterations_to_binary(q) = valid(hit);
    end
    summary.final_power_mW(q) = 1e3 * result.power(valid(end));
end

if ~exist(opt.Output_dir, 'dir'), mkdir(opt.Output_dir); end
writetable(summary, fullfile(opt.Output_dir, 'double_dc_penalty_tuning_summary.csv'));
disp(summary);
end
