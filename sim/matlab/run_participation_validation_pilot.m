function summary = run_participation_validation_pilot(seeds, output_dir)
%RUN_PARTICIPATION_VALIDATION_PILOT Validate the participation-constrained model.
%   This wrapper keeps results separate from legacy authorization-only runs.

if nargin < 1 || isempty(seeds), seeds = 1:10; end
if nargin < 2 || isempty(output_dir)
    output_dir = fullfile(pwd, 'experiment_packages', 'v1.0', 'results', ...
        'participation_model_pilot_10seeds');
end

summary = run_feasibility_gap_pilot('Seeds', seeds, 'T_max', 5, ...
    'M', 6, 'Nt', 2, 'K', 3, 'P', 2, 'N_req', 3, ...
    'Output_dir', output_dir);
end
