function run_smoke_v1(output_dir)
%RUN_SMOKE_V1 Reproducible v1.0 paired Monte Carlo pipeline smoke test.
% Configure CVX and select MOSEK before calling this function.

if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'results', 'smoke');
end

experiments_paper('N_mc', 1, 'N_req_list', 3, 'T_max', 5, ...
    'Run_robustness', false, 'N_workers', 0, ...
    'Solver', 'mosek', 'Output_dir', output_dir, 'Output_tag', 'v1_smoke');
end
