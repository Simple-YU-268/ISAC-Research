function output_file = run_mc_shard(machine_label, seed_first, seed_last, varargin)
%RUN_MC_SHARD Run one reproducible Monte Carlo seed shard on a single host.
%   output_file = RUN_MC_SHARD('macmini', 2052, 2076) runs the main N_req
%   study for the inclusive seed interval 2052:2076 and writes one MAT file
%   containing the raw per-seed records. Different hosts must use disjoint
%   intervals. Run the robustness study on one designated host only.

p = inputParser;
addParameter(p, 'N_req_list', 1:6, @(x) isnumeric(x) && isvector(x));
addParameter(p, 'Run_robustness', false, @islogical);
addParameter(p, 'N_workers', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'Output_root', fullfile(pwd, 'mc_shards'), ...
    @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opt = p.Results;

validateattributes(seed_first, {'numeric'}, {'scalar','integer','positive'});
validateattributes(seed_last, {'numeric'}, {'scalar','integer','>=',seed_first});
assert(ischar(machine_label) || isstring(machine_label), ...
    'machine_label must be text.');

n_mc = seed_last - seed_first + 1;
label = char(machine_label);
output_dir = fullfile(char(opt.Output_root), sprintf('%s_seed%d_%d', ...
    label, seed_first, seed_last));
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% experiments_paper uses Base_seed + trial_index, where trial_index starts at 1.
[nreq_result, robust, cfg] = experiments_paper( ...
    'N_mc', n_mc, 'Base_seed', seed_first - 1, ...
    'N_req_list', opt.N_req_list, ...
    'Run_robustness', opt.Run_robustness, ...
    'N_workers', opt.N_workers, ...
    'Output_dir', output_dir, ...
    'Output_tag', sprintf('%s_seed%d_%d', label, seed_first, seed_last));

metadata = struct();
metadata.machine_label = label;
metadata.seed_first = seed_first;
metadata.seed_last = seed_last;
metadata.created_at = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss');
metadata.matlab_release = version('-release');
metadata.run_robustness = opt.Run_robustness;
output_file = fullfile(output_dir, 'shard_results.mat');
save(output_file, 'metadata', 'cfg', 'nreq_result', 'robust', '-v7.3');
fprintf('Shard complete: %s\n', output_file);
end
