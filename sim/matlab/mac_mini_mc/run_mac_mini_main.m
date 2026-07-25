function output_file = run_mac_mini_main(seed_first, seed_last, varargin)
%RUN_MAC_MINI_MAIN  Mac mini entry point for a reproducible MC seed shard.
%   Run this function from any MATLAB working directory after cloning the
%   repository. The shared experiment code remains in the parent folder.

this_dir = fileparts(mfilename('fullpath'));
matlab_dir = fileparts(this_dir);
addpath(matlab_dir);

output_file = run_mc_shard('macmini', seed_first, seed_last, varargin{:});
end
