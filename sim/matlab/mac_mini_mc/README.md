# Mac mini Monte Carlo package

This folder is the Mac-specific entry point. It depends on the MATLAB code in
the parent `sim/matlab` folder, so clone the whole repository at the same Git
commit as the Windows host.

## Prerequisites

Use the same MATLAB release, CVX release, solver choice, and solver tolerances
as the Windows host. Run `cvx_setup` once after installing CVX. On a 16 GB Mac
mini, use one MATLAB process and do not enable `parfor` workers.

From the repository root, verify the installation:

```matlab
addpath('sim/matlab');
prm = generate_scenario(4,2,1,1,1,20,'auto', ...
    'AreaSize',200,'N_req',4,'eps_h',0.02,'seed',7,'Gamma_alpha',10);
b = ones(prm.M,prm.P);
res = solve_p3_with_fixed_b(prm,b,5,1e-5,1,0,1.3);
assert(res.is_physical_feasible);
```

## Seed shard

For a 50-seed main study, use Windows seeds `2027:2051` and Mac mini seeds
`2052:2076`. Seed intervals must not overlap. Run the robustness sweep on one
host only.

```matlab
addpath('sim/matlab');
run_mac_mini_main(2052, 2076, ...
    'N_req_list', 1:6, 'Run_robustness', false);
```

The raw records are saved under `mc_shards/macmini_seed2052_2076/` in
`shard_results.mat`. Copy that complete folder to the Windows host. Merge raw
records, rather than averaging per-shard CSV summaries, so feasibility
denominators and paired comparisons remain valid.

For a five-seed preflight:

```matlab
run_mac_mini_main(2052, 2056, ...
    'N_req_list', 1:4, 'Run_robustness', false);
```
