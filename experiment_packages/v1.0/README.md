# ISAC Resource-Optimization Experiment Package v1.0

This package reproduces the non-perfect-CSI resource-allocation experiments
for the Cell-Free ISAC study.  It compares the proposed optimized AP--target
association with random and nearest-AP associations under identical channel
realizations.

## What is compared

All methods use the same continuous covariance optimization, robust SINR
constraints, PCRB constraints, sensing-SINR constraints, and per-AP power
budget.  They differ only in the AP--target sensing association.

| Method | Association rule |
|---|---|
| Proposed | Fixed-penalty DC-SCA, top-N projection, then deterministic one/two-AP swap recovery and fixed-assignment re-optimization. |
| Random | Uniform random selection of exactly `N_req` APs for each target, followed by the same fixed-assignment recovery. |
| Nearest AP | The `N_req` nearest APs for each target, followed by the same fixed-assignment recovery. |

The proposed recovery accepts a candidate only after all original physical
constraints have been checked.  Infeasible trials are retained in feasibility
statistics and excluded only from explicitly labelled conditional metrics.

## Software

- MATLAB R2026a (or compatible)
- CVX 2.2
- MOSEK 11.2, selected through `cvx_solver mosek`
- Parallel Computing Toolbox for `N_workers > 0`

The experiment code is in `sim/matlab/experiments_paper.m` and the recovery
implementation is in `sim/matlab/baseline_alg2.m`.

## Reproducible commands

Start MATLAB in `sim/matlab`, configure CVX/MOSEK once, then run:

```matlab
addpath('/path/to/cvx');
addpath('/path/to/mosek/toolbox/r2022b');
cvx_setup;
cvx_solver mosek;

% Pipeline smoke test: one paired realization, N_req = 3, five SCA iterations.
experiments_paper('N_mc', 1, 'N_req_list', 3, 'T_max', 5, ...
    'Run_robustness', false, 'N_workers', 0, ...
    'Output_dir', 'experiment_packages/v1.0/results/smoke', ...
    'Output_tag', 'v1_smoke');

% Formal paired Monte Carlo run.
experiments_paper('N_mc', 100, 'N_req_list', 1:6, 'T_max', 30, ...
    'Run_robustness', true, 'N_workers', 2, ...
    'Output_dir', 'experiment_packages/v1.0/results/formal', ...
    'Output_tag', 'v1_formal');
```

Use `N_workers=2` initially.  Increase only after verifying that the local
MATLAB and MOSEK licenses permit concurrent workers.

## Fixed parameters

| Parameter | Value |
|---|---:|
| APs `M` | 8 |
| Antennas per AP `Nt` | 4 |
| Users `K` | 4 |
| Targets `P` | 2 |
| CSI uncertainty radius `eps_h` | 0.05 |
| DC iterations | 30 formal / 5 smoke |
| Rank penalty `eta_rank` | 1.0 |
| Binary penalty `eta_b` | 1.0 |
| Recovery candidates | 21 (top-N, single-swap, two-swap beam) |

## Outputs and reporting rules

Each run creates Figures 1--6, a MAT file, and an `nreq_summary.csv` file.

1. Report feasibility first: a trial is feasible only after fixed-assignment
   recovery and physical-constraint validation.
2. Report power, rate, PCRB, and sensing SINR conditionally on feasible trials.
3. For direct method comparisons, also report paired statistics computed only
   on realizations where both methods are feasible.
4. Do not claim exact binary convergence from a finite DC penalty.  Figure 1
   reports the true fixed-penalty objective and rank/binary residuals; recovery
   success is established separately by physical feasibility checks.

`results/` is intentionally excluded from version control because Monte Carlo
outputs can be regenerated from the commands above.
