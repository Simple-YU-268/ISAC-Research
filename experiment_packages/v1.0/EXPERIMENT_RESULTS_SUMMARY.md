# Current-model experiment summary

This document records the experiments that support the participation-constrained
Cell-Free ISAC model used in the paper.  All physical results use dedicated
sensing covariances, the positive sensing-participation floor
`Pmin_sen = 0.01 Pmax`, binary topology recovery, fixed-assignment
re-optimization, and the final physical validator.

## Reproducible main setting

`M=6`, `Nt=2`, `K=3`, `P=2`, `Ntheta=2`, `Pmax=0.1 W`, `eps_h=0.05`, and
`Nreq` is varied where stated.  The main Monte-Carlo sets use 30 common
seeds.  A method is counted as feasible only when its returned physical
solution passes the validator.

## What each experiment establishes

| Evidence | Raw result | Defensible conclusion |
|---|---:|---|
| Binary-DC ablation | Median binary residual: `4.268e-1` (unpenalized SDR) to `6.229e-5` (dual-DC) | The binary penalty materially removes the continuous-relaxation artifact before discrete recovery. |
| Association comparison, `Nreq=3` | Proposed/FIM/nearest/random feasible trials: `30/30`, `30/30`, `30/30`, `10/30`; conditional mean power: `30.83`, `30.93`, `39.29`, `115.82` mW | FIM-aware recovery reaches the strong FIM geometry benchmark and is substantially better than path-loss-only and random association. |
| Sparse-cluster stress, `Nreq=2` | Proposed/FIM/nearest/random feasible trials: `30/30`, `30/30`, `24/30`, `1/30` | Geometry-aware association is especially important when sensing-cluster cardinality is restricted. |
| SDR lower bound, `Nreq=3` | SDR mean power `28.10` mW; physical proposed mean power `30.83` mW; median gap `9.73%` | The implementable binary solution remains close to the continuous lower bound. |
| Robust CSI experiment | At `eps_h={0.02,0.05,0.08}`, robust outage `0%`; nominal outage `89.0%`, `89.6%`, `91.9%` | The S-procedure-based design supplies a large robustness benefit at modest power cost. |
| Cluster-size sweep | Mean power at `Nreq={2,3,4,5,6}`: `37.65`, `30.83`, `29.50`, `29.74`, `30.94` mW | Adding sensing APs initially improves FIM geometry; mandatory per-pair sensing power produces a clear energy optimum near `Nreq=4`. |
| QoS tightness | Mean PCRB ratios are numerically near one for all feasible physical methods | The reported power savings are not obtained by relaxing the tracking requirement. |

## Required interpretation safeguards

1. The FIM-greedy rule is a deliberately strong topology baseline.  In the
   present architecture it is nearly matched by the proposed recovery, rather
   than being dramatically outperformed.  The paper should therefore claim
   robust binary recovery and near-lower-bound joint covariance optimization,
   not a universal superiority over FIM geometry.
2. Random-method power, sum rate, and sensing-SINR averages are conditional on
   feasibility.  They must never be compared as unconditional performance
   averages because infeasible trials are excluded from those statistics.
3. Sum rate is descriptive: it is computed from the returned covariance but
   is not the optimization objective.  The primary optimization objectives
   are physical feasibility and transmit-power minimization under PCRB and
   robust QoS constraints.
4. The rank penalty remains part of the general formulation, but no empirical
   rank-penalty gain should be claimed for the present small MU-MISO setting:
   its communication covariance is already almost rank one after SDR.
5. The single-scenario DC trajectory figure is diagnostic only.  The
   30-seed binary-residual ablation is the statistical convergence evidence.

## Primary artifacts

- `results/nreq_method_performance_30seeds/nreq_method_performance_final.mat`
- `figures/table_method_comparison_vs_nreq.csv`
- `figures/fig10_method_comparison_vs_nreq.png`
- `results/csi_robustness/csi_robustness_final.mat`
- `results/participation_nreq_sweep_30seeds/nreq_qos_final.mat`
- `results/participation_dual_dc_ablation_seeds6to30/final.mat`
- `../math_derivation_tex_v10/numerical_results.tex`
