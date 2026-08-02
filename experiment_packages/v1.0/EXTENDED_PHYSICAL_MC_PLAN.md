# Extended physical-setting Monte-Carlo campaign

## Goal

Test whether the proposed certified binary recovery remains effective when the
physical network is enlarged, loaded more heavily, or placed in controlled
communication-sensing conflict geometries.  Every comparison uses identical
realizations for Proposed, FIM-greedy, and oracle nearest-AP topologies.

## Common settings

- 30 common seeds per configuration; `T_max=3` and MOSEK time limit 15 s.
- Dedicated-sensing participation floor: `Pmin_sen=0.01Pmax`.
- Robust CSI radius: `eps_h=0.05`.
- `Gamma_track='auto'` is recalibrated independently per scenario.  This tests
  algorithmic adaptability.  The saved per-target thresholds must be reported
  when interpreting absolute PCRB difficulty across geometry/area sweeps.
- Feasibility is unconditional; all performance means are conditional on
  physical feasibility.

## Factorized configurations

The reference configuration is `M=6, Nt=2, K=3, P=2, Nreq=3`, 400 m square,
and 20 dBm/AP.  Exactly one factor changes in each family:

| Family | Levels | Physical question |
|---|---|---|
| AP count | `M={4,6,8,10}` | Spatial diversity versus SDP cost |
| Antennas/AP | `Nt={1,2,4}` | Local array gain versus distributed cooperation |
| UE load | `K={2,3,4,5}` | Multiuser interference and robust QoS pressure |
| Target load | `P={1,2,3}` | Competition among dedicated sensing covariances |
| Area side | `{200,400,600}` m | Path loss and geometric dilution |
| AP power budget | `{17,20,23}` dBm | Hardware-energy operating range |
| Pressure geometry | edge-colocated UE/target; crowded targets | Cross-tier sensing interference and FIM geometry stress |

This produces 22 configurations, 660 scenarios, and 1,980 physical solves.
Random topology is not repeated because its failure mode is already quantified
in the primary 30-seed comparison.

## Parallel execution

From the repository root:

```matlab
addpath('sim/matlab');
% One-configuration, two-worker smoke test before the full campaign.
run_extended_physical_mc('Seeds',1:2,'N_workers',2, ...
    'Configuration_ids',"ap_count_M6",'T_max',3,'Mosek_max_time',15);

% Full 22-configuration campaign.
run_extended_physical_mc('Seeds',1:30,'N_workers',6, ...
    'T_max',3,'Mosek_max_time',15,'Resume',true);
plot_extended_physical_mc;
```

The program creates one result MAT file per physical configuration before
moving to the next.  Re-running with `Resume=true` skips completed
configurations.  The client-side `progress.log` receives worker completion
messages, allowing terminal monitoring without concurrent file writes from
workers.

## Required final checks

1. Every feasible record has finite power, PCRB ratio no larger than the
   documented numerical tolerance, and exactly `P*Nreq` active AP-target
   sensing pairs.
2. Compare proposed and baselines on shared feasible seeds for paired power
   statistics; never compare conditional averages as if they were unconditional.
3. Separate physical infeasibility, solver time limit, and other numerical
   errors in the final failure table.
