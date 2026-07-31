# Large-scale algorithm-validation campaign

This package records the second-stage numerical campaign for the dedicated
sensing-participation model. It is separate from the primary small-scale study
so that large-network scalability and the role of `N_req` are reported without
overwriting validated primary data.

## Physical configurations

All configurations use a 400 m square, 2.8 GHz carrier, two ULA antennas per
AP, 20 dBm per-AP budget, normalized communication and sensing noise variance
one, 5 percent norm-bounded CSI uncertainty, 0 dB communication and sensing
SINR targets, and per-target auto-calibrated PCRB thresholds with
`Gamma_alpha=3`.

| Identifier | APs `M` | UEs `K` | targets `P` | `N_req` |
|---|---:|---:|---:|---:|
| `M9_K4_P3` | 9 | 4 | 3 | 2:6 |
| `M12_K6_P3` | 12 | 6 | 3 | 2:6 |
| `M16_K8_P4` | 16 | 8 | 4 | 2:6 |

For every realization, the proposed double-DC recovery, FIM-greedy sensing
topology, and oracle nearest-AP topology use identical channel and geometry
draws. Nearest AP remains an oracle geometry baseline, not a deployable policy
when the exact target position is unavailable.

## Evidence to report

1. `N_req` scalability: feasibility, transmit power, sum-rate, mean PCRB
   ratio, mean sensing SINR, number of nonzero sensing APs, and wall-clock time.
2. Double-DC convergence: power, rank residual, binary residual, and the
   continuation penalty for `M12_K6_P3, N_req=3`.
3. Communication-sensing game: common-seed sweep of robust SINR target and
   PCRB allowance, showing power demand and the feasibility boundary.
4. Ablation: continuous SDR, rank-only DC, binary-only DC, and full double-DC
   recovery. The ablation is deliberately separate from this throughput study.

## Run

```matlab
addpath('sim/matlab');
run_large_scale_algorithm_validation('Seeds',1:50, ...
    'Tradeoff_seeds',1:10,'T_max',3,'Mosek_max_time',15, ...
    'N_workers',4,'Resume',true);
```

The campaign is restartable. Each scale configuration is written separately
under `experiment_packages/v1.0/results/large_scale_algorithm_validation`.
With four workers, the three scale configurations and the combined
convergence/trade-off task run concurrently and write to disjoint directories.
Use paired statistics on common feasible seeds; never pool conditional means
from different feasibility sets.
