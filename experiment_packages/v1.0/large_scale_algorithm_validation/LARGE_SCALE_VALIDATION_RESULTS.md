# Large-Scale Algorithm Validation: Frozen Results and Paper-Ready Analysis

## Status and scope

This note freezes only results that have completed all required checks.  It is intended as source material for the final numerical-results section, rather than as a replacement for the experiment scripts or raw data.

The `M=9` campaign is the formal cluster-cardinality sweep, while the `M=12` campaign is an independently audited eight-seed high-dimensional validation at a fixed `N_req=3`.  The exploratory `M=16` run remains excluded.  In particular, a solver failure under a memory-constrained high-dimensional SDP is **not** interpreted as physical infeasibility.

All results use the dedicated sensing-waveform architecture: the binary variable `b(m,p)` authorizes AP `m` to transmit the dedicated waveform for target `p`; communication beamformers remain globally coordinated.  Hence, the association mainly controls the sensing cluster and its Fisher-information geometry, while the continuous SDP optimizes the communication and sensing covariances under the joint QoS and per-AP power constraints.

## Reproducible configuration

| Item | Formal large-scale Monte Carlo setting |
|---|---|
| APs / antennas per AP | `M=9`, `Nt=2` |
| Communication users / targets | `K=4`, `P=3` |
| Sensing-cluster cardinality | `N_req in {2,3,4,5,6}` |
| Channel uncertainty radius | `eps_h=0.05` |
| Per-AP power budget | 20 dBm |
| Random scenarios | 20 common seeds (`1:20`) |
| DC-SCA iteration budget | `T_max=3` |
| Solver | CVX with MOSEK; 15 s solver limit per convex subproblem |
| Compared association rules | Proposed recovery, FIM-greedy, nearest-target AP, random feasible-cardinality assignment |

For every method and seed, a fixed-assignment continuous re-optimization is performed before accepting a result.  A result is counted as feasible only when the re-optimized covariance solution passes the physical validation checks.  The reported AP count is the number of APs with nonzero dedicated sensing power, not merely the number of authorized AP-target entries.

## Formal Monte Carlo results: `M=9`, `Nt=2`, `K=4`, `P=3`

The complete raw result is stored in [`nreq_method_performance_final.mat`](../results/large_scale_algorithm_validation/M9_K4_P3_seed01to20/nreq_method_performance_final.mat), with the table and figure in the corresponding `figures/` directory.

| `N_req` | Method | Feasible scenarios | Mean total power (mW) | Mean PCRB ratio | Mean sum rate (bit/s/Hz) | Mean sensing SINR | Active sensing APs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | Proposed | 20/20 | 43.00 | 1.0000 | 5.36 | 11.88 | 4.85 |
| 2 | FIM-greedy | 20/20 | 43.05 | 1.0000 | 5.35 | 11.70 | 4.85 |
| 2 | Nearest-target | 11/20 | 77.18 | 1.0000 | 5.67 | 14.95 | 4.82 |
| 2 | Random | 0/20 | — | — | — | — | — |
| 3 | Proposed | 20/20 | 33.63 | 1.0000 | 5.35 | 11.32 | 6.35 |
| 3 | FIM-greedy | 20/20 | 33.74 | 1.0000 | 5.30 | 11.32 | 6.25 |
| 3 | Nearest-target | 19/20 | 40.90 | 1.0000 | 5.27 | 12.37 | 6.16 |
| 3 | Random | 1/20 | 165.33 | 1.0000 | 4.73 | 14.03 | 8.00 |
| 4 | Proposed | 20/20 | 31.77 | 1.0000 | 5.38 | 11.28 | 7.50 |
| 4 | FIM-greedy | 20/20 | 31.96 | 1.0000 | 5.32 | 11.59 | 7.30 |
| 4 | Nearest-target | 20/20 | 33.69 | 1.0000 | 5.34 | 11.82 | 6.90 |
| 4 | Random | 8/20 | 163.74 | 0.9973 | 5.93 | 10.58 | 7.50 |
| 5 | Proposed | 20/20 | 32.14 | 1.0000 | 5.48 | 10.93 | 7.95 |
| 5 | FIM-greedy | 20/20 | 32.36 | 1.0000 | 5.41 | 11.17 | 7.95 |
| 5 | Nearest-target | 20/20 | 33.80 | 1.0000 | 5.49 | 11.90 | 8.00 |
| 5 | Random | 13/20 | 176.11 | 1.0000 | 5.39 | 12.57 | 8.46 |
| 6 | Proposed | 20/20 | 33.40 | 1.0000 | 5.63 | 10.66 | 8.35 |
| 6 | FIM-greedy | 20/20 | 33.63 | 1.0000 | 5.54 | 11.43 | 8.35 |
| 6 | Nearest-target | 20/20 | 34.59 | 1.0000 | 5.63 | 11.55 | 8.45 |
| 6 | Random | 15/20 | 112.01 | 1.0000 | 5.42 | 12.47 | 8.40 |

`Mean PCRB ratio` denotes the average of `tr(J_p^{-1}) / Gamma_track,p` over feasible target instances.  Values close to one show that the sensing-accuracy constraint is active rather than being silently relaxed.  The fixed-assignment audit also gives an active-pair invariant rate of one for every feasible method/configuration pair: every authorized AP-target pair required by the selected topology is represented consistently in the accepted recovery solution.

## What the formal result establishes

1. **The proposed optimization is reliable at a larger network size.**  For all five cluster cardinalities, the proposed algorithm succeeds in all 20 common random scenarios.  This is not just a continuous relaxation result: each accepted sample has an integer association and a subsequently re-optimized physical covariance solution.

2. **FIM geometry is the dominant association driver in the present asymmetric-gating architecture.**  The FIM-greedy baseline is also feasible in all scenarios and is close in power to the proposed result (roughly `0.1–0.7%` higher across the sweep).  This is an informative outcome, not evidence that the continuous optimization is redundant.  It means that the discrete sensing-cluster choice is strongly determined by target/AP geometry, while the proposed SDP/DC-SCA stage finds the minimum-power joint communication–sensing covariance for that topology and certifies it under robust QoS constraints.

3. **The continuous optimization provides the physical beamforming and QoS guarantee.**  The FIM rule only constructs an association; it does not itself choose communication covariances, dedicated sensing covariances, robust S-procedure multipliers, or AP-level power allocation.  All methods in the table use the same fixed-assignment recovery SDP to obtain a physically comparable solution.  Therefore, the small proposed-versus-FIM gap should be described as a topology-stability observation under the current model, whereas the recovery optimization remains essential for obtaining valid transmit covariances and the reported power.

4. **Topology quality matters for feasibility and energy efficiency.**  At `N_req=2`, nearest-target association is feasible in only 55% of scenarios and consumes about 79% more power than the proposed method on its successful cases.  Random association has no feasible case.  The deficit persists at larger cardinalities: random association reaches only 75% feasibility even at `N_req=6` and is several times more power intensive when it succeeds.  Thus, geometry-aware sensing clustering is necessary; merely authorizing the required number of APs is insufficient.

5. **There is a non-monotonic cluster-cardinality trade-off.**  The proposed mean power falls from 43.00 mW at `N_req=2` to its minimum of 31.77 mW at `N_req=4`, then rises mildly.  A small cluster lacks spatial FIM diversity and requires extra power to meet the PCRB target.  Beyond the useful diversity level, authorizing more dedicated sensing transmissions increases the jointly managed waveform/interference burden without producing a proportional information gain.  This supports using `N_req` as an explicit design variable rather than fixing it arbitrarily.

6. **The constraints are used tightly and consistently.**  The PCRB ratio is essentially one for all methods after re-optimization.  The near-identical sum-rate and sensing-SINR values should not be interpreted as missing trade-off: they reflect a power-minimization problem in which the QoS constraints are met at their design boundary, while the association rule primarily changes how much power is required to reach that boundary.

## Suggested manuscript wording

> Fig. X evaluates the association and recovery procedures in a larger cell-free network with nine APs, four UEs, and three targets over 20 common channel and geometry realizations.  The proposed method attains a 100% physical feasibility rate for all examined sensing-cluster cardinalities.  Its required transmit power is minimized near `N_req=4`, which demonstrates the competing effects of Fisher-information diversity and dedicated sensing-waveform resource consumption.  In contrast, the nearest-target and random association rules exhibit substantial feasibility losses, particularly for small sensing clusters.  The FIM-greedy rule remains close to the proposed method, indicating that target–AP Fisher-information geometry is the principal determinant of the discrete sensing cluster under the adopted asymmetric-gating architecture.  The proposed DC-SCA/recovery procedure is nevertheless required to construct the robust communication and sensing covariances, enforce the per-AP power constraints, and certify physical feasibility for the selected cluster.

> The normalized PCRB values remain close to one in all feasible realizations, confirming that the reported energy reduction is achieved at the sensing-accuracy boundary rather than by relaxing the tracking requirement.  The observed non-monotonic power profile further motivates optimizing the cluster cardinality instead of selecting the largest available sensing cluster by default.

## Extended high-dimensional validation: `M=12`, `Nt=2`, `K=6`, `P=3`

To extend the validation beyond the `M=9` Monte Carlo sweep, a higher-dimensional common-seed study was conducted at `N_req=3`.  The larger configuration has 24 distributed transmit dimensions, six robust communication constraints, and three target-specific sensing covariance matrices.  It was run on four isolated MATLAB workers with a 60 s MOSEK limit per convex subproblem.  This longer limit is intentional: the 15 s budget used in an early stress attempt created solver-time-limit failures that cannot be conflated with physical infeasibility.

| Item | M12 validation setting |
|---|---|
| Dimensions | `M=12`, `Nt=2`, `K=6`, `P=3`, `N_theta=2` |
| Association cardinality | `N_req=3` |
| Common random scenarios | 8 seeds (`1:8`) |
| Robustness / power | `eps_h=0.05`, 20 dBm per AP |
| Algorithm settings | `T_max=3`, MOSEK limit 60 s per convex subproblem |
| Execution | 4 isolated seed workers; 35.7 min wall-clock time |
| Peak process footprint | approximately 9.7 GB private memory across the MATLAB master and four workers |

| Method | Feasible scenarios | Mean total power over feasible cases (mW) | Mean wall-clock time over all cases (s) |
|---|---:|---:|---:|
| Proposed full recovery | 8/8 | 38.55 | 496.3 |
| FIM-greedy topology | 6/8 | 37.88 | 151.2 |
| Nearest-target topology | 6/8 | 94.89 | 156.0 |
| Random topology | 1/8 | 256.28 | 68.6 |
| Continuous SDR lower bound | 8/8 finite | 31.71 | — |

The mean power of a baseline is conditional on its feasible samples and therefore must not be compared naively across differing feasibility sets.  On the six seeds where FIM-greedy and the proposed method are both feasible, the proposed method uses 37.10 mW versus 37.88 mW for FIM-greedy, a 2.05% reduction.  On the six common feasible seeds for nearest-target association, the proposed method uses 38.72 mW versus 94.89 mW, a 59.19% reduction.  On the sole random common-feasible seed, the reduction is 84.52%.  The proposed method is feasible on all eight seeds; this robustness advantage is the central M12 result.

For the eight proposed solutions, the mean normalized PCRB is `0.999987` (range `0.999943–0.999998`), the mean sum rate is 7.85 bit/s/Hz, and the mean sensing SINR is 13.07 dB.  Thus, the increased-scale results remain at the PCRB boundary and do not obtain their power values by weakening the tracking constraint.  The post-solve association audit verified exact cardinality for all 21 feasible method samples.  The largest dedicated-sensing leakage into an unauthorized AP-target pair was `7.40e-7 W`, below the `1e-5` physical-feasibility tolerance; the metric reporting threshold was consequently raised to `1e-6 W` in the current code to prevent such solver-level leakage from being counted as an active pair.

### Suggested manuscript wording for the M12 experiment

> To examine scalability, we further consider a 12-AP cell-free ISAC network serving six UEs and tracking three targets.  At `N_req=3`, the proposed method recovers a physically feasible binary association and covariance solution for all eight common realizations, whereas the FIM-greedy and nearest-AP baselines are feasible in only six realizations and random association succeeds once.  On their common feasible realizations, the proposed design reduces the transmit power by 2.05% relative to FIM-greedy selection and by 59.19% relative to nearest-AP selection.  The normalized PCRB remains close to one, demonstrating that the gain is achieved while satisfying the tracking requirement tightly.  This result also distinguishes association robustness from conditional power: a heuristic can appear inexpensive only because its difficult realizations are excluded after infeasible recovery.

## M12 cardinality sweep: `N_req=2:5`

A second M12 study uses the same physical setting and five common seeds (`1:5`) to
compare `N_req in {2,3,4,5}`.  The `N_req=3` entries are drawn from the
independent eight-seed validation but restricted to these same five seeds; hence
every point in the following table uses an identical seed set.  Power is a
conditional mean over physically feasible recoveries, while runtime is averaged
over all attempted recoveries.

| `N_req` | Method | Feasible | Conditional mean power (mW) | Mean runtime (s) | Mean PCRB ratio |
|---:|---|---:|---:|---:|---:|
| 2 | Proposed | 4/5 | 45.07 | 440.47 | 0.999983 |
| 2 | FIM-greedy | 3/5 | 48.20 | 127.30 | 0.999989 |
| 2 | Nearest-AP | 2/5 | 43.19 | 107.33 | 0.999992 |
| 2 | Random | 0/5 | -- | 39.94 | -- |
| 3 | Proposed | 5/5 | 37.61 | 538.94 | 0.999986 |
| 3 | FIM-greedy | 4/5 | 37.58 | 166.14 | 0.999983 |
| 3 | Nearest-AP | 5/5 | 104.40 | 189.58 | 0.999788 |
| 3 | Random | 1/5 | 256.28 | 78.91 | 0.960474 |
| 4 | Proposed | 5/5 | 35.72 | 501.46 | 1.000003 |
| 4 | FIM-greedy | 5/5 | 35.80 | 176.19 | 0.999991 |
| 4 | Nearest-AP | 5/5 | 44.61 | 174.15 | 1.000007 |
| 4 | Random | 0/5 | -- | 46.14 | -- |
| 5 | Proposed | 5/5 | 35.76 | 476.71 | 1.000032 |
| 5 | FIM-greedy | 5/5 | 36.10 | 172.58 | 1.000614 |
| 5 | Nearest-AP | 5/5 | 36.95 | 160.38 | 1.000206 |
| 5 | Random | 2/5 | 401.60 | 102.05 | 0.997928 |

The proposed approach reaches its lowest observed conditional power at
`N_req=4`; the plateau from four to five associated APs supports the
FIM-diversity versus waveform-resource trade-off seen in the M9 study.  At
`N_req=2`, feasibility is the informative metric: conditional power from the
few successful heuristic samples is not a fair basis for ranking methods.
The raw HDF5 MAT files were audited after the run: all 62 feasible records over
the M12 fixed-cardinality and sweep studies have exact per-target cardinality,
and the largest sensing power in an unauthorized AP-target entry is
`7.40e-7 W`, below the `1e-5 W` physical validation tolerance.

## Scale-stress diagnostics and excluded runs

An earlier M12 sweep with a 15 s MOSEK subproblem limit generated mixed `Failed` and `Infeasible` statuses, including under a five-worker launch.  It is retained for debugging only and is not used in any table or figure.  The validated M12 result above establishes that a four-worker launch is safe when the solver is given the 60 s budget required by this high-dimensional SDP.  Solver outcomes must still be reported separately from model feasibility.

### `M=16`, `Nt=2`, `K=8`, `P=4`

An exploratory seed-1 run generated repeated CVX/MOSEK failures, including a failed SDR reference solve.  This point lies outside the validated numerical operating region of the current workstation configuration and solver limits.  It is intentionally excluded from all figures, averages, and feasibility-rate claims.

## Artifact inventory

- Formal raw output: `experiment_packages/v1.0/results/large_scale_algorithm_validation/M9_K4_P3_seed01to20/nreq_method_performance_final.mat`
- Formal CSV: `experiment_packages/v1.0/results/large_scale_algorithm_validation/M9_K4_P3_seed01to20/figures/table_method_comparison_vs_nreq.csv`
- Formal figure: `experiment_packages/v1.0/results/large_scale_algorithm_validation/M9_K4_P3_seed01to20/figures/fig10_method_comparison_vs_nreq.png`
- M12 raw output: `experiment_packages/v1.0/results/large_scale_algorithm_validation/M12_K6_P3_Nreq3_seed01to08_workers4_t60/nreq_method_performance_final.mat`
- M12 per-seed shards and execution log: `experiment_packages/v1.0/results/large_scale_algorithm_validation/M12_K6_P3_Nreq3_seed01to08_workers4_t60/`
- M12 CSV table: `experiment_packages/v1.0/results/large_scale_algorithm_validation/M12_K6_P3_Nreq3_seed01to08_workers4_t60/table_m12_fixed_nreq_validation.csv`
- M12 paper figure: `experiment_packages/v1.0/results/large_scale_algorithm_validation/M12_K6_P3_Nreq3_seed01to08_workers4_t60/fig11_m12_scalability_validation.png`
- M12 cardinality raw output: `experiment_packages/v1.0/results/large_scale_algorithm_validation/M12_K6_P3_nreq2_4_5_seed01to05_workers4_t60/nreq_method_performance_final.mat`
- M12 cardinality table and figure: `experiment_packages/v1.0/results/large_scale_algorithm_validation/M12_K6_P3_nreq2_4_5_seed01to05_workers4_t60/table_m12_nreq_scalability.csv` and `fig12_m12_nreq_scalability.png`
- Sharded reproducible runner: `sim/matlab/run_large_scale_nreq_sharded.m`
- Per-seed MC implementation: `sim/matlab/run_nreq_method_performance_mc.m`
- Plotting and audit implementation: `sim/matlab/plot_nreq_method_performance_mc.m`
- Fixed-cardinality scalability plotter: `sim/matlab/plot_large_scale_fixed_nreq_validation.m`

## Reporting guardrails

- Do not describe M12/M16 solver failures as proof of physical infeasibility.
- Do not claim a statistically validated M12/M16 feasibility rate until a memory-safe multi-seed run is complete.
- Do state that the formal `M=9` study uses common seeds, fixed-topology recovery, and post-solve physical validation.
- Do distinguish the roles of discrete association (geometry-aware cluster selection) and continuous optimization (robust covariance/beamforming design and minimum-power recovery).
