# Paper Results and Figures Evidence Package

This folder is a self-contained evidence package for the current
participation-constrained Cell-Free ISAC paper.  It contains only artifacts
that are cited by `paper/numerical_results.tex` (plus the raw result files and
MATLAB generators needed to interpret them). Two supplementary archived
ablation figures are additionally retained for traceability; they are clearly
identified as supplementary and are not cited by the current TeX manuscript.

For a bilingual, figure-by-figure explanation of the data, conclusions,
reporting conventions, and claim boundaries, see
[`RESULTS_AND_FIGURES_BILINGUAL.md`](RESULTS_AND_FIGURES_BILINGUAL.md).

## Package layout

- `figures/`: the 13 PNG figures cited by the paper and two supplementary
  archived ablation figures (`figS1_*` and `figS2_*`).
- `results/`: final MAT/CSV data from which the paper figures and numerical
  claims are derived.  The `extended_physical_mc/` subfolder retains all
  factor-level raw outputs used by the two physical-setting figures.
- `scripts/`: the current MATLAB solver, validation, experiment, plotting, and
  audit routines relevant to this package.
- `paper/`: compiled manuscript PDF and the numerical-results TeX source.

## Model and reporting conventions

The binary variable `b(m,p)` authorizes AP `m` to transmit the dedicated
sensing waveform for target `p`; it does not gate globally cooperative
communication covariances.  Every accepted solution is obtained by integer
association recovery followed by fixed-assignment continuous re-optimization
and physical validation.

Physical feasibility is always an unconditional rate.  Power, sum rate, PCRB,
and sensing SINR are conditional on the feasible solutions of the stated
method unless the text explicitly says “common feasible set.”  The continuous
SDR value is a lower bound, not a physically feasible baseline.  Solver
`Failed`/time-limit outcomes must not be relabelled as physical infeasibility.

## Figure-to-data map

| Paper figure | What it establishes | Data in `results/` | Generator / audit |
|---|---|---|---|
| `fig1_system_architecture.png` | Asymmetric gating: global communication and target-specific dedicated sensing. | Deterministic schematic, no stochastic data. | `draw_system_architecture.m` |
| `fig2_dual_dc_ablation.png` | Necessity of both rank and binary DC penalties over 25 common scenarios. | `dual_dc_ablation_25seeds.mat` | `run_dual_dc_ablation.m`, `plot_dual_dc_ablation.m` |
| `fig3_cluster_size_tradeoff.png` | Main cluster-size feasibility, power, and active sensing-AP trade-off. | `nreq_qos_final.mat` | `run_nreq_qos_sweep.m`, `plot_nreq_qos_statistics.m` |
| `fig4_qos_vs_cluster_size.png` | PCRB, communication-SINR, and sensing-SINR constraint tightness versus `N_req`. | `nreq_qos_final.mat` | `run_nreq_qos_sweep.m`, `plot_nreq_qos_statistics.m` |
| `fig5_statistical_double_dc_convergence.png` | Common-seed DC-SCA stabilization from the unpenalized SDR point to a recovery-ready topology. | `statistical_double_dc_convergence_final.mat`, `statistical_double_dc_convergence_summary.csv` | `run_statistical_double_dc_convergence.m` |
| `figS1_power_gap_cdf.png` (supplementary) | Archived 100-realization empirical CDF of the physical-recovery power gap to the SDR lower bound. | `main_config_mc_100seeds_pilot_final.mat` | `plot_paper_experiment_figures.m` |
| `figS2_recovery_ablation.png` (supplementary) | Archived 30-realization comparison of FIM-only, direct DC Top-N rounding, and full recovery. | `recovery_ablation_30seeds_final.mat` | `run_recovery_ablation.m`, `plot_paper_experiment_figures.m` |
| `fig7_dimension_sensitivity.png` | Runtime/feasibility/power response to total transmit dimension. | `network_scaling_final.mat` | `run_network_scaling_study.m` |
| `fig8_statistical_tradeoff.png` | Mean transmit-power surface and feasibility heatmap over a 5x5 QoS grid and five common seeds. | `tradeoff_mc_final.mat` | `run_isac_tradeoff_surface.m`, `plot_isac_tradeoff_topology_slices.m` |
| `fig9_csi_robustness.png` | Robust S-procedure design versus nominal CSI design under sampled channel errors. | `csi_robustness_final.mat` | `run_csi_robustness_experiment.m`, `plot_csi_robustness.m` |
| `fig10_method_comparison_vs_nreq.png` | 30-seed method comparison: proposed, FIM-greedy, nearest-AP, random, and SDR lower bound. | `main_m6_nreq_method_performance_30seeds.mat` | `run_nreq_method_performance_mc.m`, `plot_nreq_method_performance_mc.m`, `audit_current_model_method_comparison.m` |
| `fig11_extended_physical_factors.png` | 30-seed sweeps of AP count, AP antennas, UE/target load, area, and AP power budget. | `extended_physical_mc/` | `run_extended_physical_mc.m`, `plot_extended_physical_mc.m`, `audit_extended_physical_mc.m` |
| `fig12_pressure_geometries.png` | Edge UE-target co-location and crowded-target geometry stress tests. | `extended_physical_mc/stress_*.mat` | `run_extended_physical_mc.m`, `plot_extended_physical_mc.m` |
| `fig11_m12_scalability_validation.png` | Larger `M=12`, `K=6`, `P=3`, `N_req=3` validation over eight common seeds. | `m12_nreq3_8seeds.mat`, `table_m12_fixed_nreq_validation.csv` | `run_large_scale_nreq_sharded.m`, `plot_large_scale_fixed_nreq_validation.m` |
| `fig12_m12_nreq_scalability.png` | Larger-system cluster-cardinality sweep for `N_req=2:5` over five common seeds. | `m12_nreq2_4_5_5seeds.mat`, `table_m12_nreq_scalability.csv` | `run_large_scale_nreq_sharded.m`, `plot_m12_nreq_scalability.m` |

The filenames preserve historical figure-number prefixes.  In the compiled
paper the actual numbering is determined by TeX order, so the two M12 figures
do not conflict with similarly prefixed physical-setting filenames.

## Main numerical findings

1. **Double-DC stabilization:** across 25 common scenarios, the unpenalized
   SDR binary distance has median `4.336e-1`; the first binary-DC step reduces
   it to `1.139e-3`, and the second step reaches `6.014e-5`.  All 25 top-N
   supports are unchanged and recovery-ready from the first DC step onward.
   This is a continuous-phase diagnostic, not a physical-feasibility claim.
2. **Double-DC recovery:** across 25 common scenarios, the dual penalty gives
   92% physical feasibility and median binary distance `5.94e-5`; the
   binary-only variant gives `5.62e-5`, whereas the unpenalized SDR and
   rank-only variants remain near `0.43`.  Rank residuals are near numerical
   precision for all ablation variants, which isolates the binary penalty's
   role.
3. **Cluster cardinality:** in the principal sweep, the proposed method is
   feasible on all common samples and its energy is minimized at an
   intermediate sensing-cluster size.  This reflects the balance between
   Fisher-information diversity and dedicated waveform/interference burden.
4. **Robustness:** under the sampled uncertainty radii in the paper, the robust
   design has zero sampled system outage while the nominal design has about
   89--92% mean outage.  This Monte Carlo observation complements the
   S-procedure certificate rather than replacing it.
5. **Geometry matters:** FIM-greedy is a strong association baseline because
   FIM geometry strongly determines the sensing cluster.  Nearest-AP and random
   associations lose feasibility or require substantially more power in the
   stress geometries.
6. **High-dimensional confirmation:** at `M=12`, `K=6`, `P=3`, the proposed
   recovery is feasible in 8/8 cases at `N_req=3`.  In the five-common-seed
   cardinality sweep, it is 4/5 feasible at `N_req=2` and 5/5 for
   `N_req=3:5`; its lowest observed conditional mean power is 35.72 mW at
   `N_req=4`.

## Independent audits already performed

- Every graphic cited by `paper/numerical_results.tex` exists in `figures/`.
- The TeX file contains 13 figure labels and 13 corresponding references.
- The paper compiled twice without unresolved references; the rendered page
  containing the new M12 sweep was visually inspected for clipping and
  overlap.
- Across the M12 fixed-cardinality and sweep studies, all 62 feasible records
  have exact per-target association cardinality.  The maximum dedicated sensing
  power in an unauthorized AP-target pair is `7.40e-7 W`, below the
  `1e-5 W` physical validation tolerance.

## Reproduction environment

The recorded runs use MATLAB R2026a, CVX 2.2, and MOSEK 11.2.2.  Configure CVX
and MOSEK before running scripts.  The M12 studies use four isolated MATLAB
workers and a 60-second MOSEK limit per convex subproblem; their reported
runtime is therefore a computational-cost measurement, not a real-time claim.

## Scope boundary

This package supports the static MISO-equivalent observation model used by the
paper.  Target motion/cluster handover, fronthaul-energy accounting, and a
multi-antenna receive architecture are valid future extensions but require a
new model and are not silently represented by these results.
