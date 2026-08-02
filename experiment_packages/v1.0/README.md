# Participation-Constrained Cell-Free ISAC Experiment Package v1.0

This is the authoritative package for the current model.  It supersedes the
earlier zero-lower-bound and `M=8, Nt=4` draft configurations.

## Current model and main configuration

Communication covariances remain globally cooperative.  The binary association
only authorizes target-specific dedicated sensing covariance:

\[
\sum_m b_{mp}=N_{\rm req},\qquad
P_{\min}^{\rm sen}b_{mp}\leq\operatorname{tr}(\mathbf E_m\mathbf S_p)
\leq P_{\max}b_{mp}.
\]

The main setting is `M=6`, `Nt=2`, `K=3`, `P=2`, `Ntheta=2`, `Pmax=0.1 W`,
`eps_h=0.05`, and `Pmin_sen=0.01 Pmax = 1 mW`.  The primary comparison uses
30 common seeds and `Nreq=2:6`.  A point is feasible only after binary
recovery, fixed-assignment re-optimization, and physical validation.

## Primary final artifacts

- `results/nreq_method_performance_30seeds/nreq_method_performance_final.mat`:
  150 common-seed scenarios and four physical methods.
- `figures/fig10_method_comparison_vs_nreq.png` and
  `figures/table_method_comparison_vs_nreq.csv`: final method comparison.
- `results/csi_robustness/csi_robustness_final.mat`: robust-CSI study.
- `results/participation_nreq_sweep_30seeds/nreq_qos_final.mat`: cluster-size
  and QoS study.
- `EXPERIMENT_RESULTS_SUMMARY.md`: claims, numerical evidence, and caveats.

## Reproduction

Start MATLAB from the repository root, add `sim/matlab` to the path, configure
CVX and MOSEK, then invoke the corresponding `run_*.m` script.  The core
common-seed comparison is:

```matlab
addpath('sim/matlab');
run_nreq_method_performance_mc('Seeds',1:30,'N_req_list',2:6, ...
    'T_max',3,'Mosek_max_time',10);
plot_nreq_method_performance_mc;
audit_current_model_method_comparison;
```

The plot reports feasibility unconditionally.  Power, sum rate, PCRB, and
sensing SINR are conditional on physical feasibility, as labeled in the figure.
The SDR is a continuous power lower bound only, not a physical competitor.

## Evidence map and publication-readiness guidance

The current artifact set closes the main empirical loop for the proposed
participation-constrained formulation:

| Question | Evidence | Interpretation rule |
|---|---|---|
| Does double-DC recovery produce physical binary solutions? | `figures/fig2_dual_dc_ablation.png` | Compare feasibility and binary residual together; a small residual alone is not a physical certificate. |
| How does cluster size affect feasibility and energy? | `figures/fig3_cluster_size_tradeoff.png`, `figures/fig10_method_comparison_vs_nreq.png` | Report feasibility unconditionally; report power only on the corresponding feasible set. |
| Are QoS constraints actually met? | `figures/fig4_qos_vs_cluster_size.png` and raw metric files | PCRB ratios near one indicate operation at the tracking boundary, not a relaxed sensing constraint. |
| Is the robust S-procedure useful? | `figures/fig9_csi_robustness.png` | Distinguish sampled outage evidence from the analytical uncertainty-set certificate. |
| Does the conclusion persist under geometry and dimension stress? | `results/extended_physical_mc/fig11_extended_physical_factors.png`, `fig12_pressure_geometries.png`, and the M12 artifacts below | Treat solver failures separately from model infeasibility. |
| Does the high-dimensional model retain the cluster-size trade-off? | `results/large_scale_algorithm_validation/M12_K6_P3_nreq2_4_5_seed01to05_workers4_t60/fig12_m12_nreq_scalability.png` | The M12 sweep uses five common seeds; it is scalability evidence, not a replacement for the larger M9 Monte Carlo sample. |

The core paper figures are therefore complete: system architecture, dual-DC
ablation, cluster-size/QoS behavior, method comparison, dimensional and
physical-setting scalability, CSI robustness, and communication-sensing
trade-off.  The remaining work before submission is editorial rather than a
missing mandatory experiment: freeze the code commit and solver versions,
state common-seed and conditional-mean conventions in every caption, and do
not pool solver `Failed` statuses with physical `Infeasible` statuses.

Optional follow-up experiments should be presented as extensions, not required
evidence for the static formulation: target motion with cluster handover,
hardware/fronthaul energy accounting, and a larger multi-antenna receiver
model.  Each changes the model scope and should be introduced only with a
corresponding mathematical formulation.
