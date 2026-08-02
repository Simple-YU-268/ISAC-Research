# Extended Physical-Setting Monte Carlo Study: Frozen Results

## Purpose and status

This document freezes the completed physical-setting Monte Carlo campaign for
use in the final paper.  It separates claims supported by the data from claims
that the data do **not** support.  All numerical values below are taken from
the audited result set in `results/extended_physical_mc/`.

**Frozen status:** completed on 2026-07-28.

- Physical configurations: 22
- Common random seeds per configuration: 30
- Generated records: 660
- Physical method runs: 1,980 (Proposed, FIM-greedy, and Nearest-AP)
- Solver stack: MATLAB R2026a, CVX, and MOSEK 11.2.2
- Parallel execution: 6 local MATLAB workers

## Model and protocol held fixed

The experiment uses the current partially shared Cell-Free ISAC model.

- Baseline physical setting: \(M=6\), \(N_t=2\), \(K=3\), \(P=2\),
  \(N_\theta=2\), \(N_{\rm req}=3\), 400 m square deployment, and 20 dBm
  per-AP power budget.
- Communication beams are globally cooperative.  The binary association
  \(b_{mp}\) authorizes only the dedicated sensing covariance for target
  \(p\) at AP \(m\).
- The dedicated sensing waveform contributes interference at the UE; robust
  communication constraints use the S-procedure with \(\epsilon_h=0.05\).
- Each selected AP-target pair has a positive sensing-power floor of 1% of
  the AP power budget.  Therefore every validated solution has exactly
  \(P N_{\rm req}\) nonzero AP-target sensing pairs.
- `Gamma_track='auto'` is used, and each target's PCRB is represented by the
  full matrix Schur-complement LMI using the MISO-equivalent FIM.
- The three methods are evaluated on exactly the same realization.  Once a
  topology is fixed, FIM-greedy and Nearest-AP use the same fixed-assignment
  robust beamforming and dedicated sensing-covariance re-optimization as the
  proposed method.  Hence their differences isolate topology/recovery quality,
  not a different continuous beamforming solver.
- The campaign uses at most three DC-SCA iterations (`T_max=3`) and a 15 s
  MOSEK limit per SDP call.  Reported runtime is end-to-end method runtime,
  including the proposed recovery procedure.

## Factors and controlled geometries

Only one physical factor changes in each family.

| Family | Levels |
|---|---|
| AP count \(M\) | 4, 6, 8, 10 |
| Antennas per AP \(N_t\) | 1, 2, 4 |
| UE load \(K\) | 2, 3, 4, 5 |
| Target load \(P\) | 1, 2, 3 |
| Deployment side length | 200, 400, 600 m |
| Per-AP budget | 17, 20, 23 dBm |
| Stress geometries | edge UE-target co-location; crowded targets |

## Data-quality and physical-validity audit

The audit checks that completed configurations have every requested seed,
identical method order, finite positive power, and the dedicated sensing
participation invariant.  It passed with the following result:

```text
EXTENDED CAMPAIGN AUDIT PASSED: 660 scenarios,
feasible [Proposed/FIM-greedy/Nearest-AP] = [630/630/600], errors = 30.
```

The 30 error records are all from \(N_t=1\).  They are **not** SDP
infeasibility outcomes.  Before any method is optimized, automatic PCRB
calibration rejects the scenario with
`generate_scenario:UnobservableReferenceTarget`: the isotropic-reference FIM
is singular/ill-conditioned.  Accordingly, \(N_t=1\) must be reported as an
**automatic-reference observability rejection**, not as an algorithm failure
or a physical-feasibility rate of zero.  All physical-method comparisons below
therefore use the remaining 630 scenarios.

For these 630 scenarios:

- Proposed: 630/630 validated physical solutions.
- FIM-greedy: 630/630 validated physical solutions.
- Nearest-AP: 600/630 validated physical solutions (95.24%).
- Participation invariant violations: 0.
- Maximum recomputed mean PCRB ratio: 1.000272.  This is a 0.0272% numerical
  excess consistent with the 1e-5 **absolute** feasibility tolerance used by
  the solver-side validator and scenario-scaled PCRB thresholds.
- Smallest reported communication margin: 0.255 dB.
- Smallest reported sensing-SINR margin: \(9.49\times10^{-4}\) dB.

## Main numerical findings

### 1. FIM-greedy is a strong topology baseline

On the nominal \((M,N_t,K,P)=(6,2,3,2)\) configuration, the conditional mean
powers are:

| Method | Feasibility | Conditional mean power |
|---|---:|---:|
| Proposed | 30/30 | 32.194 mW |
| FIM-greedy | 30/30 | 32.241 mW |
| Nearest-AP | 30/30 | 41.978 mW |

The Proposed-to-FIM-greedy power difference is only 0.15% on this point.
This is scientifically meaningful: under the present architecture, the FIM
already captures the dominant geometric information needed for selecting a
sensing cluster.  The proposed binary-DC recovery realizes essentially this
quality while jointly validating the binary topology and robust covariance.
The data do **not** support a claim of a large power gain over FIM-greedy.

### 2. Distance-only association is materially weaker

The proposed method has a sizeable paired power advantage relative to the
oracle Nearest-AP topology, which is allowed to use actual target positions and
is therefore a favorable theoretical baseline for a distance rule.

| Setting | Paired trials | Proposed mean power | Nearest-AP mean power | Proposed paired reduction |
|---|---:|---:|---:|---:|
| \(M=4\) | 29 | 32.140 mW | 38.051 mW | 12.43% |
| \(M=6\) | 30 | 32.194 mW | 41.978 mW | 13.14% |
| \(M=8\) | 30 | 32.188 mW | 44.582 mW | 18.24% |
| \(M=10\) | 30 | 32.924 mW | 42.156 mW | 15.39% |
| \(K=4\) | 28 | 35.466 mW | 58.085 mW | 21.27% |
| \(P=3\) | 30 | 32.397 mW | 48.054 mW | 19.34% |
| Edge co-location | 29 | 41.590 mW | 89.346 mW | 49.57% |
| Crowded targets | 7 | 54.415 mW | 295.676 mW | 81.44% |

`Paired trials` means both methods returned a validated physical solution.
The comparison is therefore not biased by using a failed Nearest-AP run in
its conditional mean power.

### 3. Controlled stress geometries create the clearest feasibility gap

| Geometry | Proposed | FIM-greedy | Nearest-AP |
|---|---:|---:|---:|
| Edge UE-target co-location | 30/30 | 30/30 | 29/30 |
| Crowded targets | 30/30 | 30/30 | 7/30 |

The crowded-target geometry is the strongest evidence in this campaign.  A
nearest-distance cluster often lacks the angular diversity required by the
PCRB/FIM constraint, despite having favorable path loss.  Geometry-aware
selection remains feasible in every trial.  The high 81.44% conditional power
reduction must be read together with the 7/30 Nearest-AP feasibility rate; it
does not represent an unconditional population average.

### 4. Load and physical-resource scaling

- The proposed and FIM-greedy methods remain validated for all tested
  \(K\in\{2,3,4,5\}\) and \(P\in\{1,2,3\}\) scenarios.
- Increasing \(N_t\) from 2 to 4 lowers the proposed conditional mean power
  from 32.194 mW to 14.634 mW, showing the value of additional spatial degrees
  of freedom.
- In the 17/20/23 dBm budget sweep, proposed conditional mean power is
  16.134/32.194/64.234 mW, respectively.  The rising value is expected because
  automatic PCRB calibration scales the tested tracking requirement with the
  available reference power; it should not be described as a conventional
  ``more budget costs more'' causal result.
- In the tested 200/400/600 m range, the auto-calibrated formulation produces
  nearly constant conditional powers.  This is also a consequence of the
  scenario-scaled PCRB threshold.  It is a robustness check, not evidence that
  propagation distance has no effect in a fixed-PCRB system.

## What this experiment proves for the paper

The study supports the following defensible claims.

1. The proposed double-DC recovery and fixed-topology re-optimization produce
   validated binary physical solutions across 630 diverse, automatically
   observable scenarios, with zero sensing-participation violations.
2. The complete proposed design achieves the same 100% validated feasibility
   as a strong FIM-greedy association baseline, while preserving a small but
   consistently nonnegative energy advantage in the reported factor sweeps.
3. Geometry-aware sensing clustering is essential.  Compared with a
   target-distance oracle, the proposed method has higher physical feasibility
   under stress and substantial paired power savings, particularly for crowded
   targets.
4. The continuous robust covariance optimization is necessary to make each
   selected cluster physically feasible under PCRB, sensing-SINR, S-procedure
   communication, total-power, and positive participation-floor constraints.

## Claims to avoid

- Do not claim a large or statistically significant power gain over
  FIM-greedy.  The gap is deliberately small in this architecture.
- Do not call the \(N_t=1\) records physical infeasibility; they are rejected
  by the automatic reference-FIM calibration before optimization.
- Do not use the 81.44% crowded-target reduction without also stating that
  Nearest-AP is feasible in only 7/30 trials.
- Do not interpret the area or power-budget sweeps as fixed-QoS propagation
  laws, because `Gamma_track='auto'` intentionally rescales the PCRB target.
- Do not use runtime as a superiority claim.  The proposed recovery is slower
  than a single fixed-topology solve because it includes topology recovery and
  candidate validation.

## Suggested manuscript text

> In a 22-configuration physical-setting study with 30 common seeds per
> configuration, the proposed recovery and FIM-greedy association produced
> validated physical solutions in all 630 automatically observable scenarios,
> whereas nearest-AP association was feasible in 600/630.  The small gap to
> FIM-greedy confirms that FIM geometry is a strong topology selector in the
> proposed partially shared architecture.  The decisive benefit of the joint
> geometry-aware design appears relative to distance-only association: in the
> crowded-target stress geometry, nearest-AP association was feasible in only
> 7/30 trials, while the proposed method remained feasible in every trial and
> reduced paired conditional transmit power by 81.4%.

## Deliverables and provenance

| Artifact | Role |
|---|---|
| `results/extended_physical_mc/extended_physical_mc_final.mat` | Full raw campaign result |
| `results/extended_physical_mc/extended_physical_mc_summary.csv` | Per-method conditional summaries |
| `results/extended_physical_mc/extended_pairwise_power_comparison.csv` | Paired Proposed-vs-baseline power analysis and 95% half-widths |
| `results/extended_physical_mc/extended_qos_audit.csv` | PCRB, SINR-margin, and runtime audit |
| `results/extended_physical_mc/extended_failure_classification.csv` | Seed-level method failure categories |
| `results/extended_physical_mc/fig11_extended_physical_factors.png` | Factor-sweep figure |
| `results/extended_physical_mc/fig12_pressure_geometries.png` | Stress-geometry figure |
| `sim/matlab/run_extended_physical_mc.m` | Reproducible campaign driver |
| `sim/matlab/audit_extended_physical_mc.m` | Physical-data audit |
| `sim/matlab/analyze_extended_physical_mc.m` | Paired statistical analysis |
| `sim/matlab/plot_extended_physical_mc.m` | Figure generator |

