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

| Identifier | APs `M` | UEs `K` | targets `P` | validated setting |
|---|---:|---:|---:|---:|
| `M9_K4_P3` | 9 | 4 | 3 | 20 common seeds, `N_req=2:6` |
| `M12_K6_P3` | 12 | 6 | 3 | 8 common seeds, `N_req=3`, four workers, 60 s MOSEK limit |
| `M16_K8_P4` | 16 | 8 | 4 | exploratory only; excluded from statistical claims |

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

## Frozen results and permitted paper use

The detailed numerical values, raw-artifact paths, and manuscript-ready text
are frozen in [`LARGE_SCALE_VALIDATION_RESULTS.md`](LARGE_SCALE_VALIDATION_RESULTS.md).
Use the data at the following levels of evidence.

| Result | What it supports | Permitted use |
|---|---|---|
| `M9_K4_P3_seed01to20` | The `N_req` trade-off, feasibility-rate comparison, and conditional power comparison over 20 common realizations | Main large-scale statistical figure and table |
| `M12_K6_P3_Nreq3_seed01to08_workers4_t60` | Robustness and computation cost at a higher-dimensional operating point | Scalability figure and supporting discussion, not an `N_req` sweep |
| `smoke_M16_K8_P4` and early M12 15 s runs | Solver/resource stress boundary | Do not cite as physical infeasibility or average performance |

For the validated M12 setting, the proposed method is physically feasible in
8/8 common seeds. FIM-greedy and nearest-AP are feasible in 6/8, and random
association in 1/8. Conditional power must be compared only on the common
feasible subset: the proposed method saves 2.05% versus FIM-greedy over their
six common feasible seeds and 59.19% versus nearest-AP over its six common
feasible seeds. This distinction prevents a heuristic from appearing cheap
merely because difficult infeasible cases are omitted.

All proposed M12 solutions have a normalized PCRB close to one, confirming
that their power values meet the tracking requirement tightly. The numerical
association audit also confirms the exact three-AP-per-target cardinality. A
small SDP-level leakage below `1e-5 W` is treated as numerical residual rather
than physical participation; the reporting threshold in
`evaluate_isac_metrics.m` is set to `1e-6 W` accordingly.

## Recommended remaining experiments and figures

The present package is sufficient to establish a larger-network result, but
the following additions would make the final paper more defensible and more
complete. They are ordered by expected review value.

1. **Double-DC ablation table/figure (highest priority).** On the same M9
   common seeds, compare SDR lower bound, rank-only DC, binary-only DC, and
   full double-DC recovery. Report feasibility, power on the common-feasible
   subset, binary residual, rank residual, and runtime. This is the most
   direct evidence that the mathematical algorithm—not only the FIM topology
   heuristic—contributes to the final physical solution.

2. **Robust-CSI outage comparison (highest priority).** Design at
   `eps_h=0` and at the nominal robust radius, then test each accepted design
   under many independently sampled channel errors inside the uncertainty
   ball. Plot empirical SINR-outage probability versus error radius. This
   validates the S-procedure rather than merely stating its constraint.

3. **QoS-tightness distribution.** Add CDFs or boxplots of
   `tr(J_p^{-1})/Gamma_track,p`, robust-LMI slack, and sensing-SINR margin for
   the M9 proposed samples. It closes the common reviewer concern that lower
   transmit power was obtained by hidden QoS relaxation.

4. **M12 cardinality slice (optional, compute-intensive).** Extend the
   validated 60 s / four-worker configuration to `N_req={2,4,5}` with at
   least 5 common seeds each. This would turn the M12 result from a fixed-point
   scalability validation into a partial high-dimensional `N_req` trade-off.
   Do not use the earlier 15 s data for this figure.

5. **Topology/stability visualization (optional).** For a representative M9
   geometry, show the proposed AP-target cluster and per-AP communication and
   sensing powers at two QoS operating points. State explicitly if the binary
   topology remains unchanged while covariances/powers adapt; this is a model
   insight, not a failure of the optimizer.

6. **Runtime scaling plot (optional).** Plot median and 90th-percentile
   runtime versus `M` using M6, M9, and M12, with the solver time limit stated
   in the caption. This is preferable to claiming arbitrary real-time
   scalability from a single workstation experiment.

Avoid adding M16 averages unless a separately profiled, numerically stable
solver budget is established. Also avoid reporting random-baseline conditional
power as a central comparison when its feasibility count is one.

## Run

```matlab
addpath('sim/matlab');
run_large_scale_algorithm_validation('Seeds',1:50, ...
    'Tradeoff_seeds',1:10,'T_max',3,'Mosek_max_time',15,'Resume',true);
```

The campaign is restartable. Each scale configuration is written separately
under `experiment_packages/v1.0/results/large_scale_algorithm_validation`.
Use paired statistics on common feasible seeds; never pool conditional means
from different feasibility sets. For a high-dimensional run, profile a small
two-worker calibration before increasing worker count; the validated M12
configuration uses four workers and a 60 s MOSEK subproblem budget.
