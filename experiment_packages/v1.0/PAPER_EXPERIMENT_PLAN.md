# Paper Experiment and Figure Plan

## Common Main Configuration

Use
\[
M=6,\quad N_t=2,\quad K=3,\quad P=2,\quad N_\theta=2,\quad N_{\rm req}=3.
\]

The area is 400 m by 400 m, each AP has a 20 dBm power cap, the CSI uncertainty radius is 0.05, both SINR targets are 0 dB, and PCRB thresholds use auto calibration with `Gamma_alpha=3`. A proposed result is counted only after fixed-b re-optimization and physical validation.

The derivation-consistent method is
\[
\text{SDR initialization} \rightarrow \text{dual DC-SCA} \rightarrow \text{binary recovery} \rightarrow \text{fixed-}b\text{ validation}.
\]

The D-optimal FIM, DC Top-N, and PCRB-slack-guided candidates are recovery components. SDR is a lower bound, not a feasible competitor. Oracle true-position nearest AP is excluded from primary plots.

## Figure 1: System Architecture

Draw six APs, three UEs, two targets, the central processor, global cooperative communication beams, and target-specific sensing waveform covariances. Use separate line styles to show that `b_mp` gates dedicated sensing only, while communication remains globally cooperative.

## Figure 2: Dual DC-SCA and Recovery Trace

From three representative main-experiment seeds, plot rank residual, binary distance `max min(b,1-b)`, and the Top-N support-change indicator against DC iteration. Mark the topology-stability stopping point and the selected validated recovery candidate. State that support stability is an engineering stopping criterion, not a binary feasibility certificate.

## Figure 3: Sensing-Cluster-Size Trade-off

Use `N_req in {2,3,4,5,6}` with 50 common seeds per point. Use three aligned panels:

- physical feasibility rate with 95% Wilson intervals;
- total transmit power among feasible trials, as median with 10th--90th percentile band;
- median end-to-end time with a 90th-percentile marker.

`N_req=6` is the all-AP sensing-authorization reference.

## Figure 4: Communication and Sensing QoS versus Cluster Size

Reuse Figure 3 trials. Plot against `N_req`:

- achieved normalized PCRB, `tr(J_p^{-1})/Gamma_track,p`;
- worst-user robust SINR margin in dB;
- sensing-SINR margin in dB.

Use medians and 10th--90th percentile bands across feasible trials.

## Figure 5: Conditional Power-Gap CDF

For 100 seeds at `N_req=3`, plot the CDF of
\[
100(P_{\rm proposed}-P_{\rm SDR})/P_{\rm SDR}.
\]

Overlay D-optimal FIM-only recovery and full proposed recovery where feasible. Include each method's feasibility rate in the legend. SDR itself is shown only through the normalization, not as a feasible curve.

## Figure 6: Recovery Ablation

Use 30 common seeds at `N_req=3`. Compare D-optimal FIM recovery, DC Top-N recovery, and full proposed recovery with three grouped-bar panels:

- feasibility rate with Wilson intervals;
- median conditional power penalty;
- median time with 90th-percentile whisker.

## Figure 7: Dimension Sensitivity

Use `M in {4,6,8}`, `Nt=2`, `K=3`, `P=2`, `N_req=3`, and 30 common seeds per point. Plot total antenna dimension `N=M*Nt` versus median/90th-percentile runtime and physical feasibility rate. This justifies the `N=12` main operating point.

## Tables

| Table | Content |
| --- | --- |
| I | System, channel, solver, and hardware parameters. |
| II | 100-seed main result: feasibility, power-gap quantiles, time quantiles, DC iterations, and candidate count. |
| III | Failure taxonomy: SDR infeasibility, DC failure, fixed-b infeasibility, and validator failure. |

## Execution Order

1. Run a 10-seed `N_req=3` smoke test.
2. Run 100 main seeds for Figures 2 and 5 and Table II.
3. Run the 250-trial cluster-size sweep for Figures 3 and 4.
4. Run the 30-seed ablation and dimension studies for Figures 6 and 7.
5. Freeze seed lists, solver tolerances, code commit, and machine specification before final plotting.
