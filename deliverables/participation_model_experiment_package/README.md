# Participation-Constrained Cell-Free ISAC: Experiment Package

## Scope

This package contains the reproducible MATLAB experiments, validated raw results,
paper figures, and updated numerical-results text for the current model:

\[
b_{mp}\in\{0,1\},\quad
\sum_m b_{mp}=N_{\rm req},\quad
P_{\min}^{\rm sen}b_{mp}\leq\operatorname{tr}(\mathbf E_m\mathbf S_p)
\leq P_{\max}b_{mp}.
\]

The communication covariances remain globally cooperative.  Only dedicated
sensing covariance \(\mathbf S_p\) is gated by the AP--target participation
indicator \(b_{mp}\).  All reported physical solutions were fixed-topology
re-optimized and passed the feasibility validator.

## Reproduction environment

- MATLAB R2024b, CVX, MOSEK 11.2.2
- Default main configuration: \(M=6\), \(N_t=2\), \(K=3\), \(P=2\),
  \(N_\theta=2\), \(N_{\rm req}=3\), \(P_{\max}=0.1\) W,
  \(\epsilon_h=0.05\), and \(P_{\min}^{\rm sen}=0.01P_{\max}=1\) mW.

## Validated findings

| Study | Evidence |
|---|---|
| Binary DC mechanism | Over 30 common seeds, the median binary residual falls from \(4.268\times10^{-1}\) without the binary penalty to \(6.229\times10^{-5}\) with dual DC. |
| Association quality | In the final common 30-seed comparison at \(N_{\rm req}=3\), the FIM-greedy topology is feasible in 30/30 trials and has 21.27% lower mean power than the nearest-AP baseline. Random association is feasible in 10/30 trials; on the 10 jointly feasible trials FIM reduces mean power by 73.30%. |
| Robust CSI | For \(\epsilon_h=0.02,0.05,0.08\), the S-procedure design has zero sampled outage across 30 seeds and 100 perturbations per seed. Nominal-design mean outage is 89.0%, 89.6%, and 91.9%, respectively. |
| Cluster-size trade-off | All 30 seeds are feasible for \(N_{\rm req}=2,\ldots,6\). Mean power is 37.65, 30.83, 29.50, 29.74, and 30.94 mW, respectively; \(N_{\rm req}=4\) is the observed energy sweet spot. |
| Participation floor | At positive floors 0.5%, 1%, 2%, and 5% of \(P_{\max}\), all 30 seeds satisfy the lower bound with zero measured violation. Mean power is 30.83 mW at 1% and 37.11 mW at 5%. |
| QoS trade-off | A 5-seed, 5-by-5 grid is feasible at all 25 QoS points. Power rises with the robust communication-SINR target and with stricter PCRB requirements. |
| Scaling | For \(M=4,6,8\) (total transmit dimensions \(N=8,12,16\)), all 10 seeds are feasible; median end-to-end times are 4.75, 6.10, and 9.27 s. |
| Multi-method \(N_{\rm req}\) comparison | A 30-seed common-scenario comparison confirms 30/30 feasibility of the proposed and FIM-greedy methods for every \(N_{\rm req}\). At \(N_{\rm req}=3\), mean power is 30.83 mW (proposed), 30.93 mW (FIM), and 39.29 mW (nearest AP); random topology is feasible in only 10/30 trials and has conditional mean power 115.82 mW. |

## Package layout

- `matlab/`: experiment and plotting scripts.
- `results/`: raw MATLAB `.mat` outputs from the current participation model.
- `figures/`: regenerated figures used by the paper.
- `paper/`: numerical-results TeX source and compiled English manuscript PDF.
- `results/nreq_method_performance_30seeds/`: final common-seed multi-method
  comparison result; its figure and CSV are in `figures/`.

## Running the main scripts

From the repository root, add `sim/matlab` to the MATLAB path and call the
scripts in `matlab/`. Each script accepts an `Output_dir` parameter and writes
checkpoint files, enabling interrupted experiments to be resumed.

The raw files included here are the authoritative source for the table above.
Do not mix them with legacy zero-lower-bound results elsewhere in the repository.
