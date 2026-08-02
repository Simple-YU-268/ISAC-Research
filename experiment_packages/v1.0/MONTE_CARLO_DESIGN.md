# Monte-Carlo Experiment Design

## Objective

Evaluate the feasibility, power efficiency, and computational cost of the
physics-certified recovery algorithm under independently generated Cell-Free
ISAC scenarios. A trial is successful only when the final fixed-assignment
solution passes the physical validator.

## Main Configuration

| Parameter | Value |
| --- | ---: |
| APs, `M` | 6 |
| Antennas per AP, `Nt` | 2 |
| Total antennas, `N` | 12 |
| UEs, `K` | 3 |
| Targets, `P` | 2 |
| Target-state dimension, `N_theta` | 2 |
| Required sensing APs per target, `N_req` | 3 |
| Area | 400 m x 400 m |
| Per-AP power cap | 20 dBm (0.1 W) |
| CSI uncertainty radius | 0.05 |
| Communication and sensing SINR thresholds | 0 dB |
| PCRB threshold | auto-calibrated, `Gamma_alpha = 3` |

The main algorithm uses the mathematically defined sequence SDR initialization,
dual DC-SCA, binary-candidate recovery, fixed-b re-optimization, and physical
validation. Topology-stability stopping ends only the continuous DC phase; it
does not itself certify a binary solution.

## Experiment E1: Main Feasibility and Power-Gap Study

- Seeds: `1:100`.
- Methods:
  1. SDR lower bound: continuous `b` and relaxed rank constraints, with no DC
     penalties. It is a lower bound only, not a feasible competing method.
  2. Proposed certified recovery: D-optimal FIM candidate, DC Top-N candidate,
     and PCRB-slack-guided repair when needed.
- For every seed, retain the scenario and the status of each method.

### Primary Metrics

1. Physical feasibility rate:
   `number of validated proposed solutions / number of SDR-feasible scenarios`.
   Report a 95% Wilson confidence interval.
2. Conditional power penalty, evaluated only for trials feasible for both
   methods:
   `(P_proposed - P_SDR) / P_SDR * 100%`.
   Report mean, median, 10th/90th percentiles, and empirical CDF.
3. End-to-end wall-clock time, DC iterations, and number of fixed-b candidates.
   Report median and 90th percentile rather than only the mean.

### Required Diagnostics

- Failure categories: SDR infeasible, DC failure, fixed-b infeasible, and
  validator failure.
- Worst PCRB/SINR violation for failed candidates when slack diagnosis runs.
- Final power, communication SINR margins, sensing SINR margins, and PCRB
  margins for validated trials.

## Experiment E2: Sensing-Cluster-Size Trade-off

- Use `N_req in {2, 3, 4, 5, 6}` with 50 common random seeds per value.
- Keep `M=6, Nt=2, K=3, P=2` unchanged.
- Report feasibility rate, total power, PCRB, sensing SINR, communication SINR,
  and run time against `N_req`.

This experiment identifies the physical price of limiting sensing-cluster
size. Each `N_req` value must regenerate `Gamma_track='auto'` under the same
scenario generator; the reported threshold is recorded per seed.
The endpoint `N_req=6` authorizes all APs for every target and therefore
serves as the no-cluster-sparsity reference.

## Experiment E3: Recovery Ablation

- Use 30 common seeds under the main configuration.
- Compare:
  1. DC Top-N plus fixed-b validation only;
  2. D-optimal FIM candidate plus fixed-b validation only;
  3. Full certified recovery (proposed).
- Report feasibility, power penalty, candidate count, and time.

The oracle true-position nearest-AP rule may be shown only as a theoretical
geometry reference, never as an online deployable baseline.

## Experiment E4: Dimension Sensitivity

- Use `M in {4, 6, 8}`, `Nt=2`, `K=3`, `P=2`, and `N_req=3`.
- Use 30 common seeds per point.
- Report SDP wall-clock time and physical feasibility. This separates
algorithmic behavior from the expected growth of interior-point cost with
total antenna dimension `N=M*Nt`.

## Execution Protocol

1. Use deterministic seed lists and save a checkpoint after every trial.
2. Do not pool infeasible trials into conditional power statistics.
3. Use identical scenario geometry for all methods within a seed.
4. Record MATLAB, CVX, and MOSEK versions, machine name, CPU, RAM, and solver
   tolerances alongside each result file.
5. Run E1 first. Continue to E2--E4 only after E1's 10-seed smoke run confirms
   checkpointing and physical validation.
