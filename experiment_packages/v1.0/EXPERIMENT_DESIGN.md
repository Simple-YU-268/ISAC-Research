# Experiment Design: Oracle-Geometric Baseline

## Method definitions

The main study compares three AP--target association methods under identical
channel realizations, power budgets, robust SINR constraints, PCRB constraints,
and fixed-assignment covariance re-optimization.

| Label | Association information | Role |
|---|---|---|
| Proposed | Information permitted by the optimization model | Primary algorithm |
| Oracle nearest AP | Simulated true target position | Theoretical geometric reference only |
| Random AP | Uniform random AP subset with the same cardinality | Non-geometric baseline |

The oracle method must never be described as an online-deployable policy. It
answers only how close a distance-perfect geometric rule is to the proposed
association under the simulated model.

## Main paired Monte Carlo experiment

- Shared seeds and channel realizations for all methods.
- `N_req = 1:6`, `N_mc = 100`, and robust CSI uncertainty `eps_h = 0.05`.
- Report physical feasibility first. Report power, rate, PCRB, and sensing SINR
  conditionally on physical feasibility. Use paired statistics only when both
  methods are feasible in the same realization.
- Record wall-clock time and number of fixed-assignment recovery solves for the
  proposed method. Do not compare raw run time with the oracle method as an
  optimality measure; report it as computational cost.

## Recovery-complexity ablation

For the proposed method only, test candidate budgets `1, 3, 5, 8, 21` on a
small common seed set. The proposed candidate set must not include the oracle
nearest-AP assignment. Plot feasibility rate and median runtime against the
candidate budget. This identifies the recovery cost needed by the proposed
method itself.

## UE--target proximity stress case

Use the manually specified geometry with UE 1 located 5 m from Target 1.
Compare Proposed, Oracle nearest AP, and Random using the same true geometry.
The case tests whether robust communication and dedicated sensing interference
make a geometry-perfect sensing association infeasible or suboptimal. It is a
diagnostic stress case, not evidence that the oracle method is implementable.

## Optional future practical baseline

A prediction-based nearest-AP policy may be added only after defining a target
state prediction and its error distribution. It must be reported separately
from the oracle geometric benchmark.
