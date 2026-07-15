# ISAC MATLAB Simulation

This folder contains a MATLAB re-implementation of the (P3) convex SCA solver.
It mirrors the Python `p3_solver.py` but uses CVX/SDPT3 (or MOSEK) instead of CVXPY/Clarabel,
which should be more stable for the double-DC SCA iterations.

## Files

- `default_params.m` — generates a small feasible scenario (M=6 APs, Nt=4, K=3 UEs, P=3 targets).
- `solve_p3_sca_t.m` — single (P3-SCA-t) subproblem (CVX).
- `baseline_alg2.m` — Algorithm 2 warm start + SCA loop + binary rounding + fixed-b re-solve.
- `sanity_check.m` — runs one instance and prints metrics.

## Requirements

- MATLAB (R2020b or later recommended)
- CVX 2.2 or later: http://cvxr.com/cvx/
- A SDP solver: **MOSEK** (preferred) or **SDPT3** / **SeDuMi**

## Run

```matlab
sanity_check
```

## Notes

- `solve_p3_sca_t.m` uses `inv_pos` for the scalar PCRB constraint `M_p >= 1/J_p`,
  avoiding the ill-conditioned 2x2 Schur LMI in CVXPY/Clarabel.
- The SINR constraint is deterministic (`use_s_procedure = false`).
  To enable robust S-Procedure, set `use_s_procedure = true` and make sure `eps_h` is set.
- Per-UE / per-target channel scaling is applied in `default_params.m` so that
  `Pmax = 0.1 W` (20 dBm) and `gamma = 1` (0 dB) are jointly feasible.
