# ISAC MATLAB Simulation

This folder contains a MATLAB re-implementation of the (P3) convex SCA solver.
It mirrors the Python `p3_solver.py` but uses CVX/SDPT3 or MOSEK instead of CVXPY/Clarabel,
which is more stable for the double-DC SCA iterations.

## Files

- `default_params.m` — generates a small feasible scenario (M=6 APs, Nt=4, K=3 UEs, P=3 targets).
- `solve_p3_sca_t.m` — single (P3-SCA-t) subproblem (CVX). The PCRB constraint uses the exact matrix Schur LMI from §II-D: `M_p` is now an `N_theta x N_theta x P` Hermitian matrix and `trace(M_p(:,:,p)) <= Gamma_track(p)`.
- `baseline_alg2.m` — Algorithm 2 warm start + SCA loop + binary rounding + fixed-b re-solve + physical beam extraction.
- `sanity_check.m` — runs one instance and prints metrics.
- `validate_solution.m` — standalone feasibility checker (S-Procedure-aware).
- `evaluate.m` — physical beam extraction + sum-rate/PCRB evaluation.
- `debug_pcrb_feasibility.m` — diagnostic warm-start feasibility check.

## Requirements

- MATLAB (R2020b or later recommended)
- CVX 2.2 or later: http://cvxr.com/cvx/
- **MOSEK** 11.2+ (strongly recommended for `N_theta >= 2`)
- SDPT3 or SeDuMi as fallback for `N_theta = 1`

## Installing MOSEK for CVX

1. Download the macOS ARM64 default installer from https://www.mosek.com/downloads (e.g. MOSEK 11.2.2).
2. Extract to a known path, e.g. `/Volumes/Mac_mini_ssd/software/mosek/mosek/11.2`.
3. Place your `mosek.lic` in `.../tools/platform/osxaarch64/bin/`.
4. Add MOSEK to the MATLAB path and point the environment variable to the license:
   ```matlab
   addpath('/Volumes/Mac_mini_ssd/software/mosek/mosek/11.2/toolbox/r2022b');
   setenv('MOSEKLM_LICENSE_FILE', '/Volumes/Mac_mini_ssd/software/mosek/mosek/11.2/tools/platform/osxaarch64/bin/mosek.lic');
   cvx_setup;
   cvx_solver mosek;
   ```
   CVX should then list `Mosek 11.2.2` in `cvx_solver`.

## Run

```matlab
sanity_check
```

## Notes

- `default_params.m` now sets `N_theta = 2` (2D target position) by default.
  For `N_theta = 1`, `M_p` is stored as a `1 x 1 x P` array for a uniform interface.
  For `N_theta >= 2`, the exact Schur LMI `[M_p(:,:,p) I; I J_p] >= 0` with `trace(M_p(:,:,p)) <= Gamma_track(p)` is used; **MOSEK is required** for stable convergence.
- The robust S-Procedure is enabled by default (`use_s_procedure = true`).
  The LMI lower-right entry uses `eps_h^2 * ||hk||^2` so that the uncertainty-ball radius is correctly scaled for non-normalized channels.
- Physical beams are extracted via the principal eigenvector of each converged `W_k` and the SINR/PCRB metrics are evaluated using the rank-1 vectors.
- Per-UE / per-target channel scaling is applied in `default_params.m` so that `Pmax = 0.1 W` (20 dBm) and `gamma = 1` (0 dB) are jointly feasible.
- `active_targets` controls which targets must satisfy the service-count equality (`sum(b(:,p)) == N_req`). By default all targets are active.
