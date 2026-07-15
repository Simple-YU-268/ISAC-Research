"""
Baselines for comparison against Algorithm 2 (v10 double-DC SCA):

  B1: Centralized (P3)         - all APs serve every target (no AP selection)
  B2: Heuristic AP selection   - large-scale fading top-N_req per target
  B3: Single-shot SDR + GR     - one (P3) solve, then rank-1 + b-threshold rounding
  B4: Algorithm 2 (proposed)   - v10 double-DC SCA main loop

Each baseline returns the same dict:
  {
    "sum_rate_bps_hz": float,  # sum log2(1+SINR) over K UEs
    "sens_sinr_db":   np.ndarray,  # (P,) per-target sensing SINR in dB
    "pcrb_trace":     np.ndarray,  # (P,) tr(J_p^-1) if J_p well-conditioned, else inf
    "w_mp_active":    np.ndarray,  # (M, P) AP-target assignment (binary after post-proc)
    "obj_trace":      list[float], # objective over SCA iterations (B4 only)
    "status":         str,
    "n_iter":         int,
    "rank_def":       np.ndarray,  # (K,) per-UE rank deficiency at termination
    "binary_def":     float,       # sum g(b) at termination
  }
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cvxpy as cp

from .scenario import E_m  # type: ignore
from .p3_solver import P3Params, P3Solution, solve_p3_sca_t, initial_feasible_solve, _build_E_m


def evaluate(W_list, Z, b_mp, prm: P3Params):
    """Compute sum rate, sensing SINR, PCRB trace from covariance solution.

    Returns (sum_rate_bits_per_hz, sens_sinr_db, pcrb_trace, w_active).
    """
    K, P, N, M = prm.K, prm.P, prm.N, prm.M
    R_X = sum(W_list) + Z

    # --- SINR per UE ---
    sinr = np.zeros(K)
    for k in range(K):
        hk = prm.H[:, k]
        sig = float(np.real(hk.conj().T @ W_list[k] @ hk))
        interf = sum(float(np.real(hk.conj().T @ W_list[j] @ hk)) for j in range(K) if j != k)
        sinr[k] = sig / (interf + prm.sigma_c2)

    sum_rate = float(np.sum(np.log2(1.0 + sinr)))

    # --- sensing SINR per target ---
    sens_sinr = np.zeros(P)
    for p in range(P):
        gp = prm.G[:, p]
        sens_sinr[p] = float(np.real(gp.conj().T @ Z @ gp)) / prm.sigma_s2
    sens_sinr_db = 10 * np.log10(np.maximum(sens_sinr, 1e-12))

    # --- PCRB trace (scalar J_p) ---
    pcrb = np.zeros(P)
    for p in range(P):
        gp = prm.G[:, p]
        J_p = float(np.real(gp.conj().T @ R_X @ gp)) / prm.sigma_s2
        if J_p > 1e-9:
            pcrb[p] = 1.0 / J_p
        else:
            pcrb[p] = np.inf

    # --- AP-target activation ---
    if b_mp is None:
        w_active = np.ones((M, P), dtype=int)
    else:
        w_active = (b_mp >= 0.5).astype(int)

    return sum_rate, sens_sinr_db, pcrb, w_active


# ============================================================
# B4: Algorithm 2 - Double-DC SCA (proposed)
# ============================================================

def baseline_alg2(prm: P3Params, T_max: int = 30, eps: float = 1e-4,
                  eta_rank: float = 1.0, eta_b: float = 1.0,
                  eta_growth: float = 1.2, verbose: bool = False) -> dict:
    """v10 double-DC SCA main loop (Algorithm 2)."""
    K, P, N, M = prm.K, prm.P, prm.N, prm.M

    # --- cold start: heuristic large-scale top-N_req ---
    s = prm  # use prm
    # large-scale UE SNR per AP-UE
    beta_mk = np.zeros((M, K))
    for k in range(K):
        hk = prm.H[:, k]
        for m in range(M):
            blk = hk[m * (N // M):(m + 1) * (N // M)]
            beta_mk[m, k] = float(np.real(blk.conj() @ blk))
    beta_mp = np.zeros((M, P))
    for p in range(P):
        gp = prm.G[:, p]
        for m in range(M):
            blk = gp[m * (N // M):(m + 1) * (N // M)]
            beta_mp[m, p] = float(np.real(blk.conj() @ blk))

    b_prev = np.zeros((M, P))
    for p in range(P):
        # rank APs by their target-side large-scale gain
        order = np.argsort(-beta_mp[:, p])
        b_prev[order[:prm.N_req], p] = 1.0

    # --- warm start with eta_rank = eta_b = 0 ---
    sol0 = initial_feasible_solve(prm)
    if sol0.status not in ("optimal", "optimal_inaccurate"):
        return {"status": "initial_infeasible", "obj_trace": []}
    W_prev = [W.copy() for W in sol0.W]
    Z_prev = sol0.Z.copy() if sol0.Z is not None else np.zeros((N, N), dtype=complex)
    b_prev = np.where(sol0.b >= 0.5, 1.0, 0.0)  # snap to binary for warm start

    obj_trace = [sol0.obj]
    cur_eta_rank = eta_rank
    cur_eta_b = eta_b
    last_status = sol0.status

    for t in range(T_max):
        sol, _ = solve_p3_sca_t(prm, W_prev, b_prev, cur_eta_rank, cur_eta_b)
        if sol.status not in ("optimal", "optimal_inaccurate"):
            if verbose: print(f"  iter {t}: solver status = {sol.status}")
            break
        last_status = sol.status
        Z_sol = sol.Z if sol.Z is not None else np.zeros((N, N), dtype=complex)
        true_obj = sum(float(np.real(np.trace(sol.W[k]))) for k in range(K)) + float(np.real(np.trace(Z_sol)))
        obj_trace.append(true_obj)

        rank_def = sol.rank_deficiency if sol.rank_deficiency is not None else np.full(K, np.inf)
        bin_def = sol.binary_deficiency
        d_obj = abs(obj_trace[-1] - obj_trace[-2])
        if verbose:
            print(f"  iter {t}: true_obj={true_obj:.4f} | rank_def={np.max(rank_def):.2e} "
                  f"| bin_def={bin_def:.2e} | d_obj={d_obj:.2e}")

        W_prev = [W.copy() for W in sol.W]
        Z_prev = sol.Z.copy() if sol.Z is not None else Z_prev
        b_prev = sol.b.copy()

        if (np.max(rank_def) < eps and bin_def < eps and d_obj < eps):
            break

        # ramp eta
        cur_eta_rank = min(cur_eta_rank * eta_growth, 5.0)
        cur_eta_b = min(cur_eta_b * eta_growth, 5.0)

    # --- evaluate final SDR solution (W_prev, Z_prev, b_prev) ---
    b_final = (b_prev >= 0.5).astype(int)
    sum_rate, sens_sinr_db, pcrb, w_active = evaluate(W_prev, Z_prev, b_final, prm)

    return {
        "sum_rate_bps_hz": sum_rate,
        "sens_sinr_db": sens_sinr_db,
        "pcrb_trace": pcrb,
        "w_mp_active": w_active,
        "obj_trace": obj_trace,
        "status": last_status,
        "n_iter": len(obj_trace) - 1,
        "rank_def": np.array([float(np.real(np.trace(W_prev[k])) - np.max(np.linalg.eigvalsh(W_prev[k]).real))
                              for k in range(K)]),
        "binary_def": float(np.sum(np.clip(b_prev, 0, 1) - np.clip(b_prev, 0, 1) ** 2)),
    }


def w_k_as_list(w_k_list, K, N):
    """Helper: convert eigenvector list to W-list of outer products."""
    return [np.real(np.outer(v, v.conj())) for v in w_k_list]


# ============================================================
# B1: Centralized - no AP selection, all APs serve every target
# ============================================================

def baseline_b1_centralized(prm: P3Params, eta_rank: float = 5.0, T_max: int = 20,
                            eps: float = 1e-4, verbose: bool = False) -> dict:
    """Same as B4 but with b_mp = 1 for all (m, p) - no AP selection."""
    # modify prm: set N_req = M (so constraint sum b = N_req = M is satisfied trivially)
    prm_c = P3Params(**{**prm.__dict__, "N_req": prm.M})
    # run B4 logic with fixed b_mp = 1
    K, P, N, M = prm_c.K, prm_c.P, prm_c.N, prm_c.M

    # warm start
    sol0 = initial_feasible_solve(prm_c)
    if sol0.status not in ("optimal", "optimal_inaccurate"):
        return {"status": "initial_infeasible"}
    W_prev = [W.copy() for W in sol0.W]
    Z_prev = sol0.Z.copy() if sol0.Z is not None else np.zeros((N, N), dtype=complex)
    b_prev = np.ones((M, P))

    obj_trace = [sol0.obj]
    for t in range(T_max):
        sol, _ = solve_p3_sca_t(prm_c, W_prev, b_prev, eta_rank, 0.0)
        if sol.status not in ("optimal", "optimal_inaccurate"):
            break
        obj_trace.append(sol.obj)
        W_prev = [W.copy() for W in sol.W]
        Z_prev = sol.Z.copy() if sol.Z is not None else Z_prev
        if abs(obj_trace[-1] - obj_trace[-2]) < eps and np.max(sol.rank_deficiency) < eps:
            break
        b_prev = np.ones((M, P))
        eta_rank = min(eta_rank * 1.5, 1e3)

    sum_rate, sens_sinr_db, pcrb, w_active = evaluate(W_prev, Z_prev, np.ones((M, P)), prm)
    return {"sum_rate_bps_hz": sum_rate, "sens_sinr_db": sens_sinr_db,
            "pcrb_trace": pcrb, "w_mp_active": w_active,
            "obj_trace": obj_trace, "status": sol.status, "n_iter": len(obj_trace) - 1,
            "rank_def": np.array([float(np.real(np.trace(W_prev[k])) - np.max(np.linalg.eigvalsh(W_prev[k]).real))
                                  for k in range(K)]),
            "binary_def": 0.0}


# ============================================================
# B2: Heuristic AP selection - fix b_mp via large-scale fading
# ============================================================

def baseline_b2_heuristic(prm: P3Params, eta_rank: float = 5.0, T_max: int = 20,
                          eps: float = 1e-4, verbose: bool = False) -> dict:
    """Heuristic v5: fix b_mp via top-N_req large-scale, then solve W only."""
    K, P, N, M = prm.K, prm.P, prm.N, prm.M
    beta_mp = np.zeros((M, P))
    for p in range(P):
        gp = prm.G[:, p]
        for m in range(M):
            Nt = N // M
            blk = gp[m * Nt:(m + 1) * Nt]
            beta_mp[m, p] = float(np.real(blk.conj() @ blk))

    b_fix = np.zeros((M, P))
    for p in range(P):
        order = np.argsort(-beta_mp[:, p])
        b_fix[order[:prm.N_req], p] = 1.0

    # warm start
    sol0 = solve_p3_sca_t(prm, [np.eye(N) * (prm.Pmax / prm.K / prm.M) for _ in range(K)],
                           b_fix, 0.0, 0.0)[0]
    if sol0.status not in ("optimal", "optimal_inaccurate"):
        return {"status": "initial_infeasible"}
    W_prev = [W.copy() for W in sol0.W]
    Z_prev = sol0.Z.copy() if sol0.Z is not None else np.zeros((N, N), dtype=complex)
    obj_trace = [sol0.obj]
    for t in range(T_max):
        sol, _ = solve_p3_sca_t(prm, W_prev, b_fix, eta_rank, 0.0)
        if sol.status not in ("optimal", "optimal_inaccurate"):
            break
        obj_trace.append(sol.obj)
        W_prev = [W.copy() for W in sol.W]
        Z_prev = sol.Z.copy() if sol.Z is not None else Z_prev
        if abs(obj_trace[-1] - obj_trace[-2]) < eps and np.max(sol.rank_deficiency) < eps:
            break
        eta_rank = min(eta_rank * 1.5, 1e3)

    sum_rate, sens_sinr_db, pcrb, w_active = evaluate(W_prev, Z_prev, b_fix, prm)
    return {"sum_rate_bps_hz": sum_rate, "sens_sinr_db": sens_sinr_db,
            "pcrb_trace": pcrb, "w_mp_active": w_active,
            "obj_trace": obj_trace, "status": sol.status, "n_iter": len(obj_trace) - 1,
            "rank_def": np.array([float(np.real(np.trace(W_prev[k])) - np.max(np.linalg.eigvalsh(W_prev[k]).real))
                                  for k in range(K)]),
            "binary_def": 0.0}


# ============================================================
# B3: Single-shot SDR + GR post-process
# ============================================================

def baseline_b3_gr(prm: P3Params, n_samples: int = 50, eta_rank: float = 5.0,
                   eps: float = 1e-4, verbose: bool = False) -> dict:
    """Single (P3) solve with rank penalty only, then Gaussian randomization
    to extract rank-1 solutions + threshold round for b."""
    K, P, N, M = prm.K, prm.P, prm.N, prm.M

    # 1) Solve SDR with small rank penalty
    sol0 = solve_p3_sca_t(prm,
                           [np.eye(N) * (prm.Pmax / prm.K / prm.M) for _ in range(K)],
                           np.full((M, P), prm.N_req / prm.M), eta_rank, 0.0)[0]
    if sol0.status not in ("optimal", "optimal_inaccurate"):
        return {"status": "initial_infeasible"}

    # 2) Eigendecompose + iterate rank penalty
    W_prev = [W.copy() for W in sol0.W]
    Z_sdr = sol0.Z.copy() if sol0.Z is not None else np.zeros((N, N), dtype=complex)
    for _ in range(15):
        sol, _ = solve_p3_sca_t(prm, W_prev, np.full((M, P), prm.N_req / prm.M), eta_rank, 0.0)
        if sol.status not in ("optimal", "optimal_inaccurate"):
            break
        if np.max(sol.rank_deficiency) < eps:
            W_prev = [W.copy() for W in sol.W]
            Z_sdr = sol.Z.copy() if sol.Z is not None else Z_sdr
            break
        W_prev = [W.copy() for W in sol.W]
        Z_sdr = sol.Z.copy() if sol.Z is not None else Z_sdr
        eta_rank = min(eta_rank * 1.5, 1e3)

    # 3) Gaussian randomization: sample n_samples candidate W
    best_obj = float("inf")
    best_W = None
    best_Z = None
    best_b = None
    for s_idx in range(n_samples):
        rng = np.random.default_rng(s_idx + 1000)
        cand_W = []
        for k in range(K):
            # sample xi ~ CN(0, W_k), scale by sqrt
            w, V = np.linalg.eigh(W_prev[k])
            w = np.maximum(w, 0)
            xi = (rng.normal(size=N) + 1j * rng.normal(size=N)) / np.sqrt(2)
            cand = V @ np.diag(np.sqrt(w)) @ xi
            cand_W.append(np.real(np.outer(cand, cand.conj())))

        # b_mp: threshold round of sol0.b
        cand_b = np.where(sol0.b >= 0.5, 1.0, 0.0)
        # Adjust b_mp to satisfy sum b_mp = N_req
        for p in range(P):
            if cand_b[:, p].sum() != prm.N_req:
                # pick top-N_req from large-scale
                beta_mp_p = np.array([float(np.real(prm.G[m * (N // M):(m + 1) * (N // M), p].conj()
                                                @ prm.G[m * (N // M):(m + 1) * (N // M), p]))
                                       for m in range(M)])
                order = np.argsort(-beta_mp_p)
                cand_b[:, p] = 0.0
                cand_b[order[:prm.N_req], p] = 1.0

        # objective: sum tr(W_k) + tr(Z) (using SDR Z for sensing contribution)
        obj_val = sum(float(np.real(np.trace(W))) for W in cand_W) + float(np.real(np.trace(Z_sdr)))
        if obj_val < best_obj:
            best_obj = obj_val
            best_W = cand_W
            best_Z = Z_sdr
            best_b = cand_b

    sum_rate, sens_sinr_db, pcrb, w_active = evaluate(best_W, best_Z, best_b, prm)
    return {"sum_rate_bps_hz": sum_rate, "sens_sinr_db": sens_sinr_db,
            "pcrb_trace": pcrb, "w_mp_active": w_active,
            "obj_trace": [best_obj], "status": "ok", "n_iter": 1,
            "rank_def": np.zeros(K),
            "binary_def": 0.0}