"""
(P3) SDP subproblem solver + dual-DC SCA main loop.

Implements:
  - Algorithm 2 from math_derivation.tex:
      Double DC penalty SCA (rank-1 + binary joint recovery)
  - (P3-C1)..(P3-C10) convex SDP via CVXPY + Clarabel

Conventions:
  - Decision vars: W_k (K Hermitian N x N), Z (Hermitian N x N),
                   mu_k (K reals), M_p (P Hermitian P x P), b_mp (M x P reals)
  - R_X = sum_k W_k + Z (stacked covariance)
  - E_m: per-AP extraction (N x N diagonal 0/1)
  - Channels: H (N x K), G (N x P)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cvxpy as cp


@dataclass
class P3Params:
    """All physical parameters for the (P3) problem."""
    H: np.ndarray              # (N, K)
    G: np.ndarray              # (N, P)
    eps_h: float               # CSI error bound (relative)
    gamma_k: np.ndarray        # (K,) target SINR for UEs (linear, e.g. 1.0 = 0 dB)
    gamma_PoD: np.ndarray      # (P,) sensing SINR threshold (linear)
    Gamma_track: np.ndarray    # (P,) PCRB trace threshold
    sigma_c2: float            # comm noise power
    sigma_s2: float            # sensing noise power
    Pmax: float                # per-AP power budget
    N_req: int                 # #APs to serve each active target
    N: int                     # stacked antenna count
    M: int                     # AP count
    K: int                     # UE count
    P: int                     # target count
    use_s_procedure: bool = True   # set False to skip S-Procedure (deterministic SINR)


@dataclass
class P3Solution:
    """Result from a single solve."""
    W: list[np.ndarray] = field(default_factory=list)         # K matrices
    Z: np.ndarray | None = None                               # (N, N)
    mu: np.ndarray | None = None                              # (K,)
    M_p: list[np.ndarray] = field(default_factory=list)       # P matrices
    b: np.ndarray | None = None                               # (M, P)
    status: str = "unknown"
    obj: float = float("inf")
    rank_deficiency: np.ndarray | None = None                 # (K,) tr(W)-lam_max
    binary_deficiency: float = 0.0                            # sum g(b)


def _build_E_m(M: int, Nt: int) -> list[np.ndarray]:
    """Per-AP extraction matrices (M of them, each N x N)."""
    N = M * Nt
    out = []
    for m in range(M):
        E = np.zeros((N, N))
        E[m * Nt:(m + 1) * Nt, m * Nt:(m + 1) * Nt] = np.eye(Nt)
        out.append(E)
    return out


def _eigvec_main(W: np.ndarray) -> np.ndarray:
    """Return dominant eigenvector of Hermitian PSD matrix W."""
    # W is real-symmetric or Hermitian PSD; use eigh
    w, V = np.linalg.eigh(W)
    return V[:, -1]   # eigh returns ascending order


def _fim_data(G: np.ndarray, R_X: np.ndarray, sigma_s2: float) -> np.ndarray:
    """J_p^data = g_p^H R_X g_p / sigma_s2 (scalar FIM per target, N_theta=1)."""
    P = G.shape[1]
    J_diag = np.zeros(P)
    for p in range(P):
        gp = G[:, p]
        J_diag[p] = float((gp.conj().T @ R_X @ gp).real / sigma_s2)
    return J_diag


def solve_p3_sca_t(prm: P3Params,
                    W_prev: list[np.ndarray],
                    b_prev: np.ndarray,
                    eta_rank: float,
                    eta_b: float) -> tuple[P3Solution, np.ndarray]:
    """Solve the (P3-SCA-t) subproblem given previous-iteration W_prev, b_prev.

    Returns (solution, c_mp) where c_mp = (1 - 2 b_prev) are the Taylor
    coefficients for the linearization of g(b) at b_prev.
    """
    K, P, N, M = prm.K, prm.P, prm.N, prm.M
    E_list = _build_E_m(M, N // M)

    # Decision variables
    W_vars = [cp.Variable((N, N), hermitian=True) for _ in range(K)]
    Z_var = cp.Variable((N, N), hermitian=True)
    mu_vars = [cp.Variable(nonneg=True) for _ in range(K)]
    # M_p is scalar auxiliary for PCRB trace per target (N_theta = 1).
    # Schur: [M_p 1; 1 J_p^data] >= 0  <=>  M_p >= 1 / J_p^data.
    M_p_vars = [cp.Variable(nonneg=True) for _ in range(P)]
    b_var = cp.Variable((M, P), nonneg=True)

    # Stacked covariance
    R_X = sum(W_vars) + Z_var

    constraints = []

    # (P3-C1) SINR constraint: gamma_k * (sum_{j!=k} h_k^H W_j h_k + sigma_c2) <= h_k^H W_k h_k
    # With S-Procedure uncertainty, use the v10 LMI form; otherwise use the
    # deterministic quadratic form (still convex SDP via lifting).
    for k in range(K):
        hk = prm.H[:, k]
        if prm.use_s_procedure:
            Ak = (1.0 / prm.gamma_k[k]) * W_vars[k] - sum(W_vars[j] for j in range(K) if j != k)
            top_left = Ak + mu_vars[k] * np.eye(N)
            top_right = Ak @ hk
            bot_left = hk.conj().T @ Ak
            bot_right = cp.real(hk.conj().T @ Ak @ hk) - prm.sigma_c2 - mu_vars[k] * prm.eps_h ** 2
            LMI = cp.bmat([[top_left, cp.reshape(top_right, (N, 1), order='C')],
                           [cp.reshape(bot_left, (1, N), order='C'), cp.reshape(bot_right, (1, 1), order='C')]])
            constraints.append(LMI >> 0)
        else:
            # Deterministic SINR (no robustness): gamma_k * (interf + sigma_c2) <= sig
            sig = cp.real(hk.conj().T @ W_vars[k] @ hk)
            interf = sum(cp.real(hk.conj().T @ W_vars[j] @ hk) for j in range(K) if j != k)
            constraints.append(prm.gamma_k[k] * (interf + prm.sigma_c2) <= sig)

    # (P3-C2) Sensing SINR per target p: tr(g_p g_p^H Z) >= gamma_PoD * sigma_s2
    for p in range(P):
        gp = prm.G[:, p]
        constraints.append(cp.real(gp.conj().T @ Z_var @ gp) >= prm.gamma_PoD[p] * prm.sigma_s2)

    # (P3-C3) PCRB scalar: M_p >= 1 / J_p^data
    # (P3-C4) M_p <= Gamma_track,p
    # J_p^data = g_p^H R_X g_p / sigma_s^2 (scalar FIM for one parameter per target)
    for p in range(P):
        gp = prm.G[:, p]
        J_p = cp.real(gp.conj().T @ R_X @ gp) / prm.sigma_s2
        constraints.append(cp.inv_pos(J_p) <= M_p_vars[p])
        constraints.append(M_p_vars[p] <= prm.Gamma_track[p])

    # (P3-C5'a) per-AP power with Sum Big-M gate: tr(E_m R_X) <= Pmax * sum_p b_mp
    for m in range(M):
        constraints.append(cp.real(cp.trace(E_list[m] @ R_X))
                           <= prm.Pmax * cp.sum(b_var[m, :]))
    # (P3-C5'b) hard ceiling: tr(E_m R_X) <= Pmax (always)
    for m in range(M):
        constraints.append(cp.real(cp.trace(E_list[m] @ R_X)) <= prm.Pmax)

    # (P3-C6) service count: sum_m b_mp = N_req, for active targets
    # All targets active in our setting
    for p in range(P):
        constraints.append(cp.sum(b_var[:, p]) == prm.N_req)

    # (P3-C7) W_k PSD
    for k in range(K):
        constraints.append(W_vars[k] >> 0)
    # (P3-C8) Z PSD
    constraints.append(Z_var >> 0)
    # (P3-C9) mu_k >= 0 (already via nonneg=True)
    # (P3-C10) box: 0 <= b_mp <= 1
    constraints.append(b_var <= 1.0)

    # ---------- objective: linearized DC penalty ----------
    # Original: sum tr(W_k) + tr(Z)
    # + eta_rank * sum_k [tr(W_k) - tr(u_max,k u_max,k^H W_k)]
    # + eta_b * sum (1 - 2 b_mp_prev) b_mp
    # DC_rank(W_k) = tr(W_k) - lambda_max(W_k)
    # Linearization: tr(W_k) - v_k^H W_k v_k = tr((I - v_k v_k^H) W_k)
    # Therefore objective = (1 + eta_rank) * sum tr(W_k) + tr(Z)
    #                     + eta_rank * sum tr(-v_k v_k^H W_k)
    #                     + eta_b * sum c_mp b_mp
    main_obj = (1 + eta_rank) * sum(cp.real(cp.trace(W_vars[k])) for k in range(K)) + cp.real(cp.trace(Z_var))

    rank_penalty = 0.0
    for k in range(K):
        # eigvec of W_prev[k]
        u_max_k = _eigvec_main(W_prev[k])
        rank_penalty = rank_penalty + cp.real(cp.trace(-np.outer(u_max_k, u_max_k.conj()) @ W_vars[k]))

    c_mp = 1.0 - 2.0 * b_prev     # (M, P) Taylor coefficient
    # DC surrogate: g(b) = b - b^2, linearized as (1 - 2 b_prev) b + b_prev^2
    # Include constant so the reported objective is the true surrogate and nonnegative.
    b_penalty = eta_b * (cp.sum(cp.multiply(c_mp, b_var)) + np.sum(b_prev ** 2))

    obj = cp.Minimize(main_obj + eta_rank * rank_penalty + b_penalty)
    prob = cp.Problem(obj, constraints)

    # Solve
    try:
        prob.solve(solver=cp.CLARABEL, verbose=True)

        status = prob.status
        obj_val = float(prob.value) if prob.value is not None else float("inf")
    except cp.SolverError as e:
        status = f"solver_error: {e}"
        obj_val = float("inf")
        print(f"[DEBUG] Clarabel exception: {e}")

    sol = P3Solution(status=status, obj=obj_val)

    if status in ("optimal", "optimal_inaccurate") and obj_val < float("inf"):
        sol.W = [np.asarray(W_vars[k].value) for k in range(K)]
        sol.Z = np.asarray(Z_var.value)
        sol.mu = np.array([mu_vars[k].value for k in range(K)])
        sol.M_p = [np.asarray(M_p_vars[p].value) for p in range(P)]
        sol.b = np.asarray(b_var.value)
        # rank deficiency: tr(W) - lam_max(W)
        sol.rank_deficiency = np.array([
            float(np.real(np.trace(sol.W[k])) - np.max(np.linalg.eigvalsh(sol.W[k]).real))
            for k in range(K)
        ])
        # binary deficiency: sum g(b)
        b_clip = np.clip(sol.b, 0.0, 1.0)
        sol.binary_deficiency = float(np.sum(b_clip - b_clip ** 2))
    return sol, c_mp


def initial_feasible_solve(prm: P3Params) -> P3Solution:
    """Solve (P3) with eta_rank = eta_b = 0 to get an SDR warm-start.
    Returns the unconstrained SDR solution.
    """
    return solve_p3_sca_t(prm,
                          W_prev=[np.eye(prm.N) * (prm.Pmax / prm.K / prm.M) for _ in range(prm.K)],
                          b_prev=np.full((prm.M, prm.P), prm.N_req / prm.M),
                          eta_rank=0.0,
                          eta_b=0.0)[0]