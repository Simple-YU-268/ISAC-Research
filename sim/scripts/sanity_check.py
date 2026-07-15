"""
Single-point sanity check: M=6, K=3, P=3, Nt=4, Pmax=20 dBm.
Run all 4 baselines and print comparison.
"""

from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from isac_sim.scenario import Scenario, make_positions, large_scale, small_scale
from isac_sim.p3_solver import P3Params, initial_feasible_solve, solve_p3_sca_t
from isac_sim.baselines import (
    baseline_alg2, baseline_b1_centralized, baseline_b2_heuristic, baseline_b3_gr
)


def build_scenario(M=6, K=3, P=3, Nt=4, Pmax_dbm=20.0, seed=42,
                   use_s_procedure=False,
                   gamma_k_db=0.0, gamma_PoD_db=0.0, Gamma_track=10.0,
                   sigma_s2=None):
    """Build a scenario at scale B2.

    SNR calibration: typical cell-free ISAC papers normalize sigma_s2 so that
    sensing SINR targets (gamma_PoD) are achievable. Here we set sigma_s2 such
    that a single-AP Z = Pmax * I gives sensing SINR ~ gamma_PoD when |g|^2 ~ 1.
    """
    s = Scenario(M=M, K=K, P=P, Nt=Nt, seed=seed)
    s = make_positions(s)
    beta_mk, beta_mp = large_scale(s)
    H, G = small_scale(s, beta_mk, beta_mp)
    Pmax = 10 ** ((Pmax_dbm - 30) / 10)  # Watts

    # Default: normalized so sigma_s2 = sigma_c2 = 1, channels scaled so that
    # a single-AP full-power allocation gives ~10 dB comm SNR.
    if sigma_s2 is None:
        sigma_s2 = 1.0

    # Channel scaling: per-UE / per-target scaling so that each UE/target
    # can reach the same target SNR from its best AP with full Pmax.
    # This makes gamma=0 dB and Gamma_track feasible for all users/targets
    # while preserving the original channel direction and relative AP ratios.
    def per_ap_block_norm_sq(Mat, M, Nt):
        n = Mat.shape[1]
        out = np.zeros((M, n))
        for m in range(M):
            blk = Mat[m*Nt:(m+1)*Nt, :]
            out[m, :] = np.sum(np.abs(blk) ** 2, axis=0)
        return out

    h_block = per_ap_block_norm_sq(H, M, Nt)
    g_block = per_ap_block_norm_sq(G, M, Nt)
    h_best = np.max(h_block, axis=0)          # (K,)
    g_best = np.max(g_block, axis=0)          # (P,)
    target_snr = 100.0                        # 20 dB margin above gamma=0 dB
    h_scale = np.sqrt(target_snr * sigma_s2 / Pmax / np.maximum(h_best, 1e-12))
    g_scale = np.sqrt(target_snr * sigma_s2 / Pmax / np.maximum(g_best, 1e-12))
    H = H * h_scale[np.newaxis, :]
    G = G * g_scale[np.newaxis, :]

    prm = P3Params(
        H=H, G=G,
        eps_h=0.1,
        gamma_k=np.full(K, 10 ** (gamma_k_db / 10)),       # target SINR for UEs
        gamma_PoD=np.full(P, 10 ** (gamma_PoD_db / 10)),   # sensing SINR threshold
        Gamma_track=np.full(P, Gamma_track),               # PCRB trace threshold
        sigma_c2=sigma_s2,   # normalized
        sigma_s2=sigma_s2,
        Pmax=Pmax,
        N_req=2,
        N=s.N, M=s.M, K=s.K, P=s.P,
        use_s_procedure=use_s_procedure,
    )
    return s, prm


def main():
    print("=" * 60)
    print("ISAC Simulation - v10 (P3) Double-DC SCA - Single Point Test")
    print("=" * 60)
    print(f"  Topology: M=6 APs, K=3 UEs, P=3 targets, Nt=4 each")
    print(f"  Solver: CVXPY {__import__('cvxpy').__version__} + Clarabel")
    print()

    s, prm = build_scenario()
    print(f"  sigma_c2 = {prm.sigma_c2:.3e} W")
    print(f"  Pmax     = {prm.Pmax:.3e} W ({10*np.log10(prm.Pmax)+30:.1f} dBm)")
    print(f"  Channel norms: H = {[f'{np.linalg.norm(prm.H[:, k]):.2f}' for k in range(prm.K)]}")
    print(f"                  G = {[f'{np.linalg.norm(prm.G[:, p]):.2f}' for p in range(prm.P)]}")
    print()

    # First: verify (P3) cold start solves
    print("--- Cold start SDR (eta_rank = eta_b = 0) ---")
    t0 = time.time()
    sol0 = initial_feasible_solve(prm)
    dt0 = time.time() - t0
    print(f"  status: {sol0.status}")
    print(f"  obj:    {sol0.obj:.4f}")
    print(f"  rank deficiency: {[f'{r:.2e}' for r in (sol0.rank_deficiency if sol0.rank_deficiency is not None else np.zeros(prm.K))]}")
    print(f"  binary deficiency: {sol0.binary_deficiency:.4e}")
    print(f"  time:   {dt0:.2f}s")
    print()

    # Run all 4 baselines
    results = {}
    for name, fn, kwargs in [
        ("B1: Centralized (no AP selection)", baseline_b1_centralized, {"verbose": True}),
        ("B2: Heuristic AP selection (v5)",  baseline_b2_heuristic,    {"verbose": True}),
        ("B3: Single-shot SDR + GR",         baseline_b3_gr,           {"verbose": True, "n_samples": 30}),
        ("B4: Algorithm 2 (v10 double-DC)",  baseline_alg2,            {"verbose": True, "T_max": 30}),
    ]:
        print(f"--- {name} ---")
        t0 = time.time()
        res = fn(prm, **kwargs)
        dt = time.time() - t0
        results[name] = res
        print(f"  status:    {res.get('status','?')}")
        print(f"  iters:     {res.get('n_iter','?')}")
        print(f"  sum_rate:  {res.get('sum_rate_bps_hz', 0):.4f} bit/s/Hz")
        print(f"  sens_sinr: {[f'{x:.2f}' for x in res.get('sens_sinr_db', np.zeros(prm.P))]} dB")
        print(f"  pcrb:      {[f'{x:.4f}' for x in res.get('pcrb_trace', np.zeros(prm.P))]}")
        print(f"  rank_def:  {[f'{r:.2e}' for r in res.get('rank_def', np.zeros(prm.K))]}")
        print(f"  bin_def:   {res.get('binary_def', 0):.4e}")
        if res.get("obj_trace"):
            print(f"  obj trace: {[f'{x:.3f}' for x in res['obj_trace'][:8]]}...")
        print(f"  time:      {dt:.2f}s")
        if "w_mp_active" in res:
            print(f"  AP-target assignment:")
            for p in range(prm.P):
                active = np.where(res["w_mp_active"][:, p] > 0)[0]
                print(f"    target {p}: APs {list(active)}")
        print()

    # Comparison summary
    print("=" * 60)
    print("Summary (higher sum_rate & sensing SINR, lower PCRB = better)")
    print("=" * 60)
    print(f"{'Method':<40} {'Sum rate':>10} {'Avg sens (dB)':>14} {'Avg PCRB':>10}")
    print("-" * 78)
    for name, res in results.items():
        avg_sens = float(np.mean(res.get("sens_sinr_db", [0])))
        finite_pcrb = [x for x in res.get("pcrb_trace", [np.inf]) if np.isfinite(x)]
        avg_pcrb = float(np.mean(finite_pcrb)) if finite_pcrb else float("inf")
        print(f"{name:<40} {res.get('sum_rate_bps_hz', 0):>10.4f} {avg_sens:>14.2f} {avg_pcrb:>10.4f}")


if __name__ == "__main__":
    main()