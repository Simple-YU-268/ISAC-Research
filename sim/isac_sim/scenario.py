"""
ISAC Cell-Free scenario generator.

Implements M-AP, K-UE, P-target topology with Rician-fading channels
and large-scale path loss. Default scale B2: M=6, K=3, P=3, N_t=4.

Conventions (matching math_derivation.tex):
  - APs indexed m = 0..M-1, each with N_t antennas -> M*N_t stacked
  - UEs indexed k = 0..K-1, single-antenna
  - Targets indexed p = 0..P-1, each with N_s streams (default 1)
  - Channel h_k in C^{M*N_t}, g_p in C^{M*N_t}
  - Cell-free: each AP has its own Z_m (sensing covariance slice)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Scenario:
    M: int                       # number of APs
    K: int                       # number of UEs
    P: int                       # number of sensing targets
    Nt: int                      # antennas per AP
    Ns: int = 1                  # streams per target
    ap_pos: np.ndarray | None = None   # (M, 2) in meters
    ue_pos: np.ndarray | None = None   # (K, 2) in meters
    tg_pos: np.ndarray | None = None   # (P, 2) in meters
    fc: float = 3.5e9            # carrier freq (Hz)
    noise_figure_db: float = 9.0
    noise_bw_hz: float = 20e6
    sigma_s2: float = 1.0        # sensing noise power (normalized)
    seed: int = 42

    @property
    def N(self) -> int:           # total antennas
        return self.M * self.Nt

    @property
    def sigma_c2(self) -> float:  # comm noise power
        # kTB + NF, with T=290K, B=noise_bw_hz
        kT = 1.380649e-23 * 290
        noise_w = kT * self.noise_bw_hz
        return noise_w * 10 ** (self.noise_figure_db / 10)


# ---------- geometry ----------

def make_positions(s: Scenario) -> Scenario:
    """Generate random positions in 200m x 200m square if not provided."""
    rng = np.random.default_rng(s.seed)
    if s.ap_pos is None:
        s.ap_pos = np.asarray(rng.uniform(0, 200, size=(s.M, 2)))
    if s.ue_pos is None:
        s.ue_pos = np.asarray(rng.uniform(0, 200, size=(s.K, 2)))
    if s.tg_pos is None:
        s.tg_pos = np.asarray(rng.uniform(0, 200, size=(s.P, 2)))
    return s


# ---------- path loss ----------

def pathloss_db(d_m: np.ndarray, fc_hz: float) -> np.ndarray:
    """3GPP UMi NLOS-like path loss (d in meters, fc in Hz)."""
    fc_ghz = fc_hz / 1e9
    return 36.7 * np.log10(d_m) + 22.7 + 26 * np.log10(fc_ghz)


def large_scale(s: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Return (beta_mk, beta_mp) large-scale fading in linear scale.
    beta_mk: (M, K) UE-side, beta_mp: (M, P) target-side.
    Includes shadowing (8 dB std).
    """
    rng = np.random.default_rng(s.seed + 1)
    beta_mk = np.zeros((s.M, s.K))
    beta_mp = np.zeros((s.M, s.P))
    for k in range(s.K):
        d = np.linalg.norm(s.ap_pos - s.ue_pos[k], axis=1)
        pl = pathloss_db(d, s.fc)
        sh = rng.normal(0, 8, size=s.M)
        beta_mk[:, k] = 10 ** (-(pl + sh) / 10)
    for p in range(s.P):
        d = np.linalg.norm(s.ap_pos - s.tg_pos[p], axis=1)
        pl = pathloss_db(d, s.fc)
        sh = rng.normal(0, 8, size=s.M)
        beta_mp[:, p] = 10 ** (-(pl + sh) / 10)
    return beta_mk, beta_mp


# ---------- small-scale fading (Rician, K-factor 10 dB) ----------

def small_scale(s: Scenario, beta_mk: np.ndarray, beta_mp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (H, G) with stacked AP antennas.
    H: (M*Nt, K), G: (M*Nt, P).
    h_k = [sqrt(beta_1k)*h_1k; ...; sqrt(beta_Mk)*h_Mk], each h_mk in C^{Nt}
    """
    K_rice = 10 ** (10 / 10)   # Rician K-factor (10 dB)
    rng = np.random.default_rng(s.seed + 2)

    def rician_vec(shape, K_factor, rng):
        # Rician: sqrt(K/(K+1))*LoS + sqrt(1/(K+1))*Rayleigh
        los = np.exp(1j * rng.uniform(0, 2 * np.pi, size=shape))
        ray = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
        return np.sqrt(K_factor / (K_factor + 1)) * los + np.sqrt(1 / (K_factor + 1)) * ray

    H = np.zeros((s.N, s.K), dtype=complex)
    G = np.zeros((s.N, s.P), dtype=complex)
    for k in range(s.K):
        for m in range(s.M):
            blk = rician_vec((s.Nt,), K_rice, rng)
            scale = np.sqrt(beta_mk[m, k])
            H[m * s.Nt:(m + 1) * s.Nt, k] = scale * blk
    for p in range(s.P):
        for m in range(s.M):
            blk = rician_vec((s.Nt,), K_rice, rng)
            scale = np.sqrt(beta_mp[m, p])
            G[m * s.Nt:(m + 1) * s.Nt, p] = scale * blk
    return H, G


# ---------- imperfect CSI: add bounded uncertainty ----------

def add_csi_uncertainty(H: np.ndarray, eps_h: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (H_hat, delta_H_norm_bound).
    Worst-case: we only need the bound, not the actual delta."""
    # Channel estimation error norm bound = eps_h * ||H_hat||
    # H_hat is the nominal (we just use H as the estimate)
    H_hat = H.copy()
    delta_bound = eps_h * np.linalg.norm(H_hat, axis=0)  # (K,)
    return H_hat, delta_bound


# ---------- per-AP subblock extraction (E_m) ----------

def block_indices(s: Scenario, m: int) -> tuple[int, int]:
    """Row/col indices for AP m in the stacked M*Nt x M*Nt covariance."""
    return m * s.Nt, (m + 1) * s.Nt


def E_m(s: Scenario, m: int) -> np.ndarray:
    """Diagonal matrix selecting AP m's antennas from stacked covariance."""
    E = np.zeros((s.N, s.N))
    lo, hi = block_indices(s, m)
    E[lo:hi, lo:hi] = np.eye(s.Nt)
    return E


# ---------- FIM (Fisher Information Matrix) for sensing ----------

def fim_data(s: Scenario, R_X: np.ndarray, G: np.ndarray, sigma_s2: float) -> np.ndarray:
    """FIM for target angle-Doppler, given sample covariance R_X.
    J_data = (1/sigma_s2^2) * G^H R_X G (per pulse, ignoring scaling).
    Dimension: P x P (target-by-target).
    Note: this is a simplified isotropic model; full version would have angle+Doppler
    per target, but for power allocation we use target-wise trace.
    """
    # Normalize G columns to unit-norm per target for fair scaling
    G_norm = G / (np.linalg.norm(G, axis=0, keepdims=True) + 1e-12)
    # J = (1/sigma_s2) * Re(G^H R_X G) for complex Gaussian observation
    J = (G_norm.conj().T @ R_X @ G_norm).real / sigma_s2
    return J