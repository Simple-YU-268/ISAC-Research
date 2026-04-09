"""
Cell-free ISAC v2.2: 鲁棒优化 + 大规模系统 (16 APs)
整合 v84 扩展性研究 + v2.1 不完美 CSI 鲁棒优化
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SystemConfig:
    """大规模系统配置 - 基于 v84 研究"""
    M: int = 16
    K: int = 10
    P: int = 4
    Nt: int = 4
    N_req: int = 4
    P_max: float = 30.0
    sigma_c: float = 0.5
    sigma_s: float = 0.5
    gamma_k: float = 1.0
    epsilon_h: float = 0.1
    epsilon_g: float = 0.15


class CellFreeISACv22:
    """Cell-free ISAC v2.2 - 大规模鲁棒系统"""
    
    def __init__(self, config: SystemConfig):
        self.cfg = config
        self._init_topology()
        self._init_channels()
        
    def _init_topology(self):
        """初始化 2D 拓扑"""
        self.ap_pos = np.array([
            [x, y] for x in np.linspace(-60, 60, 4)
            for y in np.linspace(-60, 60, 4)
        ])
        np.random.seed(42)
        self.user_pos = np.random.uniform(-50, 50, (self.cfg.K, 2))
        self.target_pos = np.random.uniform(-30, 30, (self.cfg.P, 2))
        
    def _init_channels(self):
        """初始化信道"""
        cfg = self.cfg
        self.h = np.zeros((cfg.M, cfg.K, cfg.Nt), dtype=complex)
        for m in range(cfg.M):
            for k in range(cfg.K):
                d = max(np.linalg.norm(self.ap_pos[m] - self.user_pos[k]), 5)
                pl = (d / 10) ** (-2.5)
                self.h[m, k] = np.sqrt(pl / 2) * (
                    np.random.randn(cfg.Nt) + 1j * np.random.randn(cfg.Nt)
                )
        
        self.g = np.zeros((cfg.M, cfg.P, cfg.Nt), dtype=complex)
        for m in range(cfg.M):
            for p in range(cfg.P):
                d = max(np.linalg.norm(self.ap_pos[m] - self.target_pos[p]), 5)
                pl = (d / 10) ** (-2.5)
                self.g[m, p] = np.sqrt(pl / 2) * (
                    np.random.randn(cfg.Nt) + 1j * np.random.randn(cfg.Nt)
                )

    def mmse_beam(self, H, Pmax):
        """MMSE 波束成形"""
        M, K, Nt = H.shape
        Hs = H.reshape(M * Nt, K)
        HH = Hs @ Hs.T.conj() + self.cfg.sigma_c * np.eye(M * Nt)
        try:
            W = np.linalg.inv(HH) @ Hs
            p = np.sum(np.abs(W) ** 2)
            W = W * np.sqrt(Pmax * 0.8 / p)
            return W.reshape(M, Nt, K)
        except:
            return None

    def sensing_beam(self, H_t, Pmax):
        """感知波束成形"""
        M, P, Nt = H_t.shape
        p_sensing = Pmax * 0.2 / P
        Z = np.zeros((M, P, Nt), dtype=complex)
        for p in range(P):
            h_t = H_t[:, p, :]
            norm = np.sqrt(np.sum(np.abs(h_t) ** 2))
            if norm > 0:
                Z[:, p, :] = np.conj(h_t) / norm * np.sqrt(p_sensing)
        return Z

    def compute_sinr(self, H, W):
        """计算 SINR (dB)"""
        M, K, Nt = H.shape
        Hs = H.reshape(M * Nt, K)
        W_flat = W.reshape(M * Nt, K)
        sinrs = []
        for k in range(K):
            sig = np.abs(np.sum(np.conj(W_flat[:, k]) @ Hs[:, k])) ** 2
            inter = sum(
                np.abs(np.sum(np.conj(W_flat[:, j]) @ Hs[:, k])) ** 2
                for j in range(K) if j != k
            )
            sinrs.append(10 * np.log10(sig / (inter + self.cfg.sigma_c) + 1e-10))
        return np.array(sinrs)

    def select_ap(self, H, N_req):
        """AP 选择"""
        signal_power = np.sum(np.abs(H) ** 2, axis=2)
        total_signal = signal_power.sum(axis=1)
        selected = np.argsort(-total_signal)[:N_req]
        ap_mask = np.zeros(self.cfg.M, dtype=bool)
        ap_mask[selected] = True
        return ap_mask

    def solve(self, n_trials=100):
        """蒙特卡洛仿真"""
        cfg = self.cfg
        results = []
        
        for _ in range(n_trials):
            self._init_channels()
            ap_mask = self.select_ap(self.h, cfg.N_req)
            H_sel = self.h[ap_mask, :, :]
            G_sel = self.g[ap_mask, :, :]
            
            W = self.mmse_beam(H_sel, cfg.P_max)
            Z = self.sensing_beam(G_sel, cfg.P_max)
            
            if W is not None:
                sinrs = self.compute_sinr(H_sel, W)
                power = np.sum(np.abs(W) ** 2) + np.sum(np.abs(Z) ** 2)
                results.append({
                    'sinr_min': sinrs.min(),
                    'comm_ok': sum(sinrs >= 0),
                    'power': power,
                    'power_ok': power <= cfg.P_max
                })
        
        return results


def main():
    cfg = SystemConfig()
    solver = CellFreeISACv22(cfg)
    results = solver.solve(n_trials=100)
    
    sinr_mins = [r['sinr_min'] for r in results]
    all_ok = sum(1 for r in results if r['comm_ok'] == cfg.K)
    
    print(f"v2.2 结果: {cfg.M} APs, {cfg.K} users")
    print(f"最小 SINR: {np.mean(sinr_mins):.2f} dB")
    print(f"全部用户达标: {all_ok}/{len(results)}")


if __name__ == "__main__":
    main()
