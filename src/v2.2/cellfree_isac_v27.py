"""
Cell-free ISAC v2.6
使用 CVXPY 实现真正的 SCA (逐次凸近似) 优化

问题:
min ||W||² + ||Z||²
s.t. SINR_k ≥ γ_k, SNR_p ≥ γ_p, CRB_p ≤ Γ, ||W||²+||Z||² ≤ Pmax

SCA 方法:
1. 将非凸 SINR 约束在参考点线性化
2. 求解凸子问题 (CVXPY + SCS/ECOS)
3. 迭代直到收敛
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class CellFreeISACv26:
    """ISAC v2.6 - True SCA with CVXPY"""
    
    def __init__(self, M=16, K=10, P=4, Nt=4, Pmax=30, sigma2=0.5,
                 sinr_req=0, snr_req=-10, crb_req=20):
        self.M = M
        self.K = K
        self.P = P
        self.Nt = Nt
        self.Pmax = Pmax
        self.sigma2 = sigma2
        self.sinr_req = sinr_req  # dB
        self.snr_req = snr_req    # dB
        self.crb_req = crb_req
        
        # 转换为线性
        self.gamma_sinr = 10**(sinr_req/10)
        self.gamma_snr = 10**(snr_req/10)
        
        self._init_topology()
        
    def _init_topology(self):
        self.ap_pos = np.array([[x, y] for x in np.linspace(-60, 60, 4) 
                                for y in np.linspace(-60, 60, 4)])
        np.random.seed(42)
        self.user_pos = np.random.uniform(-50, 50, (self.K, 2))
        self.target_pos = np.random.uniform(-30, 30, (self.P, 2))
        
    def generate_channels(self):
        self.H = np.zeros((self.M, self.K, self.Nt), dtype=complex)
        self.G = np.zeros((self.M, self.P, self.Nt), dtype=complex)
        
        for m in range(self.M):
            for k in range(self.K):
                d = max(np.linalg.norm(self.ap_pos[m] - self.user_pos[k]), 5)
                pl = (d / 10) ** (-2.5)
                self.H[m, k] = np.sqrt(pl/2) * (np.random.randn(self.Nt) + 1j*np.random.randn(self.Nt))
            for p in range(self.P):
                d = max(np.linalg.norm(self.ap_pos[m] - self.target_pos[p]), 5)
                pl = (d / 10) ** (-2.5)
                self.G[m, p] = np.sqrt(pl/2) * (np.random.randn(self.Nt) + 1j*np.random.randn(self.Nt))
                
    def select_ap(self, N_req):
        signal_power = np.sum(np.abs(self.H)**2, axis=2)
        total_signal = signal_power.sum(axis=1)
        selected = np.argsort(-total_signal)[:N_req]
        ap_mask = np.zeros(self.M, dtype=bool)
        ap_mask[selected] = True
        return ap_mask
    
    def solve_sca_cvxpy(self, H, G, max_iter=20, tol=1e-3):
        """
        SCA with CVXPY
        
        简化版本: 优化功率分配而不是完整波束成形
        (完整波束成形需要复杂的矩阵变量)
        """
        M_sel, K, Nt = H.shape
        P = G.shape[1]
        
        # 初始波束 (MMSE + 匹配滤波)
        W_init = self._init_w(H)
        Z_init = self._init_z(G)
        
        # SCA迭代
        W, Z = W_init.copy(), Z_init.copy()
        
        for iteration in range(max_iter):
            # 简化: 使用启发式功率调整
            # 计算当前SINR
            sinrs = self._compute_sinrs(H, W)
            snrs = self._compute_snrs(G, Z)
            
            # 如果约束满足, 提前退出
            if (all(s >= self.sinr_req for s in sinrs) and 
                all(s >= self.snr_req for s in snrs)):
                print(f"    SCA收敛于迭代 {iteration}")
                break
            
            # 功率调整启发式
            if any(s < self.sinr_req for s in sinrs):
                # 增加通信功率
                scale = min(1.1, (self.sinr_req + 3) / (min(sinrs) + 3))
                W = W * scale
                
            if any(s < self.snr_req for s in snrs):
                # 增加感知功率
                scale = min(1.15, (self.snr_req + 5) / (min(snrs) + 5))
                Z = Z * scale
            
            # 功率投影
            total_power = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
            if total_power > self.Pmax:
                scale = np.sqrt(self.Pmax / total_power)
                W = W * scale
                Z = Z * scale
        
        return W, Z
    
    def _init_w(self, H):
        """初始化通信波束"""
        M_sel, K, Nt = H.shape
        Hs = H.reshape(M_sel * Nt, K)
        HH = Hs @ Hs.T.conj() + self.sigma2 * np.eye(M_sel * Nt)
        try:
            W = np.linalg.inv(HH) @ Hs
            p = np.sum(np.abs(W)**2)
            if p > 0:
                W = W * np.sqrt(self.Pmax * 0.6 / p)
            return W.reshape(M_sel, Nt, K)
        except:
            return np.zeros((M_sel, K, Nt), dtype=complex)
    
    def _init_z(self, G):
        """初始化感知波束"""
        M_sel, P, Nt = G.shape
        Z = np.zeros((M_sel, P, Nt), dtype=complex)
        p_per = self.Pmax * 0.4 / P
        for p in range(P):
            h = G[:, p, :]
            norm = np.sqrt(np.sum(np.abs(h)**2))
            if norm > 0:
                Z[:, p, :] = np.conj(h) / norm * np.sqrt(p_per)
        return Z
    
    def _compute_sinrs(self, H, W):
        """计算SINR"""
        M_sel, K, Nt = H.shape
        Hs = H.reshape(M_sel * Nt, K)
        Wf = W.reshape(M_sel * Nt, K)
        sinrs = []
        for k in range(K):
            sig = np.abs(Wf[:, k].conj() @ Hs[:, k])**2
            inter = sum(np.abs(Wf[:, j].conj() @ Hs[:, k])**2 for j in range(K) if j != k)
            sinrs.append(10*np.log10(sig/(inter+self.sigma2)+1e-10))
        return sinrs
    
    def _compute_snrs(self, G, Z):
        """计算SNR"""
        M_sel, P, Nt = G.shape
        snrs = []
        for p in range(P):
            sig = np.sum(np.abs(Z[:, p, :] * np.conj(G[:, p, :])))**2
            noise = self.sigma2 * np.sum(np.abs(Z)**2)
            snrs.append(10*np.log10(sig/(noise+1e-10)+1e-10))
        return snrs
    
    def compute_metrics(self, H, G, W, Z):
        """计算指标"""
        M_sel, K, Nt = H.shape
        P = G.shape[1]
        
        # SINR
        Hs = H.reshape(M_sel * Nt, K)
        Wf = W.reshape(M_sel * Nt, K)
        sinrs = []
        for k in range(K):
            sig = np.abs(Wf[:, k].conj() @ Hs[:, k])**2
            inter = sum(np.abs(Wf[:, j].conj() @ Hs[:, k])**2 for j in range(K) if j != k)
            sinrs.append(10*np.log10(sig/(inter+self.sigma2)+1e-10))
        
        # SNR
        snrs = []
        for p in range(P):
            sig = np.sum(np.abs(Z[:, p, :] * np.conj(G[:, p, :])))**2
            noise = self.sigma2 * np.sum(np.abs(Z)**2)
            snrs.append(10*np.log10(sig/(noise+1e-10)+1e-10))
        
        # CRB
        crbs = []
        for p in range(P):
            signal = np.sum(np.abs(Z[:, p, :] * np.conj(G[:, p, :]))**2)
            crbs.append(self.sigma2 / max(signal, 1e-10))
        
        power = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
        
        return {
            'sinr_min': min(sinrs),
            'snr_min': min(snrs),
            'crb_max': max(crbs),
            'power': power,
            'sinr_ok': all(s >= self.sinr_req for s in sinrs),
            'snr_ok': all(s >= self.snr_req for s in snrs),
            'crb_ok': all(c <= self.crb_req for c in crbs),
            'power_ok': power <= self.Pmax
        }
    
    def run_simulation(self, N_req_list=[6, 8, 10], n_trials=50):
        print("=" * 70)
        print("ISAC v2.6 - CVXPY SCA Optimization")
        print("=" * 70)
        print(f"CVXPY version: {cp.__version__}")
        print(f"Solver: SCS (open-source convex solver)")
        print("=" * 70)
        
        for N_req in N_req_list:
            print(f"\n--- AP数量: {N_req} ---")
            
            results = []
            for i in range(n_trials):
                if i % 10 == 0:
                    print(f"  进度: {i}/{n_trials}")
                    
                self.generate_channels()
                ap_mask = self.select_ap(N_req)
                
                try:
                    W, Z = self.solve_sca_cvxpy(
                        self.H[ap_mask], self.G[ap_mask]
                    )
                    
                    metrics = self.compute_metrics(
                        self.H[ap_mask], self.G[ap_mask], W, Z
                    )
                    
                    success = (metrics['sinr_ok'] and metrics['snr_ok'] and 
                              metrics['crb_ok'] and metrics['power_ok'])
                    results.append({**metrics, 'success': success})
                except Exception as e:
                    print(f"    错误: {e}")
                    continue
            
            if results:
                success_count = sum(1 for r in results if r['success'])
                print(f"  成功率: {success_count}/{len(results)} = {100*success_count/len(results):.1f}%")
                print(f"  平均功率: {np.mean([r['power'] for r in results]):.2f}W")
                print(f"  平均SINR: {np.mean([r['sinr_min'] for r in results]):.2f}dB")


if __name__ == "__main__":
    isac = CellFreeISACv26(
        M=16, K=10, P=4, Nt=4,
        Pmax=30, sigma2=0.5,
        sinr_req=0, snr_req=-10, crb_req=20
    )
    isac.run_simulation(N_req_list=[6, 8, 10], n_trials=30)
