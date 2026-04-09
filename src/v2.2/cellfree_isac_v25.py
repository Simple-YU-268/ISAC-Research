"""
Cell-free ISAC v2.2 Penalty Method
基于 v2.2 Enhanced + 惩罚函数迭代优化

解决非凸约束:
1. 将 SINR/SNR/CRB 约束转化为惩罚项
2. 迭代优化: 违反约束 → 增加惩罚权重 → 重新优化
"""

import numpy as np
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class CellFreeISACv22Penalty:
    """ISAC with Penalty Method for Non-Convex Constraints"""
    
    def __init__(self, M=16, K=10, P=4, Nt=4, Pmax=30, sigma2=0.5,
                 sinr_req=0, snr_req=-10, crb_req=20, error_var=0.05):
        self.M = M
        self.K = K
        self.P = P
        self.Nt = Nt
        self.Pmax = Pmax
        self.sigma2 = sigma2
        self.sinr_req = sinr_req
        self.snr_req = snr_req
        self.crb_req = crb_req
        self.error_var = error_var
        
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
    
    def mmse_with_penalty(self, H, G, lambda_penalty=1.0, max_iter=5):
        """
        带惩罚项的MMSE优化
        
        目标: min ||W||² + ||Z||² + λ·(惩罚项)
        惩罚项 = max(0, γ - SINR)² + max(0, γ - SNR)² + max(0, CRB - Γ)²
        """
        M_sel, K, Nt = H.shape
        _, P, _ = G.shape
        
        # 初始化
        W = np.zeros((M_sel, K, Nt), dtype=complex)
        Z = np.zeros((M_sel, P, Nt), dtype=complex)
        
        # 初始MMSE
        Hs = H.reshape(M_sel * Nt, K)
        HH = Hs @ Hs.T.conj() + self.sigma2 * np.eye(M_sel * Nt)
        try:
            W_flat = np.linalg.inv(HH) @ Hs
            p = np.sum(np.abs(W_flat)**2)
            if p > 0:
                W_flat = W_flat * np.sqrt(self.Pmax * 0.6 / p)
            W = W_flat.reshape(M_sel, Nt, K)
        except:
            pass
        
        # 初始Z
        p_sens = self.Pmax * 0.4 / P
        for p in range(P):
            h = G[:, p, :]
            norm = np.sqrt(np.sum(np.abs(h)**2))
            if norm > 0:
                Z[:, p, :] = np.conj(h) / norm * np.sqrt(p_sens)
        
        # 迭代优化惩罚项
        for iteration in range(max_iter):
            # 计算当前性能
            metrics = self.compute_metrics(H, G, W, Z)
            
            # 检查约束
            violations = 0
            if metrics['sinr_min'] < self.sinr_req:
                violations += (self.sinr_req - metrics['sinr_min'])**2
            if metrics['snr_min'] < self.snr_req:
                violations += (self.snr_req - metrics['snr_min'])**2
            if metrics['crb_max'] > self.crb_req:
                violations += (metrics['crb_max'] - self.crb_req)**2
            
            if violations == 0:
                break  # 所有约束满足
                
            # 简单启发式: 增加功率给表现差的链路
            if metrics['sinr_min'] < self.sinr_req:
                # 增加通信功率
                scale = min(1.2, (self.sinr_req + 5) / (metrics['sinr_min'] + 5))
                W = W * scale
                
            if metrics['snr_min'] < self.snr_req:
                # 增加感知功率
                scale = min(1.3, (self.snr_req + 10) / (metrics['snr_min'] + 10))
                Z = Z * scale
            
            # 功率投影
            total_power = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
            if total_power > self.Pmax:
                scale = np.sqrt(self.Pmax / total_power)
                W = W * scale
                Z = Z * scale
        
        return W, Z
    
    def compute_metrics(self, H, G, W, Z):
        M_sel, K, Nt = H.shape
        _, P, _ = G.shape
        
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
            'power': power
        }
    
    def run_simulation(self, N_req_list=[6, 8, 10], n_trials=50):
        print("=" * 70)
        print("ISAC v2.2 Penalty Method - 惩罚函数优化")
        print("=" * 70)
        
        for N_req in N_req_list:
            print(f"\n--- AP数量: {N_req} ---")
            
            results = []
            for i in range(n_trials):
                if i % 10 == 0:
                    print(f"  进度: {i}/{n_trials}")
                    
                self.generate_channels()
                ap_mask = self.select_ap(N_req)
                
                W, Z = self.mmse_with_penalty(
                    self.H[ap_mask], self.G[ap_mask]
                )
                
                metrics = self.compute_metrics(
                    self.H[ap_mask], self.G[ap_mask], W, Z
                )
                
                success = (
                    metrics['sinr_min'] >= self.sinr_req and
                    metrics['snr_min'] >= self.snr_req and
                    metrics['crb_max'] <= self.crb_req and
                    metrics['power'] <= self.Pmax
                )
                
                results.append({**metrics, 'success': success})
            
            success_count = sum(1 for r in results if r['success'])
            print(f"  成功率: {success_count}/{len(results)} = {100*success_count/len(results):.1f}%")
            print(f"  平均功率: {np.mean([r['power'] for r in results]):.2f}W")
            print(f"  平均SINR: {np.mean([r['sinr_min'] for r in results]):.2f}dB")


if __name__ == "__main__":
    isac = CellFreeISACv22Penalty(
        M=16, K=10, P=4, Nt=4,
        Pmax=30, sigma2=0.5,
        sinr_req=0, snr_req=-10, crb_req=20
    )
    isac.run_simulation(N_req_list=[6, 8, 10], n_trials=50)
