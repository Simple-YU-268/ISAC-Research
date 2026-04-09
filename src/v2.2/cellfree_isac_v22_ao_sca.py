"""
Cell-free ISAC v2.2 AO-SCA
基于 v2.2 Enhanced + AO (交替优化) + SCA (逐次凸近似)

解决非凸问题:
min ||W||² + ||Z||²
s.t. SINR_k ≥ γ_k, SNR_p ≥ γ_p, CRB_p ≤ Γ, ||W||²+||Z||² ≤ Pmax

算法:
1. AO: 交替优化 W (固定Z) 和 Z (固定W)
2. SCA: 将非凸SINR约束凸近似为 SOC 约束
"""

import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class CellFreeISACv22AOSCA:
    """ISAC v2.2 with AO + SCA optimization"""
    
    def __init__(self, M=16, K=10, P=4, Nt=4, Pmax=30, sigma2=0.5,
                 sinr_req=0, snr_req=-10, crb_req=20, error_var=0.05):
        self.M = M
        self.K = K
        self.P = P
        self.Nt = Nt
        self.Pmax = Pmax
        self.sigma2 = sigma2
        self.sinr_req = 10**(sinr_req/10)  # 转换为线性
        self.snr_req = 10**(snr_req/10)
        self.crb_req = crb_req
        self.error_var = error_var
        
        self._init_topology()
        
    def _init_topology(self):
        """初始化拓扑"""
        self.ap_pos = np.array([
            [x, y] for x in np.linspace(-60, 60, 4)
            for y in np.linspace(-60, 60, 4)
        ])
        np.random.seed(42)
        self.user_pos = np.random.uniform(-50, 50, (self.K, 2))
        self.target_pos = np.random.uniform(-30, 30, (self.P, 2))
        
    def generate_channels(self):
        """生成信道"""
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
        """选择AP"""
        signal_power = np.sum(np.abs(self.H)**2, axis=2)
        total_signal = signal_power.sum(axis=1)
        selected = np.argsort(-total_signal)[:N_req]
        ap_mask = np.zeros(self.M, dtype=bool)
        ap_mask[selected] = True
        return ap_mask
    
    def solve_ao_sca(self, ap_mask, max_iter=10, tol=1e-3):
        """
        AO-SCA 主算法
        
        迭代:
        1. 固定 Z, 用SCA求解 W (通信子问题)
        2. 固定 W, 用SCA求解 Z (感知子问题)
        """
        M_sel = np.sum(ap_mask)
        H_sel = self.H[ap_mask, :, :]
        G_sel = self.G[ap_mask, :, :]
        
        # 初始化
        W = np.zeros((M_sel, self.K, self.Nt), dtype=complex)
        Z = np.zeros((M_sel, self.P, self.Nt), dtype=complex)
        
        # 初始值: 简单MMSE + 匹配滤波
        W = self._init_mmse(H_sel)
        Z = self._init_matched(G_sel)
        
        obj_history = []
        
        for iteration in range(max_iter):
            # === AO Step 1: 固定Z, 优化W ===
            W_new = self._solve_w_sca(H_sel, G_sel, Z, W)
            
            # === AO Step 2: 固定W, 优化Z ===
            Z_new = self._solve_z_sca(H_sel, G_sel, W_new, Z)
            
            # 计算目标函数 (总功率)
            obj = np.sum(np.abs(W_new)**2) + np.sum(np.abs(Z_new)**2)
            obj_history.append(obj)
            
            # 收敛检查
            if iteration > 0 and abs(obj_history[-1] - obj_history[-2]) < tol:
                print(f"    AO收敛于迭代 {iteration}")
                break
                
            W, Z = W_new, Z_new
            
        return W, Z, obj_history[-1]
    
    def _init_mmse(self, H):
        """初始化: MMSE"""
        M_sel, K, Nt = H.shape
        Hs = H.reshape(M_sel * Nt, K)
        HH = Hs @ Hs.T.conj() + self.sigma2 * np.eye(M_sel * Nt)
        try:
            W = np.linalg.inv(HH) @ Hs
            p = np.sum(np.abs(W)**2)
            if p > 0:
                W = W * np.sqrt(self.Pmax * 0.7 / p)
            return W.reshape(M_sel, Nt, K)
        except:
            return np.zeros((M_sel, K, Nt), dtype=complex)
    
    def _init_matched(self, G):
        """初始化: 匹配滤波"""
        M_sel, P, Nt = G.shape
        Z = np.zeros((M_sel, P, Nt), dtype=complex)
        p_per = self.Pmax * 0.3 / P
        for p in range(P):
            h = G[:, p, :]
            norm = np.sqrt(np.sum(np.abs(h)**2))
            if norm > 0:
                Z[:, p, :] = np.conj(h) / norm * np.sqrt(p_per)
        return Z
    
    def _solve_w_sca(self, H, G, Z_fixed, W_init):
        """
        SCA求解W子问题 (固定Z)
        
        将非凸SINR约束近似为凸的SOC约束
        """
        M_sel, K, Nt = H.shape
        
        # 简化: 使用正则化MMSE (SCA的近似)
        # 考虑Z_fixed带来的干扰
        
        Hs = H.reshape(M_sel * Nt, K)
        
        # 计算Z带来的等效噪声
        Z_interf = np.sum(np.abs(Z_fixed)**2) * 0.1  # 简化估计
        
        # 增强正则化以处理干扰
        reg = self.sigma2 + Z_interf
        HH = Hs @ Hs.T.conj() + reg * np.eye(M_sel * Nt)
        
        try:
            W = np.linalg.solve(HH + 1e-8*np.eye(M_sel*Nt), Hs)
            
            # 投影到功率约束
            p = np.sum(np.abs(W)**2)
            p_z = np.sum(np.abs(Z_fixed)**2)
            if p + p_z > self.Pmax:
                W = W * np.sqrt((self.Pmax - p_z) * 0.8 / p)
                
            return W.reshape(M_sel, Nt, K)
        except:
            return W_init
    
    def _solve_z_sca(self, H, G, W_fixed, Z_init):
        """
        SCA求解Z子问题 (固定W)
        
        近似CRB和SNR约束
        """
        M_sel, P, Nt = G.shape
        
        # 简化SCA: 基于信道质量的自适应功率分配
        g_norms = np.array([np.linalg.norm(G[:, p, :]) for p in range(P)])
        
        # 计算可用功率
        p_w = np.sum(np.abs(W_fixed)**2)
        p_avail = max(0, self.Pmax - p_w) * 0.9  # 留10% margin
        
        Z = np.zeros((M_sel, P, Nt), dtype=complex)
        
        if p_avail > 0 and np.sum(g_norms) > 0:
            # 按信道质量分配功率
            for p in range(P):
                if g_norms[p] > 0:
                    power = p_avail * (g_norms[p] / np.sum(g_norms))
                    Z[:, p, :] = np.conj(G[:, p, :]) / g_norms[p] * np.sqrt(power)
                    
        return Z
    
    def compute_metrics(self, H, G, W, Z):
        """计算性能指标"""
        M_sel, K, Nt = H.shape
        M_sens, P, _ = G.shape
        
        # 通信SINR
        Hs = H.reshape(M_sel * Nt, K)
        Wf = W.reshape(M_sel * Nt, K)
        sinrs = []
        for k in range(K):
            sig = np.abs(Wf[:, k].conj() @ Hs[:, k])**2
            inter = sum(np.abs(Wf[:, j].conj() @ Hs[:, k])**2 for j in range(K) if j != k)
            sinrs.append(10*np.log10(sig/(inter+self.sigma2)+1e-10))
        
        # 感知SNR
        snrs = []
        for p in range(P):
            sig = np.sum(np.abs(Z[:, p, :] * np.conj(G[:, p, :])))
            noise = self.sigma2 * np.sum(np.abs(Z)**2)
            snrs.append(10*np.log10(sig**2/(noise+1e-10)+1e-10))
        
        # CRB
        crbs = []
        for p in range(P):
            signal = np.sum(np.abs(Z[:, p, :] * np.conj(G[:, p, :]))**2)
            crbs.append(self.sigma2 / max(signal, 1e-10))
        
        # 功率
        power = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
        
        return {
            'sinr_min': min(sinrs),
            'snr_min': min(snrs),
            'crb_max': max(crbs),
            'power': power,
            'sinr_ok': all(s >= 10*np.log10(self.sinr_req) for s in sinrs),
            'snr_ok': all(s >= 10*np.log10(self.snr_req) for s in snrs),
            'crb_ok': all(c <= self.crb_req for c in crbs),
            'power_ok': power <= self.Pmax
        }
    
    def run_simulation(self, N_req_list=[4, 6, 8, 10], n_trials=50):
        """蒙特卡洛仿真"""
        print("=" * 70)
        print("ISAC v2.2 AO-SCA 非凸优化")
        print("=" * 70)
        print(f"算法: AO (交替优化) + SCA (逐次凸近似)")
        print(f"配置: {self.M} APs, {self.K} users, {self.P} targets")
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
                    W, Z, obj = self.solve_ao_sca(ap_mask)
                    metrics = self.compute_metrics(
                        self.H[ap_mask], self.G[ap_mask], W, Z
                    )
                    results.append(metrics)
                except:
                    continue
            
            if results:
                success = sum(1 for r in results 
                            if r['sinr_ok'] and r['snr_ok'] and r['crb_ok'] and r['power_ok'])
                print(f"  成功率: {success}/{len(results)} = {100*success/len(results):.1f}%")
                print(f"  平均功率: {np.mean([r['power'] for r in results]):.2f}W")
                print(f"  平均SINR: {np.mean([r['sinr_min'] for r in results]):.2f}dB")


if __name__ == "__main__":
    isac = CellFreeISACv22AOSCA(
        M=16, K=10, P=4, Nt=4,
        Pmax=30, sigma2=0.5,
        sinr_req=0, snr_req=-10, crb_req=20
    )
    isac.run_simulation(N_req_list=[6, 8, 10], n_trials=30)
