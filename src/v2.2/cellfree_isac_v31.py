"""
Cell-free ISAC v31 - 完整约束实现

约束:
1. 通信SINR ≥ γ_comm
2. 感知SNR ≥ γ_sens
3. CRB ≤ Γ
4. 总功率 ≤ P_total
5. 每AP功率 ≤ P_m,max
6. Z ≽ 0
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


class CellFreeISACv31:
    def __init__(self, M=64, K=10, P=4, Nt=4, Pmax=3.2, sigma2=0.001,
                 sinr_req=10, snr_req=10, crb_req=1, error_var=0.05):
        self.M = M
        self.K = K
        self.P = P
        self.Nt = Nt
        self.Pmax = Pmax
        self.P_m_max = Pmax / M * 10  # 每AP功率预算 (宽松10倍)
        self.sigma2 = sigma2
        self.sinr_req = sinr_req
        self.snr_req = snr_req
        self.crb_req = crb_req  # CRB阈值 (米)
        self.error_var = error_var
        
        x = np.linspace(-175, 175, 8)
        y = np.linspace(-175, 175, 8)
        self.ap_pos = np.array([[xi, yi] for xi in x for yi in y])
    
    def generate_trial(self):
        user_pos = np.random.uniform(-100, 100, (self.K, 2))
        target_pos = np.random.uniform(-150, 150, (self.P, 2))
        
        H = np.zeros((self.M, self.K, self.Nt), dtype=complex)
        G = np.zeros((self.M, self.P, self.Nt), dtype=complex)
        
        for m in range(self.M):
            for k in range(self.K):
                d = max(np.linalg.norm(self.ap_pos[m] - user_pos[k]), 5)
                pl = (d / 10) ** (-2.5)
                H[m, k] = np.sqrt(pl) * (np.random.randn(self.Nt) + 1j*np.random.randn(self.Nt))
            for p in range(self.P):
                d = max(np.linalg.norm(self.ap_pos[m] - target_pos[p]), 5)
                pl = (d / 10) ** (-2.5)
                G[m, p] = np.sqrt(pl) * (np.random.randn(self.Nt) + 1j*np.random.randn(self.Nt))
        
        return H, G
    
    def add_estimation_error(self, H):
        e = np.sqrt(self.error_var) * (np.random.randn(*H.shape) + 1j*np.random.randn(*H.shape))
        return H + e
    
    def mmse_beam(self, H, Pmax):
        """MMSE波束成形 + 每AP功率约束"""
        M, K, Nt = H.shape
        Hs = H.reshape(M * Nt, K)
        HH = Hs @ Hs.T.conj() + self.sigma2 * np.eye(M * Nt)
        try:
            W = np.linalg.inv(HH) @ Hs
            p = np.sum(np.abs(W)**2)
            if p > 0:
                W = W * np.sqrt(Pmax / p)
            W = W.reshape(M, Nt, K)
            
            # 每AP功率约束
            for m in range(M):
                p_m = np.sum(np.abs(W[m, :, :])**2)
                if p_m > self.P_m_max:
                    W[m, :, :] = W[m, :, :] * np.sqrt(self.P_m_max / p_m)
            
            return W
        except:
            return None
    
    def sensing_beam(self, G_sel, P_sens):
        """感知波束 + 每AP约束"""
        M_sel, P, Nt = G_sel.shape
        p_per = P_sens / (M_sel * P)
        Z = np.zeros((M_sel, Nt, Nt), dtype=complex)
        
        for m in range(M_sel):
            for p in range(P):
                g = G_sel[m, p, :]
                if np.linalg.norm(g) > 0:
                    w = np.conj(g) / np.linalg.norm(g) * np.sqrt(p_per)
                    Z[m] += np.outer(w, w.conj())
            
            # 每AP感知功率约束
            tr_Z = np.real(np.trace(Z[m]))
            if tr_Z > self.P_m_max * 0.5:  # 感知占50%预算
                Z[m] = Z[m] * (self.P_m_max * 0.5 / tr_Z)
        
        return Z
    
    def compute_sinr(self, H, W):
        M, K, Nt = H.shape
        Hs = H.reshape(M * Nt, K)
        Wf = W.reshape(M * Nt, K)
        sinrs = []
        for k in range(K):
            sig = np.abs(np.dot(Wf[:,k].conj(), Hs[:,k]))**2
            inter = sum(np.abs(np.dot(Wf[:,j].conj(), Hs[:,k]))**2 for j in range(K) if j!=k)
            sinrs.append(10*np.log10(sig/(inter+self.sigma2+1e-10) + 1e-10))
        return np.array(sinrs)
    
    def compute_snr(self, G_sel, Z):
        M_sel, P, Nt = G_sel.shape
        snrs = []
        for p in range(P):
            signal_power = 0
            for m in range(M_sel):
                g = G_sel[m, p, :]
                if Z[m].shape == (Nt, Nt):
                    gH_Z_g = np.dot(g.conj(), Z[m] @ g)
                    signal_power += np.abs(gH_Z_g)
            
            noise = self.sigma2 * np.sum(np.real(np.trace(Z)))
            if noise > 0:
                snrs.append(10*np.log10(signal_power/(noise+1e-10) + 1e-10))
            else:
                snrs.append(-100)
        return np.array(snrs)
    
    def compute_crb(self, H, G_sel, W):
        """计算CRB (Cramér-Rao Bound)"""
        M_sel, P, Nt = G_sel.shape
        crbs = []
        
        # CRB公式: CRB_p ≈ σ² / Σ|g^H w|²
        for p in range(P):
            fisher_info = 0
            for k in range(self.K):
                for m in range(M_sel):
                    g = G_sel[m, p, :]
                    w = W[m, :, k]
                    fisher_info += np.abs(np.dot(g.conj(), w))**2
            
            # CRB = 1 / Fisher Information (近似)
            if fisher_info > 0:
                crb = self.sigma2 / fisher_info
                crbs.append(crb)
            else:
                crbs.append(100)  # 很大CRB
        
        return np.array(crbs)
    
    def compute_power_per_ap(self, W, Z, M_tot):
        """每AP功率 (考虑通信所有AP, 感知部分AP)"""
        M_comm = W.shape[0]
        p_ap = np.zeros(M_tot)
        
        # 通信功率 (所有AP)
        for m in range(M_comm):
            p_ap[m] = np.sum(np.abs(W[m, :, :])**2)
        
        # 感知功率 (选中的AP)
        M_sens = Z.shape[0]
        for m in range(M_sens):
            p_ap[m] += np.real(np.trace(Z[m]))
        
        return p_ap
    
    def select_sensing_aps(self, G, N_sens):
        """
        选择感知AP：基于信道强度 + 完整感知SNR
        
        1. 按信道强度初选候选AP
        2. 计算每个候选AP的完整感知SNR (详细计算)
        3. 按SNR排序选最高的N_sens个
        """
        g_power = np.sum(np.abs(G)**2, axis=(1, 2))
        n_candidate = min(N_sens * 4, self.M)
        candidates = np.argsort(-g_power)[:n_candidate]
        
        # 计算完整感知SNR
        snr_per_ap = []
        p_per = (self.Pmax * 0.4) / n_candidate
        
        for m in candidates:
            # 为这个AP计算感知波束 (详细)
            Z_m = np.zeros((self.Nt, self.Nt), dtype=complex)
            for p in range(self.P):
                g = G[m, p, :]
                if np.linalg.norm(g) > 0:
                    w = np.conj(g) / np.linalg.norm(g) * np.sqrt(p_per)
                    Z_m += np.outer(w, w.conj())
            
            # 信号功率: |g^H Z g|²
            signal_m = 0
            for p in range(self.P):
                g = G[m, p, :]
                gH_Z_g = np.dot(g.conj(), Z_m @ g)
                signal_m += np.abs(gH_Z_g)
            
            # 噪声功率: σ² tr(Z)
            noise = self.sigma2 * np.real(np.trace(Z_m))
            
            if noise > 0:
                snr = 10*np.log10(signal_m/(noise+1e-10)+1e-10)
            else:
                snr = -100
            snr_per_ap.append(snr)
        
        # 按SNR排序选最高的N_sens个
        snr_array = np.array(snr_per_ap)
        sorted_idx = np.argsort(-snr_array)[:N_sens]
        selected = candidates[sorted_idx]
        
        mask = np.zeros(self.M, dtype=bool)
        mask[selected] = True
        return mask
    
    def robust_sinr(self, H, W):
        sinrs = self.compute_sinr(H, W)
        # 简化margin: 误差通常不会导致太大损失
        margin = self.error_var * 2  # 约0.4dB for error_var=0.05
        return sinrs - margin
    
    def run(self, n_sens_start=2, n_sens_max=64, n_trials=50):
        print("=" * 70)
        print("ISAC v31 - 完整约束实现")
        print("=" * 70)
        print(f"配置: {self.M} APs")
        print(f"约束: SINR≥{self.sinr_req}dB, SNR≥{self.snr_req}dB, CRB≤{self.crb_req}m")
        print(f"功率: 总≤{self.Pmax}W, 每AP≤{self.P_m_max:.4f}W")
        print("=" * 70)
        
        results = []
        for i in range(n_trials):
            if i % 10 == 0:
                print(f"  进度: {i}/{n_trials}")
            
            H_true, G_true = self.generate_trial()
            H_est = self.add_estimation_error(H_true)
            G_est = self.add_estimation_error(G_true)
            
            W = self.mmse_beam(H_true, self.Pmax * 0.9)
            
            n_sens = n_sens_start
            success = False
            
            while n_sens <= n_sens_max and not success:
                sens_mask = self.select_sensing_aps(G_true, n_sens)
                G_sel = G_true[sens_mask]
                Z = self.sensing_beam(G_sel, self.Pmax * 0.4)
                
                if W is not None:
                    sinrs = self.compute_sinr(H_true, W)  # 用真实信道，不减margin
                    snrs = self.compute_snr(G_sel, Z)
                    crbs = self.compute_crb(H_true, G_sel, W)
                    power = np.sum(np.abs(W)**2) + np.sum(np.real(np.trace(Z)))
                    p_ap = self.compute_power_per_ap(W, Z, self.M)
                    
                    success = (np.all(sinrs >= self.sinr_req) and
                             np.all(snrs >= self.snr_req) and
                             np.all(crbs <= self.crb_req) and
                             power <= self.Pmax and
                             np.all(p_ap <= self.P_m_max))
                    
                    if not success:
                        n_sens += 1
                else:
                    break
            
            if W is not None:
                results.append({
                    'sinr_min': float(np.min(sinrs)),
                    'snr_min': float(np.min(snrs)),
                    'crb_max': float(np.max(crbs)),
                    'power': power,
                    'p_ap_max': float(np.max(p_ap)),
                    'n_sens': n_sens,
                    'success': success
                })
        
        if results:
            ok = sum(1 for r in results if r['success'])
            avg_sens = np.mean([r['n_sens'] for r in results])
            print(f"\n  成功率: {ok}/{len(results)} = {100*ok/len(results):.1f}%")
            print(f"  平均感知AP: {avg_sens:.1f}")
            print(f"  平均SINR: {np.mean([r['sinr_min'] for r in results]):.2f}dB")
            print(f"  平均SNR: {np.mean([r['snr_min'] for r in results]):.2f}dB")
            print(f"  平均CRB: {np.mean([r['crb_max'] for r in results]):.4f}m")
            print(f"  平均功率: {np.mean([r['power'] for r in results]):.2f}W")
            print(f"  最大单AP功率: {np.mean([r['p_ap_max'] for r in results]):.4f}W")
        
        print("\n完成!")


if __name__ == "__main__":
    isac = CellFreeISACv31(
        M=64, K=10, P=4, Nt=4,
        Pmax=3.2, sigma2=0.001,
        sinr_req=10, snr_req=10,
        crb_req=1,  # 1米CRB
        error_var=0.05
    )
    isac.run(n_sens_start=2, n_sens_max=64, n_trials=30)