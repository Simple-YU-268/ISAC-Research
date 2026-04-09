"""
Cell-free ISAC v32 - 联合AP优化

通信和感知使用同一组AP，联合优化选择
联合 metric = α × h_comm + (1-α) × g_sens
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


class CellFreeISACv32:
    def __init__(self, M=64, K=10, P=4, Nt=4, Pmax=3.2, sigma2=0.001,
                 sinr_req=10, snr_req=10, crb_req=1, alpha=0.5, error_var=0.05):
        self.M = M
        self.K = K
        self.P = P
        self.Nt = Nt
        self.Pmax = Pmax
        self.sigma2 = sigma2
        self.sinr_req = sinr_req
        self.snr_req = snr_req
        self.crb_req = crb_req
        self.alpha = alpha  # 通信权重 (1-α=感知权重)
        self.epsilon = np.sqrt(error_var)  # √(ε²)
        
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
        """信道估计误差"""
        e = self.epsilon * (np.random.randn(*H.shape) + 1j*np.random.randn(*H.shape))
        return H + e
    
    def select_joint_aps(self, H, G, N_joint):
        """
        联合AP选择 - 使用估计的信道
        
        实际系统中：只能获得带误差的信道估计
        选择metric基于H和G（不是真实位置）
        """
        # 归一化通信强度
        h_power = np.sum(np.abs(H)**2, axis=(1, 2))
        h_power = h_power / (np.max(h_power) + 1e-10)
        
        # 归一化感知强度  
        g_power = np.sum(np.abs(G)**2, axis=(1, 2))
        g_power = g_power / (np.max(g_power) + 1e-10)
        
        # 联合评分
        joint_score = self.alpha * h_power + (1 - self.alpha) * g_power
        
        # 选择前N_joint个
        selected = np.argsort(-joint_score)[:N_joint]
        mask = np.zeros(self.M, dtype=bool)
        mask[selected] = True
        return mask
    
    def mmse_beam(self, H_sel, Pmax):
        """MMSE波束成形"""
        M, K, Nt = H_sel.shape
        Hs = H_sel.reshape(M * Nt, K)
        HH = Hs @ Hs.T.conj() + self.sigma2 * np.eye(M * Nt)
        try:
            W = np.linalg.inv(HH) @ Hs
            p = np.sum(np.abs(W)**2)
            if p > 0:
                W = W * np.sqrt(Pmax / p)
            return W.reshape(M, Nt, K)
        except:
            return None
    
    def sensing_beam(self, G_sel, P_sens):
        """感知波束"""
        M_sel, P, Nt = G_sel.shape
        p_per = P_sens / (M_sel * P)
        Z = np.zeros((M_sel, Nt, Nt), dtype=complex)
        
        for m in range(M_sel):
            for p in range(P):
                g = G_sel[m, p, :]
                if np.linalg.norm(g) > 0:
                    w = np.conj(g) / np.linalg.norm(g) * np.sqrt(p_per)
                    Z[m] += np.outer(w, w.conj())
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
    
    def compute_snr(self, G, Z):
        M_sel, P, Nt = G.shape
        snrs = []
        for p in range(P):
            signal_power = 0
            for m in range(M_sel):
                g = G[m, p, :]
                if Z[m].shape == (Nt, Nt):
                    gH_Z_g = np.dot(g.conj(), Z[m] @ g)
                    signal_power += np.abs(gH_Z_g)
            
            noise = self.sigma2 * np.sum(np.real(np.trace(Z)))
            if noise > 0:
                snrs.append(10*np.log10(signal_power/(noise+1e-10) + 1e-10))
            else:
                snrs.append(-100)
        return np.array(snrs)
    
    def compute_crb(self, G, W):
        M_sel, P, Nt = G.shape
        crbs = []
        
        for p in range(P):
            fisher_info = 0
            for k in range(self.K):
                for m in range(M_sel):
                    g = G[m, p, :]
                    w = W[m, :, k]
                    fisher_info += np.abs(np.dot(g.conj(), w))**2
            
            if fisher_info > 0:
                crb = self.sigma2 / fisher_info
                crbs.append(crb)
            else:
                crbs.append(100)
        
        return np.array(crbs)
    
    def run(self, n_joint_list=[4, 8, 16, 32], n_trials=30):
        print("=" * 70)
        print("ISAC v32 - 联合AP优化")
        print("=" * 70)
        print(f"α通信权重={self.alpha}, (1-α)感知权重={1-self.alpha}")
        print(f"约束: SINR≥{self.sinr_req}dB, SNR≥{self.snr_req}dB, CRB≤{self.crb_req}m")
        print("=" * 70)
        
        for N_joint in n_joint_list:
            print(f"\n--- 联合AP数: {N_joint} ---")
            
            results = []
            for i in range(n_trials):
                if i % 10 == 0:
                    print(f"  {i}/{n_trials}")
                
                H_true, G_true = self.generate_trial()
                # 添加信道估计误差 (实际系统只能得到估计信道)
                H_est = self.add_estimation_error(H_true)
                G_est = self.add_estimation_error(G_true)
                
                # 用估计信道选择AP (实际可行的方式)
                joint_mask = self.select_joint_aps(H_est, G_est, N_joint)
                H_sel = H_true[joint_mask]
                G_sel = G_true[joint_mask]
                
                # 通信波束 (选中的AP)
                W = self.mmse_beam(H_sel, self.Pmax * 0.8)
                Z = self.sensing_beam(G_sel, self.Pmax * 0.2)
                
                if W is not None:
                    sinrs = self.compute_sinr(H_sel, W)
                    snrs = self.compute_snr(G_sel, Z)
                    crbs = self.compute_crb(G_sel, W)
                    power = np.sum(np.abs(W)**2) + np.sum(np.real(np.trace(Z)))
                    
                    success = (np.all(sinrs >= self.sinr_req) and
                             np.all(snrs >= self.snr_req) and
                             np.all(crbs <= self.crb_req) and
                             power <= self.Pmax)
                    
                    results.append({
                        'sinr_min': float(np.min(sinrs)),
                        'snr_min': float(np.min(snrs)),
                        'crb_max': float(np.max(crbs)),
                        'power': power,
                        'success': success
                    })
            
            if results:
                ok = sum(1 for r in results if r['success'])
                print(f"  成功率: {ok}/{len(results)} = {100*ok/len(results):.1f}%")
                print(f"  平均SINR: {np.mean([r['sinr_min'] for r in results]):.2f}dB")
                print(f"  平均SNR: {np.mean([r['snr_min'] for r in results]):.2f}dB")
                print(f"  平均功率: {np.mean([r['power'] for r in results]):.2f}W")
        
        print("\n完成!")


if __name__ == "__main__":
    isac = CellFreeISACv32(
        M=64, K=10, P=4, Nt=4,
        Pmax=3.2, sigma2=0.001,
        sinr_req=10, snr_req=10,
        crb_req=1,
        alpha=0.5,
        error_var=0.05
    )
    isac.run(n_joint_list=[4, 8, 16, 32], n_trials=30)