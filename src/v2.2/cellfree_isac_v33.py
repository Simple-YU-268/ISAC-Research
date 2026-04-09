"""
Cell-free ISAC v33 - 通信64全参与 + 感知搜索最优AP数

方案：
- 通信：所有64 APs参与（固定）
- 感知：尝试不同AP数，找满足SNR的最小值
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


class CellFreeISACv33:
    def __init__(self, M=64, K=10, P=4, Nt=4, Pmax=3.2, sigma2=0.001,
                 sinr_req=10, snr_req=10, crb_req=1):
        self.M = M
        self.K = K
        self.P = P
        self.Nt = Nt
        self.Pmax = Pmax
        self.sigma2 = sigma2
        self.sinr_req = sinr_req
        self.snr_req = snr_req
        self.crb_req = crb_req
        self.epsilon = np.sqrt(0.05)
        
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
        e = self.epsilon * (np.random.randn(*H.shape) + 1j*np.random.randn(*H.shape))
        return H + e
    
    def select_sensing_aps(self, G, N_sens):
        """基于信道强度选择最佳N_sens个AP"""
        g_power = np.sum(np.abs(G)**2, axis=(1, 2))
        selected = np.argsort(-g_power)[:N_sens]
        mask = np.zeros(self.M, dtype=bool)
        mask[selected] = True
        return mask
    
    def mmse_beam_comm(self, H, Pmax):
        """通信波束 - 所有AP参与"""
        M, K, Nt = H.shape
        Hs = H.reshape(M * Nt, K)
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
        signal = 0
        noise = 0
        for p in range(P):
            for m in range(M_sel):
                g = G[m, p, :]
                if Z[m].shape == (Nt, Nt):
                    gH_Z_g = np.dot(g.conj(), Z[m] @ g)
                    signal += np.abs(gH_Z_g)
        noise = self.sigma2 * np.sum(np.real(np.trace(Z)))
        if noise > 0:
            return 10*np.log10(signal/(noise+1e-10) + 1e-10)
        return -100
    
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
                crbs.append(self.sigma2 / fisher_info)
            else:
                crbs.append(100)
        return np.array(crbs)
    
    def run(self, sens_list=[2,4,8,16,32,64], n_trials=30):
        print("=" * 70)
        print("ISAC v33 - 通信64全参与 + 感知搜索")
        print("=" * 70)
        print(f"通信: {self.M} APs全参与")
        print(f"感知: 搜索最小满足SNR的AP数")
        print(f"约束: SINR≥{self.sinr_req}dB, SNR≥{self.snr_req}dB, CRB≤{self.crb_req}m")
        print("=" * 70)
        
        results = []
        for i in range(n_trials):
            if i % 10 == 0:
                print(f"  {i}/{n_trials}")
            
            H_true, G_true = self.generate_trial()
            H_est = self.add_estimation_error(H_true)
            G_est = self.add_estimation_error(G_true)
            
            # 通信：所有64 APs（固定）
            W = self.mmse_beam_comm(H_est, self.Pmax * 0.8)
            sinrs = self.compute_sinr(H_est, W)
            
            # 感知：搜索满足SNR的最小AP数
            n_sens = 2
            success = False
            snr_val = -100
            crb_val = 100
            power = 0
            
            for n_try in sens_list:
                sens_mask = self.select_sensing_aps(G_est, n_try)
                G_sel = G_true[sens_mask]  # 用真实信道计算性能
                Z = self.sensing_beam(G_sel, self.Pmax * 0.2)
                
                snr_val = self.compute_snr(G_sel, Z)
                crb_val = self.compute_crb(G_sel, W)
                power = np.sum(np.abs(W)**2) + np.sum(np.real(np.trace(Z)))
                
                if (np.all(sinrs >= self.sinr_req) and 
                    snr_val >= self.snr_req and 
                    np.all(crb_val <= self.crb_req) and 
                    power <= self.Pmax):
                    n_sens = n_try
                    success = True
                    break
            
            results.append({
                'sinr_min': float(np.min(sinrs)),
                'snr': snr_val,
                'crb_max': float(np.max(crb_val)),
                'power': power,
                'n_sens': n_sens,
                'success': success
            })
        
        if results:
            ok = sum(1 for r in results if r['success'])
            print(f"\n  成功率: {ok}/{len(results)} = {100*ok/len(results):.1f}%")
            print(f"  平均感知AP: {np.mean([r['n_sens'] for r in results]):.1f}")
            print(f"  平均SINR: {np.mean([r['sinr_min'] for r in results]):.2f}dB")
            print(f"  平均SNR: {np.mean([r['snr'] for r in results]):.2f}dB")
            print(f"  平均功率: {np.mean([r['power'] for r in results]):.2f}W")
        
        print("\n完成!")


if __name__ == "__main__":
    isac = CellFreeISACv33(
        M=64, K=10, P=4, Nt=4,
        Pmax=3.2, sigma2=0.001,
        sinr_req=10, snr_req=10,
        crb_req=1
    )
    isac.run(sens_list=[2,4,8,16,32,64], n_trials=30)