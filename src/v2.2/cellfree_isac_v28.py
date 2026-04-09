"""
Cell-free ISAC v28 - 大规模系统 (64 APs)
自适应感知AP选择: 起始2个，不达标自动增加
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


class CellFreeISACv28:
    """ISAC v28 - 64 APs: 全AP通信 + 自适应感知"""
    
    def __init__(self, M=64, K=10, P=4, Nt=4, Pmax=100, sigma2=0.5,
                 sinr_req=0, snr_req=-10, error_var=0.05):
        self.M = M
        self.K = K
        self.P = P
        self.Nt = Nt
        self.Pmax = Pmax
        self.sigma2 = sigma2
        self.sinr_req = sinr_req
        self.snr_req = snr_req
        self.error_var = error_var
        
        # 64 APs 8x8网格
        x = np.linspace(-175, 175, 8)
        y = np.linspace(-175, 175, 8)
        self.ap_pos = np.array([[xi, yi] for xi in x for yi in y])
    
    def generate_trial(self):
        """每次trial生成随机位置"""
        user_pos = np.random.uniform(-100, 100, (self.K, 2))
        target_pos = np.random.uniform(-150, 150, (self.P, 2))
        
        H = np.zeros((self.M, self.K, self.Nt), dtype=complex)
        G = np.zeros((self.M, self.P, self.Nt), dtype=complex)
        
        for m in range(self.M):
            for k in range(self.K):
                d = max(np.linalg.norm(self.ap_pos[m] - user_pos[k]), 5)
                pl = (d / 10) ** (-2.5)
                H[m, k] = np.sqrt(pl/2) * (np.random.randn(self.Nt) + 1j*np.random.randn(self.Nt))
            for p in range(self.P):
                d = max(np.linalg.norm(self.ap_pos[m] - target_pos[p]), 5)
                pl = (d / 10) ** (-2.5)
                G[m, p] = np.sqrt(pl/2) * (np.random.randn(self.Nt) + 1j*np.random.randn(self.Nt))
        
        return H, G
    
    def add_estimation_error(self, H):
        return H + np.sqrt(self.error_var/2) * (np.random.randn(*H.shape) + 1j*np.random.randn(*H.shape))
    
    def select_sensing_aps(self, G, N_sens):
        """选择感知最佳的 N_sens 个 AP"""
        g_power = np.sum(np.abs(G)**2, axis=(1, 2))
        selected = np.argsort(-g_power)[:N_sens]
        mask = np.zeros(self.M, dtype=bool)
        mask[selected] = True
        return mask
    
    def mmse_beam_all_aps(self, H, Pmax):
        """所有AP参与通信波束成形"""
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
        """感知波束成形 - 匹配滤波"""
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
        """计算SINR"""
        M, K, Nt = H.shape
        Hs = H.reshape(M * Nt, K)
        Wf = W.reshape(M * Nt, K)
        sinrs = []
        for k in range(K):
            sig = np.abs(np.dot(Wf[:,k].conj(), Hs[:,k]))**2
            inter = sum(np.abs(np.dot(Wf[:,j].conj(), Hs[:,k]))**2 for j in range(K) if j!=k)
            sinrs.append(10*np.log10(sig/(inter+self.sigma2+1e-10)+1e-10))
        return np.array(sinrs)
    
    def compute_snr(self, G_sel, Z):
        """计算SNR"""
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
                snr_linear = signal_power / (noise + 1e-10)
                snrs.append(10*np.log10(snr_linear + 1e-10))
            else:
                snrs.append(-100)
        return np.array(snrs)
    
    def run(self, n_sens_start=2, n_sens_max=64, n_trials=50):
        print("=" * 70)
        print("ISAC v28 - 64 APs: 全AP通信 + 自适应感知AP")
        print("=" * 70)
        print(f"配置: {self.M} APs 全部参与通信")
        print(f"感知: 自适应 (起始{n_sens_start}, 最大{n_sens_max})")
        print(f"功率预算: {self.Pmax}W")
        print(f"约束: SINR≥{self.sinr_req}dB, SNR≥{self.snr_req}dB")
        print("=" * 70)
        
        results = []
        for i in range(n_trials):
            if i % 10 == 0:
                print(f"  进度: {i}/{n_trials}")
            
            H_true, G_true = self.generate_trial()
            H_est = self.add_estimation_error(H_true)
            G_est = self.add_estimation_error(G_true)
            
            # 通信: 所有64 APs
            W = self.mmse_beam_all_aps(H_true, self.Pmax * 0.6)
            
            # 自适应感知AP
            n_sens = n_sens_start
            success = False
            
            while n_sens <= n_sens_max and not success:
                sens_mask = self.select_sensing_aps(G_true, n_sens)
                G_sel = G_true[sens_mask]
                Z = self.sensing_beam(G_sel, self.Pmax * 0.4)
                
                if W is not None:
                    sinrs = self.compute_sinr(H_true, W)
                    snrs = self.compute_snr(G_sel, Z)
                    power = np.sum(np.abs(W)**2) + np.sum(np.real(np.trace(Z)))
                    
                    success = (np.all(sinrs >= self.sinr_req) and
                             np.all(snrs >= self.snr_req) and
                             power <= self.Pmax)
                    
                    if not success:
                        n_sens += 1  # 不满足则增加AP
                else:
                    break
            
            if W is not None:
                results.append({
                    'sinr_min': float(np.min(sinrs)),
                    'snr_min': float(np.min(snrs)),
                    'power': power,
                    'n_sens': n_sens,
                    'success': success
                })
        
        if results:
            ok = sum(1 for r in results if r['success'])
            avg_sens = np.mean([r['n_sens'] for r in results])
            print(f"\n  成功率: {ok}/{len(results)} = {100*ok/len(results):.1f}%")
            print(f"  平均使用感知AP: {avg_sens:.1f}")
            print(f"  平均SINR: {np.mean([r['sinr_min'] for r in results]):.2f}dB")
            print(f"  平均SNR: {np.mean([r['snr_min'] for r in results]):.2f}dB")
            print(f"  平均功率: {np.mean([r['power'] for r in results]):.2f}W")
        
        print("\n完成!")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/simple_yu/.openclaw/workspace/ISAC_ML')
    
    from config.params_industry import *
    
    isac = CellFreeISACv28(
        M=M, K=K, P=P, Nt=Nt,
        Pmax=Pmax, sigma2=sigma2,
        sinr_req=sinr_req, snr_req=snr_req,
        error_var=error_var
    )
    isac.run(n_sens_start=2, n_sens_max=64, n_trials=30)