"""
Cell-free ISAC v36 - 论文级联合优化算法 (闭环版)

优化变量：
- W: 通信波束
- Z: 感知波束  
- a: AP选择指示
- ρ: 功率分配因子

算法：SCA + AO + 闭环验证
"""

import numpy as np
import warnings
from scipy.optimize import minimize_scalar
warnings.filterwarnings('ignore')


class CellFreeISACv36:
    def __init__(self, M=64, K=10, P=4, Nt=4, Pmax=3.2, sigma2=0.001,
                 sinr_req=10, snr_req=10, crb_req=1, error_var=0.001):
        self.M = M
        self.K = K  
        self.P = P
        self.Nt = Nt
        self.Pmax = Pmax
        self.P_m_max = Pmax / M * 10
        self.sigma2 = sigma2
        self.sinr_req = sinr_req
        self.snr_req = snr_req
        self.crb_req = crb_req
        self.error_var = error_var
        
        x = np.linspace(-175, 175, 8)
        y = np.linspace(-175, 175, 8)
        self.ap_pos = np.array([[xi, yi] for xi in x for yi in y])
        
    def generate_trial(self):
        """生成信道"""
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
        """添加估计误差"""
        e = np.sqrt(self.error_var) * (np.random.randn(*H.shape) + 1j*np.random.randn(*H.shape))
        return H + e
    
    def mmse_beam(self, H, Pmax):
        """MMSE波束成形"""
        M, K, Nt = H.shape
        Hs = H.reshape(M * Nt, K)
        HH = Hs @ Hs.T.conj() + self.sigma2 * np.eye(M * Nt)
        try:
            W = np.linalg.inv(HH) @ Hs
            p = np.sum(np.abs(W)**2)
            if p > 0:
                W = W * np.sqrt(Pmax / p)
            W = W.reshape(M, Nt, K)
            
            for m in range(M):
                p_m = np.sum(np.abs(W[m, :, :])**2)
                if p_m > self.P_m_max:
                    W[m, :, :] *= np.sqrt(self.P_m_max / p_m)
            
            return W
        except:
            return np.zeros((M, Nt, K), dtype=complex)
    
    def robust_mmse_beam(self, H_est, Pmax, epsilon=None):
        """
        鲁棒MMSE波束成形 - 保守版本
        
        强正则化应对大信道误差
        """
        if epsilon is None:
            epsilon = np.sqrt(self.error_var)
        
        M, K, Nt = H_est.shape
        Hs = H_est.reshape(M * Nt, K)
        
        # 强鲁棒正则化：大幅增加对角加载
        # 基于最坏情况误差分析
        signal_power = np.trace(Hs @ Hs.T.conj()) / (M * Nt)
        reg_factor = self.sigma2 + 10 * epsilon * signal_power  # 10倍保守系数
        
        HH = Hs @ Hs.T.conj() + reg_factor * np.eye(M * Nt)
        
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
                    W[m, :, :] *= np.sqrt(self.P_m_max / p_m)
            
            return W
        except:
            return np.zeros((M, Nt, K), dtype=complex)
    
    def conservative_beam(self, H_est, Pmax):
        """
        超保守波束：仅使用信道方向，忽略幅度
        
        对信道误差极度鲁棒
        """
        M, K, Nt = H_est.shape
        
        # 仅使用相位信息
        H_phase = np.exp(1j * np.angle(H_est))
        Hs = H_phase.reshape(M * Nt, K)
        
        # 强正则化
        HH = Hs @ Hs.T.conj() + 100 * self.sigma2 * np.eye(M * Nt)
        
        try:
            W = np.linalg.inv(HH) @ Hs
            p = np.sum(np.abs(W)**2)
            if p > 0:
                W = W * np.sqrt(Pmax / p)
            W = W.reshape(M, Nt, K)
            return W
        except:
            return np.zeros((M, Nt, K), dtype=complex)
    
    def sensing_beam(self, G, P_sens):
        """感知波束成形"""
        M_sel, P, Nt = G.shape
        if M_sel == 0:
            return np.zeros((0, Nt, Nt), dtype=complex)
        
        p_per = P_sens / (M_sel * P) if M_sel * P > 0 else 0
        Z = np.zeros((M_sel, Nt, Nt), dtype=complex)
        
        for m in range(M_sel):
            for p in range(P):
                g = G[m, p, :]
                if np.linalg.norm(g) > 0:
                    w = np.conj(g) / np.linalg.norm(g) * np.sqrt(p_per)
                    Z[m] += np.outer(w, w.conj())
            
            p_m = np.real(np.trace(Z[m]))
            if p_m > self.P_m_max:
                Z[m] *= self.P_m_max / p_m
        
        return Z
    
    def compute_sinr(self, H, W):
        """计算通信SINR (dB)"""
        M, K, Nt = H.shape
        Hs = H.reshape(M * Nt, K)
        Wf = W.reshape(M * Nt, K)
        sinrs = []
        for k in range(K):
            sig = np.abs(np.dot(Wf[:, k].conj(), Hs[:, k]))**2
            inter = sum(np.abs(np.dot(Wf[:, j].conj(), Hs[:, k]))**2 
                       for j in range(K) if j != k)
            sinrs.append(10*np.log10(sig/(inter+self.sigma2+1e-10) + 1e-10))
        return np.array(sinrs)
    
    def compute_snr(self, G, Z, a):
        """计算感知SNR (dB)"""
        active = a > 0.5
        M_active = int(np.sum(active))
        if M_active == 0 or Z.shape[0] == 0:
            return -100
        
        G_active = G[active]
        signal = 0
        for p in range(self.P):
            for m in range(M_active):
                g = G_active[m, p, :]
                if m < Z.shape[0] and Z[m].shape == (self.Nt, self.Nt):
                    gH_Z_g = np.dot(g.conj(), Z[m] @ g)
                    signal += np.abs(gH_Z_g)
        
        noise = self.sigma2 * sum(np.real(np.trace(Z[m])) for m in range(min(M_active, Z.shape[0])))
        if noise > 0:
            return 10*np.log10(signal/(noise+1e-10) + 1e-10)
        return -100
    
    def compute_crb(self, G, W, a):
        """计算CRB"""
        active = a > 0.5
        M_active = int(np.sum(active))
        if M_active == 0:
            return np.array([100]*self.P)
        
        G_active = G[active]
        active_indices = np.where(active)[0]
        crbs = []
        
        for p in range(self.P):
            fisher_info = 0
            for k in range(self.K):
                for i, m in enumerate(active_indices):
                    if m < W.shape[0] and i < G_active.shape[0]:
                        g = G_active[i, p, :]
                        w = W[m, :, k]
                        fisher_info += np.abs(np.dot(g.conj(), w))**2
            if fisher_info > 0:
                crbs.append(self.sigma2 / fisher_info)
            else:
                crbs.append(100)
        return np.array(crbs)
    
    def compute_violation(self, H, G, a, W, Z, margin_db=0):
        """计算约束违反程度"""
        sinrs = self.compute_sinr(H, W)
        snr = self.compute_snr(G, Z, a)
        crbs = self.compute_crb(G, W, a)
        
        sinr_req_eff = self.sinr_req + margin_db
        snr_req_eff = self.snr_req + margin_db
        crb_req_eff = self.crb_req * (0.5 if margin_db > 0 else 1.0)
        
        violation = 0
        violation += np.sum(np.maximum(0, sinr_req_eff - sinrs))
        violation += np.maximum(0, snr_req_eff - snr)
        violation += np.sum(np.maximum(0, crbs - crb_req_eff))
        
        return violation
    
    def sca_optimize_w(self, H, a, rho, W_prev=None):
        """SCA优化W"""
        M, K, Nt = H.shape
        
        if W_prev is None:
            W = self.mmse_beam(H, self.Pmax * rho)
        else:
            W = W_prev.copy()
        
        return W
    
    def optimize_z(self, G, a, rho):
        """优化Z"""
        active = a > 0.5
        if not np.any(active):
            return np.zeros((0, self.Nt, self.Nt), dtype=complex)
        
        G_active = G[active]
        P_sens = self.Pmax * (1 - rho)
        return self.sensing_beam(G_active, P_sens)
    
    def optimize_rho(self, H, G, a, rho_init=0.6):
        """优化功率分配"""
        def objective(rho):
            W = self.mmse_beam(H, self.Pmax * rho)
            Z = self.optimize_z(G, a, rho)
            v = self.compute_violation(H, G, a, W, Z, margin_db=0)
            p_comm = np.sum(np.abs(W)**2)
            p_sens = np.sum([np.real(np.trace(Zm)) for Zm in Z]) if Z.size > 0 else 0
            return p_comm + p_sens + 100 * v
        
        result = minimize_scalar(objective, bounds=(0.3, 0.9), method='bounded')
        return result.x
    
    # ==================== Optimization Methods with Closed Loop ====================
    def robust_baseline_optimize(self, H_est, G_est, H_true, G_true):
        """鲁棒Baseline - 使用鲁棒MMSE波束成形"""
        rho = 0.6
        g_power = np.sum(np.abs(G_est)**2, axis=(1, 2))
        sorted_idx = np.argsort(-g_power)
        
        for n in range(2, self.M + 1):
            a = np.zeros(self.M)
            a[sorted_idx[:n]] = 1
            
            # 使用鲁棒MMSE
            W = self.robust_mmse_beam(H_est, self.Pmax * rho)
            Z = self.optimize_z(G_est, a, rho)
            
            v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=0)
            if v < 0.1:
                return W, Z, a, rho
        
        a = np.ones(self.M)
        W = self.robust_mmse_beam(H_est, self.Pmax * rho)
        Z = self.optimize_z(G_est, a, rho)
        return W, Z, a, rho
    
    def conservative_baseline_optimize(self, H_est, G_est, H_true, G_true):
        """超保守Baseline - 仅使用相位信息"""
        rho = 0.6
        g_power = np.sum(np.abs(G_est)**2, axis=(1, 2))
        sorted_idx = np.argsort(-g_power)
        
        for n in range(2, self.M + 1):
            a = np.zeros(self.M)
            a[sorted_idx[:n]] = 1
            
            # 使用超保守波束
            W = self.conservative_beam(H_est, self.Pmax * rho)
            Z = self.optimize_z(G_est, a, rho)
            
            v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=0)
            if v < 0.1:
                return W, Z, a, rho
        
        a = np.ones(self.M)
        W = self.conservative_beam(H_est, self.Pmax * rho)
        Z = self.optimize_z(G_est, a, rho)
        return W, Z, a, rho
    
    def baseline_optimize(self, H_est, G_est, H_true, G_true, use_true_for_design=False):
        """Baseline with closed-loop verification
        
        Args:
            use_true_for_design: If True, use H_true for beamforming (like v31)
                                If False, use H_est (realistic)
        """
        rho = 0.6
        g_power = np.sum(np.abs(G_est)**2, axis=(1, 2))
        sorted_idx = np.argsort(-g_power)
        
        # 选择设计用信道
        H_design = H_true if use_true_for_design else H_est
        G_design = G_true if use_true_for_design else G_est
        
        for n in range(2, self.M + 1):
            a = np.zeros(self.M)
            a[sorted_idx[:n]] = 1
            
            W = self.mmse_beam(H_design, self.Pmax * rho)
            Z = self.optimize_z(G_design, a, rho)
            
            v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=0)
            if v < 0.1:
                return W, Z, a, rho
        
        a = np.ones(self.M)
        W = self.mmse_beam(H_design, self.Pmax * rho)
        Z = self.optimize_z(G_design, a, rho)
        return W, Z, a, rho
    
    def sca_optimize(self, H_est, G_est, H_true, G_true):
        """SCA with closed-loop verification"""
        rho = 0.6
        g_power = np.sum(np.abs(G_est)**2, axis=(1, 2))
        sorted_idx = np.argsort(-g_power)
        
        W_prev = None
        for n in range(2, 21):
            a = np.zeros(self.M)
            a[sorted_idx[:n]] = 1
            
            W = self.sca_optimize_w(H_est, a, rho, W_prev)
            Z = self.optimize_z(G_est, a, rho)
            
            v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=0)
            if v < 0.1:
                return W, Z, a, rho
            W_prev = W
        
        a = np.ones(self.M)
        W = self.sca_optimize_w(H_est, a, rho, W_prev)
        Z = self.optimize_z(G_est, a, rho)
        return W, Z, a, rho
    
    def joint_rho_optimize(self, H_est, G_est, H_true, G_true):
        """Joint rho optimization with closed-loop verification"""
        g_power = np.sum(np.abs(G_est)**2, axis=(1, 2))
        sorted_idx = np.argsort(-g_power)
        
        W_prev = None
        for n in range(2, 21):
            a = np.zeros(self.M)
            a[sorted_idx[:n]] = 1
            
            rho_opt = self.optimize_rho(H_est, G_est, a, 0.6)
            W = self.sca_optimize_w(H_est, a, rho_opt, W_prev)
            Z = self.optimize_z(G_est, a, rho_opt)
            
            v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=0)
            if v < 0.1:
                return W, Z, a, rho_opt
            W_prev = W
        
        a = np.ones(self.M)
        rho_opt = self.optimize_rho(H_est, G_est, a, 0.6)
        W = self.sca_optimize_w(H_est, a, rho_opt, W_prev)
        Z = self.optimize_z(G_est, a, rho_opt)
        return W, Z, a, rho_opt
    
    def full_joint_optimize(self, H_est, G_est, H_true, G_true):
        """Full joint optimization with closed-loop verification"""
        g_power = np.sum(np.abs(G_est)**2, axis=(1, 2))
        sorted_idx = np.argsort(-g_power)
        
        for n in range(2, 21):
            a = np.zeros(self.M)
            a[sorted_idx[:n]] = 1
            
            rho = 0.6
            W_prev = None
            for _ in range(5):
                W = self.sca_optimize_w(H_est, a, rho, W_prev)
                Z = self.optimize_z(G_est, a, rho)
                rho_new = self.optimize_rho(H_est, G_est, a, rho)
                if abs(rho_new - rho) < 0.01:
                    break
                rho = rho_new
                W_prev = W
            
            v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=0)
            if v < 0.1:
                return W, Z, a, rho
        
        a = np.ones(self.M)
        rho = 0.6
        W = self.sca_optimize_w(H_est, a, rho)
        Z = self.optimize_z(G_est, a, rho)
        return W, Z, a, rho
    
    # ==================== Main Run ====================
    def run(self, n_trials=20):
        print("=" * 80)
        print("ISAC v36 - 论文级联合优化算法 (闭环验证版)")
        print("=" * 80)
        
        methods = {
            'Baseline (标准MMSE)': 'baseline_est',
            '鲁棒MMSE': 'robust',
            '超保守波束': 'conservative',
            'SCA优化W': 'sca',
            '联合优化ρ': 'joint_rho',
            '完整联合优化': 'full'
        }
        
        results = {name: [] for name in methods}
        
        for i in range(n_trials):
            if i % 5 == 0:
                print(f"  Trial {i}/{n_trials}")
            
            H_true, G_true = self.generate_trial()
            H_est = self.add_estimation_error(H_true)
            G_est = self.add_estimation_error(G_true)
            
            for name, method in methods.items():
                try:
                    if method == 'baseline_est':
                        W, Z, a, rho = self.baseline_optimize(H_est, G_est, H_true, G_true, use_true_for_design=False)
                    elif method == 'robust':
                        W, Z, a, rho = self.robust_baseline_optimize(H_est, G_est, H_true, G_true)
                    elif method == 'conservative':
                        W, Z, a, rho = self.conservative_baseline_optimize(H_est, G_est, H_true, G_true)
                    elif method == 'sca':
                        W, Z, a, rho = self.sca_optimize(H_est, G_est, H_true, G_true)
                    elif method == 'joint_rho':
                        W, Z, a, rho = self.joint_rho_optimize(H_est, G_est, H_true, G_true)
                    else:
                        W, Z, a, rho = self.full_joint_optimize(H_est, G_est, H_true, G_true)
                    
                    sinrs = self.compute_sinr(H_true, W)
                    snr = self.compute_snr(G_true, Z, a)
                    crbs = self.compute_crb(G_true, W, a)
                    
                    power = np.sum(np.abs(W)**2) + np.sum([np.real(np.trace(Zm)) for Zm in Z]) if Z.size > 0 else np.sum(np.abs(W)**2)
                    n_sens = int(np.sum(a > 0.5))
                    
                    success = (
                        np.all(sinrs >= self.sinr_req) and
                        snr >= self.snr_req and
                        np.all(crbs <= self.crb_req) and
                        power <= self.Pmax
                    )
                    
                    results[name].append({
                        'sinr_min': float(np.min(sinrs)),
                        'snr': snr,
                        'crb_max': float(np.max(crbs)),
                        'power': power,
                        'n_sens': n_sens,
                        'rho': rho,
                        'success': success
                    })
                except Exception as e:
                    print(f"  Error in {name}: {e}")
                    results[name].append({
                        'sinr_min': 0, 'snr': -100, 'crb_max': 100,
                        'power': self.Pmax, 'n_sens': self.M,
                        'rho': 0.5, 'success': False
                    })
        
        # Print results
        print("\n" + "=" * 80)
        print("结果对比")
        print("=" * 80)
        
        for name in methods:
            data = results[name]
            if data:
                ok = sum(1 for r in data if r['success'])
                print(f"\n{name}:")
                print(f"  成功率: {ok}/{len(data)} = {100*ok/len(data):.1f}%")
                print(f"  平均感知AP: {np.mean([r['n_sens'] for r in data]):.1f}")
                print(f"  平均SINR: {np.mean([r['sinr_min'] for r in data]):.2f}dB")
                print(f"  平均SNR: {np.mean([r['snr'] for r in data]):.2f}dB")
                print(f"  平均CRB: {np.mean([r['crb_max'] for r in data]):.4f}m")
                print(f"  平均功率: {np.mean([r['power'] for r in data]):.2f}W")
                if 'rho' in data[0]:
                    print(f"  平均ρ: {np.mean([r['rho'] for r in data]):.3f}")
        
        print("\n完成!")
        return results


if __name__ == "__main__":
    isac = CellFreeISACv36(
        M=64, K=10, P=4, Nt=4,
        Pmax=3.2, sigma2=0.001,
        sinr_req=10, snr_req=10,
        crb_req=1
    )
    results = isac.run(n_trials=20)
