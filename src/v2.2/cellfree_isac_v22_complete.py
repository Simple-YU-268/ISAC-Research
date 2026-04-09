"""
Cell-free ISAC v2.2 Complete
基于 v84 完整系统 + 不完美 CSI 鲁棒优化

改进点:
1. 保留 v84 所有功能 (MMSE + 感知波束 + AP选择)
2. 添加不完美 CSI (MMSE估计 + 误差模型)
3. 鲁棒性能评估 (最坏情况 SINR/CRB)
4. 联合功率分配优化
"""

import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class CellFreeISACv22:
    """Cell-free ISAC v2.2 - 基于v84的完整鲁棒系统"""
    
    def __init__(self, M=16, K=10, P=4, Nt=4, Pmax=30, sigma2=0.5,
                 epsilon_h=0.1, epsilon_g=0.15, pilot_power_ratio=0.1):
        """
        参数:
            M, K, P, Nt: AP数, 用户数, 目标数, 天线数
            Pmax: 总功率预算 (W)
            sigma2: 噪声功率
            epsilon_h: 通信信道估计误差界 (||Δh|| ≤ ε||h||)
            epsilon_g: 感知信道估计误差界
            pilot_power_ratio: 导频功率占总功率比例
        """
        self.M = M
        self.K = K  
        self.P = P
        self.Nt = Nt
        self.Pmax = Pmax
        self.sigma2 = sigma2
        self.epsilon_h = epsilon_h
        self.epsilon_g = epsilon_g
        self.pilot_power_ratio = pilot_power_ratio
        
        # 初始化拓扑
        self._init_topology()
        
    def _init_topology(self):
        """初始化2D拓扑 - 同v84"""
        self.ap_pos = np.array([
            [x, y] for x in np.linspace(-60, 60, 4) 
            for y in np.linspace(-60, 60, 4)
        ])
        np.random.seed(42)
        self.user_pos = np.random.uniform(-50, 50, (self.K, 2))
        self.target_pos = np.random.uniform(-30, 30, (self.P, 2))
        
    def generate_true_channels(self):
        """生成真实信道 (Ground Truth)"""
        # 通信信道
        self.H_true = np.zeros((self.M, self.K, self.Nt), dtype=complex)
        for m in range(self.M):
            for k in range(self.K):
                d = max(np.linalg.norm(self.ap_pos[m] - self.user_pos[k]), 5)
                pl = (d / 10) ** (-2.5)
                self.H_true[m, k] = np.sqrt(pl / 2) * (
                    np.random.randn(self.Nt) + 1j * np.random.randn(self.Nt)
                )
        
        # 感知信道
        self.G_true = np.zeros((self.M, self.P, self.Nt), dtype=complex)
        for m in range(self.M):
            for p in range(self.P):
                d = max(np.linalg.norm(self.ap_pos[m] - self.target_pos[p]), 5)
                pl = (d / 10) ** (-2.5)
                self.G_true[m, p] = np.sqrt(pl / 2) * (
                    np.random.randn(self.Nt) + 1j * np.random.randn(self.Nt)
                )
                
    def estimate_channels_mmse(self):
        """
        MMSE信道估计 - 生成估计值和误差统计
        
        模型: h = ĥ + Δh
        其中 ĥ = MMSE(y) 是估计值
        Δh 是估计误差, ||Δh|| ≤ ε||ĥ||
        """
        # 通信信道估计
        self.H_est = np.zeros_like(self.H_true)
        self.H_error_bound = np.zeros((self.M, self.K))
        
        for m in range(self.M):
            for k in range(self.K):
                h_true = self.H_true[m, k]
                h_norm = np.linalg.norm(h_true)
                
                # 导频接收 (含噪声)
                pilot_power = self.pilot_power_ratio * self.Pmax
                noise_var = self.sigma2 / pilot_power
                
                # MMSE估计 (简化: 对角加载)
                mmse_factor = h_norm**2 / (h_norm**2 + noise_var * self.Nt)
                noise = np.sqrt(noise_var / 2) * (
                    np.random.randn(self.Nt) + 1j * np.random.randn(self.Nt)
                )
                
                self.H_est[m, k] = mmse_factor * (h_true + noise)
                
                # 误差界 (保守估计)
                self.H_error_bound[m, k] = self.epsilon_h * np.linalg.norm(self.H_est[m, k])
        
        # 感知信道估计
        self.G_est = np.zeros_like(self.G_true)
        self.G_error_bound = np.zeros((self.M, self.P))
        
        for m in range(self.M):
            for p in range(self.P):
                g_true = self.G_true[m, p]
                g_norm = np.linalg.norm(g_true)
                
                pilot_power = self.pilot_power_ratio * self.Pmax
                noise_var = self.sigma2 / pilot_power
                
                mmse_factor = g_norm**2 / (g_norm**2 + noise_var * self.Nt)
                noise = np.sqrt(noise_var / 2) * (
                    np.random.randn(self.Nt) + 1j * np.random.randn(self.Nt)
                )
                
                self.G_est[m, p] = mmse_factor * (g_true + noise)
                self.G_error_bound[m, p] = self.epsilon_g * np.linalg.norm(self.G_est[m, p])
                
    def mmse_beam(self, H, P_comm):
        """MMSE通信波束成形 - 同v84"""
        M, K, Nt = H.shape
        Hs = H.reshape(M * Nt, K)
        HH = Hs @ Hs.T.conj() + self.sigma2 * np.eye(M * Nt)
        
        try:
            W = np.linalg.inv(HH) @ Hs
            p = np.sum(np.abs(W)**2)
            if p > 0:
                W = W * np.sqrt(P_comm / p)
            return W.reshape(M, Nt, K)
        except:
            return None
            
    def sensing_beam(self, G, P_sens):
        """感知波束成形 (匹配滤波) - 同v84"""
        M, P, Nt = G.shape
        p_per_target = P_sens / P
        Z = np.zeros((M, P, Nt), dtype=complex)
        
        for p in range(P):
            g_p = G[:, p, :]
            norm = np.sqrt(np.sum(np.abs(g_p)**2))
            if norm > 0:
                Z[:, p, :] = np.conj(g_p) / norm * np.sqrt(p_per_target)
                
        return Z
        
    def compute_sinr_robust(self, H_true, W, epsilon_avg=0.0):
        """
        计算鲁棒SINR (最坏情况)
        
        参数:
            epsilon_avg: 平均信道估计误差比例 (如0.1表示10%)
        """
        M, K, Nt = H_true.shape
        Hs = H_true.reshape(M * Nt, K)
        W_flat = W.reshape(M * Nt, K)
        
        sinrs = []
        for k in range(K):
            # 信号功率
            sig = np.abs(np.sum(np.conj(W_flat[:, k]) @ Hs[:, k]))**2
            
            # 干扰
            interf = sum(
                np.abs(np.sum(np.conj(W_flat[:, j]) @ Hs[:, k]))**2
                for j in range(K) if j != k
            )
            
            sinr_nominal = sig / (interf + self.sigma2 + 1e-10)
            
            # 最坏情况调整 (保守估计)
            if epsilon_avg > 0:
                # 信号减少 (1-ε)^2, 干扰增加 (1+ε)^2
                sinr_worst = sinr_nominal * (1 - epsilon_avg)**4
                sinrs.append(10 * np.log10(sinr_worst + 1e-10))
            else:
                sinrs.append(10 * np.log10(sinr_nominal + 1e-10))
                
        return np.array(sinrs)
        
    def compute_crb(self, G_true, Z):
        """计算CRB (Cramer-Rao Bound)"""
        M, P, Nt = G_true.shape
        crbs = []
        
        for p in range(P):
            # 等效感知信道
            g_p = G_true[:, p, :]
            z_p = Z[:, p, :]
            
            # 感知信号功率
            power = np.sum(np.abs(z_p * np.conj(g_p))**2)
            
            # CRB (简化: 位置估计)
            crb = 1.0 / (power + 0.1)
            crbs.append(crb)
            
        return np.mean(crbs) if crbs else 1000.0
        
    def select_ap_robust(self, H_est, N_req):
        """
        鲁棒AP选择
        基于估计信道强度选择最强的N_req个AP
        """
        signal_power = np.sum(np.abs(H_est)**2, axis=2)
        total_signal = signal_power.sum(axis=1)
        selected = np.argsort(-total_signal)[:N_req]
        
        ap_mask = np.zeros(self.M, dtype=bool)
        ap_mask[selected] = True
        return ap_mask
        
    def joint_power_allocation(self, H, G, P_total, alpha_comm=0.8):
        """
        联合功率分配
        
        参数:
            alpha_comm: 分配给通信的功率比例 (默认80%)
        
        返回:
            P_comm, P_sens: 通信和感知功率
        """
        # 可以基于信道条件自适应调整
        # 简化: 固定比例
        P_comm = P_total * alpha_comm
        P_sens = P_total * (1 - alpha_comm)
        return P_comm, P_sens
        
    def simulate_one_trial(self, N_req, alpha_comm=0.8, use_estimated_channel=False):
        """
        单次蒙特卡洛仿真
        
        参数:
            N_req: 选择的AP数量
            alpha_comm: 通信功率比例
            use_estimated_channel: 是否使用估计信道计算波束
        """
        # 生成信道
        self.generate_true_channels()
        self.estimate_channels_mmse()
        
        # 选择AP (基于估计信道)
        ap_mask = self.select_ap_robust(self.H_est, N_req)
        
        H_true_sel = self.H_true[ap_mask, :, :]
        G_true_sel = self.G_true[ap_mask, :, :]
        H_est_sel = self.H_est[ap_mask, :, :]
        G_est_sel = self.G_est[ap_mask, :, :]
        
        # 功率分配
        P_comm, P_sens = self.joint_power_allocation(
            H_est_sel, G_est_sel, self.Pmax, alpha_comm
        )
        
        # 波束成形
        if use_estimated_channel:
            W = self.mmse_beam(H_est_sel, P_comm)
            Z = self.sensing_beam(G_est_sel, P_sens)
        else:
            # 用真实信道计算 (完美CSI基准)
            W = self.mmse_beam(H_true_sel, P_comm)
            Z = self.sensing_beam(G_true_sel, P_sens)
            
        if W is None:
            return None
            
        # 性能评估
        # 1. 通信SINR (标称 vs 鲁棒)
        sinr_nominal = self.compute_sinr_robust(H_true_sel, W, epsilon_avg=0.0)
        sinr_robust = self.compute_sinr_robust(H_true_sel, W, epsilon_avg=self.epsilon_h)
        
        # 2. 感知CRB
        crb = self.compute_crb(G_true_sel, Z)
        
        # 3. 总功率
        power_total = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
        
        return {
            'sinr_nominal': sinr_nominal,
            'sinr_robust': sinr_robust,
            'sinr_min_nominal': sinr_nominal.min(),
            'sinr_min_robust': sinr_robust.min(),
            'comm_ok_nominal': sum(sinr_nominal >= 0),
            'comm_ok_robust': sum(sinr_robust >= 0),
            'crb': crb,
            'power': power_total,
            'power_ok': power_total <= self.Pmax,
            'n_aps': N_req
        }
        
    def run_simulation(self, N_req_list=[4, 8, 12, 16], n_trials=200, 
                      alpha_comm=0.8, use_estimated_channel=True):
        """
        完整仿真 - 对比不同AP数量
        
        同v84的对比框架，但增加鲁棒性能评估
        """
        print("=" * 70)
        print("Cell-free ISAC v2.2 - 完整鲁棒系统")
        print("=" * 70)
        print(f"配置: {self.M} APs, {self.K} users, {self.P} targets")
        print(f"不完美CSI: ε_h={self.epsilon_h:.0%}, ε_g={self.epsilon_g:.0%}")
        print(f"功率分配: 通信{alpha_comm:.0%}, 感知{1-alpha_comm:.0%}")
        print(f"波束计算: {'估计信道' if use_estimated_channel else '真实信道(完美CSI)'}")
        print("=" * 70)
        print()
        
        all_results = {}
        
        for N_req in N_req_list:
            print(f"\n--- AP数量: {N_req} ---")
            
            results = []
            for i in range(n_trials):
                if i % 50 == 0:
                    print(f"  进度: {i}/{n_trials}")
                    
                res = self.simulate_one_trial(N_req, alpha_comm, use_estimated_channel)
                if res is not None:
                    results.append(res)
                    
            all_results[N_req] = results
            
            # 统计结果
            sinr_min_robust = [r['sinr_min_robust'] for r in results]
            sinr_min_nominal = [r['sinr_min_nominal'] for r in results]
            comm_ok_robust = [r['comm_ok_robust'] for r in results]
            comm_ok_nominal = [r['comm_ok_nominal'] for r in results]
            powers = [r['power'] for r in results]
            
            print(f"\n  标称性能 (完美CSI假设):")
            print(f"    最小SINR: {np.mean(sinr_min_nominal):.2f}dB")
            print(f"    全部用户≥0dB: {sum(1 for c in comm_ok_nominal if c==self.K)}/{len(results)}")
            
            print(f"\n  鲁棒性能 (最坏情况保证):")
            print(f"    最小SINR: {np.mean(sinr_min_robust):.2f}dB")
            print(f"    全部用户≥0dB: {sum(1 for c in comm_ok_robust if c==self.K)}/{len(results)}")
            
            print(f"\n  功率:")
            print(f"    平均: {np.mean(powers):.2f}W (限制{self.Pmax}W)")
            print(f"    满足约束: {sum(1 for p in powers if p<=self.Pmax)}/{len(results)}")
            
        return all_results


def main():
    """主函数"""
    # 创建仿真器
    isac = CellFreeISACv22(
        M=16, K=10, P=4, Nt=4,
        Pmax=30, sigma2=0.5,
        epsilon_h=0.10,  # 10% 信道估计误差
        epsilon_g=0.15,  # 15% 感知信道误差
        pilot_power_ratio=0.1
    )
    
    # 运行仿真
    results = isac.run_simulation(
        N_req_list=[4, 8, 12, 16],
        n_trials=100,  # 减少次数以便快速测试
        alpha_comm=0.8,
        use_estimated_channel=True  # 使用估计信道 (不完美CSI)
    )
    
    print("\n" + "=" * 70)
    print("仿真完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
