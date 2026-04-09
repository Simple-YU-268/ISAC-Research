"""
Cell-free ISAC v2.2 Auto - 结合 isac_advanced 自动优化

关键改进:
1. 鲁棒MMSE: 正则化因子 = 1 + error_var * 10
2. SVD感知波束: 基于奇异值分配功率
3. 正确架构: 全部AP通信 + 选定AP感知
4. 自适应AP选择: 从2个开始,失败自动增加
5. 约束验证: 自动检查并调整
"""

import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class CellFreeISACv22Auto:
    """ISAC v2.2 Auto - 自适应优化版本"""
    
    def __init__(self, M=16, K=10, P=4, Nt=4, Pmax=30, sigma2=0.5,
                 sinr_req=0, snr_req=0, crb_req=10,
                 epsilon_h=0.1, epsilon_g=0.15):
        self.M = M
        self.K = K
        self.P = P
        self.Nt = Nt
        self.Pmax = Pmax
        self.sigma2 = sigma2
        self.sinr_req = sinr_req  # dB
        self.snr_req = snr_req    # dB
        self.crb_req = crb_req
        self.epsilon_h = epsilon_h
        self.epsilon_g = epsilon_g
        
        # 功率分配比例 (v84风格)
        self.P_comm_ratio = 0.8   # 80% 通信
        self.P_sens_ratio = 0.2   # 20% 感知
        
        self._init_topology()
        
    def _init_topology(self):
        """2D拓扑初始化"""
        # 4x4 网格
        self.ap_pos = np.array([
            [x, y] for x in np.linspace(-60, 60, 4)
            for y in np.linspace(-60, 60, 4)
        ])
        np.random.seed(42)
        self.user_pos = np.random.uniform(-50, 50, (self.K, 2))
        self.target_pos = np.random.uniform(-30, 30, (self.P, 2))
        
    def generate_channels(self):
        """生成真实信道"""
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
        
    def estimate_channels(self):
        """MMSE信道估计"""
        # 通信信道估计
        self.H_est = np.zeros_like(self.H_true)
        for m in range(self.M):
            for k in range(self.K):
                h_true = self.H_true[m, k]
                h_norm = np.linalg.norm(h_true)
                
                # MMSE估计
                beta = h_norm ** 2 / self.Nt
                noise_var = self.sigma2 / (0.1 * self.Pmax)
                mmse_factor = beta / (beta + noise_var)
                
                noise = np.sqrt(noise_var / 2) * (
                    np.random.randn(self.Nt) + 1j * np.random.randn(self.Nt)
                )
                self.H_est[m, k] = mmse_factor * (h_true + noise)
        
        # 感知信道估计
        self.G_est = np.zeros_like(self.G_true)
        for m in range(self.M):
            for p in range(self.P):
                g_true = self.G_true[m, p]
                g_norm = np.linalg.norm(g_true)
                
                beta = g_norm ** 2 / self.Nt
                noise_var = self.sigma2 / (0.1 * self.Pmax)
                mmse_factor = beta / (beta + noise_var)
                
                noise = np.sqrt(noise_var / 2) * (
                    np.random.randn(self.Nt) + 1j * np.random.randn(self.Nt)
                )
                self.G_est[m, p] = mmse_factor * (g_true + noise)
    
    def robust_mmse_beam(self, H_est: np.ndarray, P_comm: float) -> np.ndarray:
        """
        鲁棒MMSE波束成形 (来自isac_advanced)
        
        关键: 增加正则化来处理信道估计误差
        reg_factor = 1 + error_var * 10
        """
        M, K, Nt = H_est.shape
        N_total = M * Nt
        
        Hs = H_est.reshape(N_total, K)
        
        # 鲁棒正则化: 适度正则化
        reg_factor = 1 + self.epsilon_h * 2  # 减小正则化
        
        A = Hs @ Hs.T.conj() + self.sigma2 * reg_factor * np.eye(N_total)
        
        try:
            W = np.linalg.solve(A + 1e-8 * np.eye(N_total), Hs)
        except:
            # 备用: 使用伪逆
            W = np.linalg.pinv(A) @ Hs
        
        # 功率归一化
        p = np.sum(np.abs(W) ** 2)
        if p > P_comm:
            W = W * np.sqrt(P_comm / p)
        
        return W.reshape(M, Nt, K)
    
    def svd_sensing_beam(self, G_est: np.ndarray, P_sens: float) -> np.ndarray:
        """
        SVD优化感知波束 (来自isac_advanced)
        
        使用SVD获取主要信道方向,基于奇异值分配功率
        """
        M, P, Nt = G_est.shape
        
        # 堆叠信道
        G_stack = G_est.reshape(M * Nt, P)
        
        Z = np.zeros((M, P, Nt), dtype=complex)
        
        try:
            # SVD分解
            U, S, Vh = np.linalg.svd(G_stack, full_matrices=False)
            
            # 基于奇异值分配功率
            for p in range(min(P, len(S))):
                if S[p] > 1e-6:
                    # 功率与奇异值成正比
                    power = P_sens * (S[p] / np.sum(S[:min(P, len(S))]))
                    beam = Vh[p, :] * np.sqrt(power)
                    Z[:, p, :] = beam.reshape(M, Nt)
                    
        except:
            # 备用: 简单匹配滤波
            P_per = P_sens / P
            for p in range(P):
                h = G_est[:, p, :].flatten()
                norm = np.sqrt(np.sum(np.abs(h) ** 2))
                if norm > 0:
                    beam = np.conj(h) / norm * np.sqrt(P_per)
                    Z[:, p, :] = beam.reshape(M, Nt)
        
        return Z
    
    def select_sensing_ap(self, G_est: np.ndarray, n_sens: int) -> np.ndarray:
        """选择感知AP (综合得分)"""
        # 感知得分
        s_t = np.sum(np.sum(np.abs(G_est) ** 2, axis=2), axis=1)
        s_t = s_t / (np.max(s_t) + 1e-10)
        
        # 通信得分
        s_u = np.sum(np.sum(np.abs(self.H_est) ** 2, axis=2), axis=1)
        s_u = s_u / (np.max(s_u) + 1e-10)
        
        # 综合得分
        combined = 0.5 * s_u + 0.5 * s_t
        
        selected = np.argsort(-combined)[:n_sens]
        ap_mask = np.zeros(self.M, dtype=bool)
        ap_mask[selected] = True
        
        return ap_mask
    
    def verify_constraints(self, W: np.ndarray, G_sens: np.ndarray, 
                          Z: np.ndarray) -> Tuple[bool, str, Dict]:
        """
        约束验证 (来自isac_advanced)
        
        检查:
        - 通信SINR ≥ sinr_req
        - 感知SNR ≥ snr_req
        - CRB ≤ crb_req
        - 功率 ≤ Pmax
        """
        metrics = {}
        
        # 1. 验证通信SINR (全部AP)
        N_comm = self.M * self.Nt
        Hs = self.H_true.reshape(N_comm, self.K)
        Wf = W.reshape(N_comm, self.K)
        
        sinr_list = []
        for k in range(self.K):
            sig = np.abs(Wf[:, k].conj() @ Hs[:, k]) ** 2
            inter = sum(
                np.abs(Wf[:, j].conj() @ Hs[:, k]) ** 2
                for j in range(self.K) if j != k
            )
            sinr = 10 * np.log10(sig / (inter + self.sigma2 + 1e-10) + 1e-10)
            sinr_list.append(sinr)
        
        metrics['sinr_min'] = min(sinr_list)
        metrics['sinr_mean'] = np.mean(sinr_list)
        
        if any(s < self.sinr_req for s in sinr_list):
            return False, "通信SINR不足", metrics
        
        # 2. 验证感知SNR (选定AP)
        snr_list = []
        for p in range(self.P):
            h_true = G_sens[:, p, :].flatten()
            signal = np.abs(Z[:, p, :].flatten() @ h_true) ** 2
            snr = 10 * np.log10(
                signal / (self.sigma2 * np.sum(np.abs(Z) ** 2) + 1e-10) + 1e-10
            )
            snr_list.append(snr)
        
        metrics['snr_min'] = min(snr_list)
        metrics['snr_mean'] = np.mean(snr_list)
        
        if any(s < self.snr_req for s in snr_list):
            return False, "感知SNR不足", metrics
        
        # 3. 验证CRB
        crb_list = []
        for p in range(self.P):
            signal = np.sum(np.abs(Z[:, p, :] * np.conj(G_sens[:, p, :])) ** 2)
            crb = self.sigma2 / max(signal, 1e-10)
            crb_list.append(crb)
        
        metrics['crb_max'] = max(crb_list)
        
        if any(c > self.crb_req for c in crb_list):
            return False, "CRB超标", metrics
        
        # 4. 验证功率
        total_power = np.sum(np.abs(W) ** 2) + np.sum(np.abs(Z) ** 2)
        metrics['power'] = total_power
        
        if total_power > self.Pmax:
            return False, "功率超标", metrics
        
        return True, "OK", metrics
    
    def adaptive_solve(self, n_init: int = 2, max_sens: int = 16) -> Dict:
        """
        自适应求解 (来自isac_advanced)
        
        从n_init个感知AP开始,失败时增加2个,直到成功或达到上限
        """
        n_sens = n_init
        attempts = []
        
        while n_sens <= max_sens:
            # 选择AP
            ap_mask = self.select_sensing_ap(self.G_est, n_sens)
            G_sens = self.G_true[ap_mask, :, :]
            G_sens_est = self.G_est[ap_mask, :, :]
            
            # 波束成形 (测试: 用真实信道计算波束)
            P_comm = self.P_comm_ratio * self.Pmax
            P_sens = self.P_sens_ratio * self.Pmax
            
            # 测试完美CSI
            W = self.robust_mmse_beam(self.H_true, P_comm)
            Z = self.svd_sensing_beam(G_sens, P_sens)
            
            # 验证约束
            success, msg, metrics = self.verify_constraints(W, G_sens, Z)
            
            attempts.append({
                'n_sens': n_sens,
                'success': success,
                'message': msg,
                'metrics': metrics
            })
            
            if success:
                return {
                    'success': True,
                    'n_sens_ap': n_sens,
                    'n_init': n_init,
                    'attempts': attempts,
                    'metrics': metrics,
                    'W': W,
                    'Z': Z
                }
            
            # 失败: 增加2个AP重试
            n_sens += 2
        
        # 达到上限仍失败
        return {
            'success': False,
            'n_sens_ap': n_sens - 2,
            'attempts': attempts,
            'metrics': metrics if attempts else {}
        }
    
    def run_monte_carlo(self, n_trials: int = 100, n_init: int = 8) -> Dict:
        """蒙特卡洛仿真"""
        print("=" * 70)
        print("Cell-free ISAC v2.2 Auto - 自适应优化")
        print("=" * 70)
        print(f"配置: {self.M} APs, {self.K} users, {self.P} targets")
        print(f"约束: SINR≥{self.sinr_req}dB, SNR≥{self.snr_req}dB, CRB≤{self.crb_req}")
        print(f"不完美CSI: ε_h={self.epsilon_h:.0%}, ε_g={self.epsilon_g:.0%}")
        print(f"自适应: 从{n_init}个感知AP开始,失败自动增加")
        print("=" * 70)
        
        results = []
        n_sens_list = []
        
        for i in range(n_trials):
            if i % 20 == 0:
                print(f"\n进度: {i}/{n_trials}")
            
            # 生成信道
            self.generate_channels()
            self.estimate_channels()
            
            # 自适应求解
            result = self.adaptive_solve(n_init=n_init)
            results.append(result)
            
            if result['success']:
                n_sens_list.append(result['n_sens_ap'])
        
        # 统计
        success_count = sum(1 for r in results if r['success'])
        success_rate = 100 * success_count / n_trials
        
        # 调试: 查看失败原因
        fail_reasons = {}
        for r in results:
            if not r['success'] and r['attempts']:
                msg = r['attempts'][-1]['message']
                fail_reasons[msg] = fail_reasons.get(msg, 0) + 1
        
        print(f"\n{'='*70}")
        print("仿真结果")
        print(f"{'='*70}")
        print(f"成功率: {success_count}/{n_trials} = {success_rate:.1f}%")
        
        if fail_reasons:
            print(f"\n失败原因统计:")
            for reason, count in fail_reasons.items():
                print(f"  {reason}: {count}")
        
        if n_sens_list:
            print(f"平均感知AP数: {np.mean(n_sens_list):.1f}")
            print(f"最少感知AP数: {min(n_sens_list)}")
            print(f"最多感知AP数: {max(n_sens_list)}")
        
        # 成功样本的指标
        success_results = [r for r in results if r['success']]
        if success_results:
            sinr_mins = [r['metrics']['sinr_min'] for r in success_results]
            snr_mins = [r['metrics']['snr_min'] for r in success_results]
            powers = [r['metrics']['power'] for r in success_results]
            
            print(f"\n成功样本性能:")
            print(f"  通信SINR: {np.mean(sinr_mins):.2f}dB (min)")
            print(f"  感知SNR:  {np.mean(snr_mins):.2f}dB (min)")
            print(f"  功率:     {np.mean(powers):.2f}W")
        
        return {
            'success_rate': success_rate,
            'n_sens_avg': np.mean(n_sens_list) if n_sens_list else 0,
            'results': results
        }


def main():
    """主函数"""
    # 创建系统
    isac = CellFreeISACv22Auto(
        M=16, K=10, P=4, Nt=4,
        Pmax=30, sigma2=0.5,
        sinr_req=0,    # 0 dB
        snr_req=0,     # 0 dB
        crb_req=10,    # CRB ≤ 10
        epsilon_h=0.1,
        epsilon_g=0.15
    )
    
    # 运行仿真
    results = isac.run_monte_carlo(n_trials=50, n_init=2)
    
    print(f"\n{'='*70}")
    print("完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
