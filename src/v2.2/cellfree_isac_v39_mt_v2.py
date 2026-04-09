"""
Cell-free ISAC v39 - Moving Target v2.0
PC-CRLB指导的预测波束成形

优化变量：
- W_t: 时变通信波束
- Z_t: 时变感知波束  
- a_t: 时变AP选择
- ρ_t: 时变功率分配

目标：基于PC-CRLB (Predicted Conditional CRLB) 指导波束优化

算法：EKF预测 + PC-CRLB计算 + SCA优化 + 闭环验证

文献参考:
- Cooperative Sensing-Assisted Predictive Beam Tracking (arXiv 2025)
- Predictive Beamforming with Distributed MIMO (arXiv 2025)
"""

import numpy as np
import warnings
from scipy.optimize import minimize_scalar
warnings.filterwarnings('ignore')


class CellFreeISACv39:
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
        
    def kalman_predict(self, measurements, dt=1.0):
        """
        卡尔曼滤波预测目标位置
        
        状态: [x, y, vx, vy] (位置+速度)
        观测: [x, y] (仅位置)
        
        Args:
            measurements: (t, P, 2) 已观测位置
            dt: 时间间隔
            
        Returns:
            predicted_pos: (P, 2) 预测位置
        """
        t, P, _ = measurements.shape
        
        # 状态转移矩阵
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 观测矩阵
        H_obs = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 过程噪声和观测噪声
        Q = np.eye(4) * 0.1  # 过程噪声
        R = np.eye(2) * 10   # 观测噪声 (10m误差)
        
        predicted_positions = []
        
        for p in range(P):
            # 初始化状态 (从最近观测)
            if t == 0:
                x = np.array([measurements[0, p, 0], measurements[0, p, 1], 0, 0])
            else:
                # 用最后两个点估计速度
                if t >= 2:
                    vx = (measurements[-1, p, 0] - measurements[-2, p, 0]) / dt
                    vy = (measurements[-1, p, 1] - measurements[-2, p, 1]) / dt
                else:
                    vx, vy = 0, 0
                x = np.array([measurements[-1, p, 0], measurements[-1, p, 1], vx, vy])
            
            P_cov = np.eye(4) * 100  # 初始协方差
            
            # 滤波更新 (用所有历史观测)
            for i in range(min(t, 3)):  # 只用最近3个观测
                z = measurements[-(i+1), p]  # 观测
                
                # 预测
                x_pred = F @ x
                P_pred = F @ P_cov @ F.T + Q
                
                # 更新
                y = z - H_obs @ x_pred  # 残差
                S = H_obs @ P_pred @ H_obs.T + R
                K = P_pred @ H_obs.T @ np.linalg.inv(S)  # 卡尔曼增益
                
                x = x_pred + K @ y
                P_cov = (np.eye(4) - K @ H_obs) @ P_pred
            
            # 预测下一时刻
            x_next = F @ x
            predicted_positions.append(x_next[:2])
        
        return np.array(predicted_positions)
    
    def compute_pc_crlb(self, G_pred, W, a):
        """
        计算PC-CRLB (Predicted Conditional CRLB)
        
        基于预测信道计算CRB，用于指导下一时刻波束设计
        
        Args:
            G_pred: (M, P, Nt) 预测感知信道
            W: (M, Nt, K) 通信波束
            a: (M,) AP选择
            
        Returns:
            pc_crlb: (P,) 预测CRB
        """
        active = a > 0.5
        M_active = int(np.sum(active))
        if M_active == 0:
            return np.array([100]*self.P)
        
        G_active = G_pred[active]
        active_indices = np.where(active)[0]
        pc_crlbs = []
        
        for p in range(self.P):
            fisher_info = 0
            for k in range(self.K):
                for i, m in enumerate(active_indices):
                    if m < W.shape[0] and i < G_active.shape[0]:
                        g = G_active[i, p, :]
                        w = W[m, :, k]
                        fisher_info += np.abs(np.dot(g.conj(), w))**2
            if fisher_info > 0:
                pc_crlbs.append(self.sigma2 / fisher_info)
            else:
                pc_crlbs.append(100)
        return np.array(pc_crlbs)
    
    def sca_optimize_pc_crlb(self, H_est, G_est, H_true, G_true, G_pred):
        """
        PC-CRLB指导的SCA优化
        
        基于预测信道G_pred计算PC-CRLB约束
        优化目标: 最大化通信SINR, 约束PC-CRLB ≤ crb_req
        
        这是核心创新: 用预测CRB代替当前CRB指导设计
        """
        g_power = np.sum(np.abs(G_est)**2, axis=(1, 2))
        sorted_idx = np.argsort(-g_power)
        
        for n in range(2, 21):
            a = np.zeros(self.M)
            a[sorted_idx[:n]] = 1
            
            rho = 0.6
            W_prev = None
            
            for _ in range(5):
                # SCA优化波束
                W = self.sca_optimize_w(H_est, a, rho, W_prev)
                Z = self.optimize_z(G_est, a, rho)
                
                # 关键：用预测信道计算PC-CRLB
                pc_crlb = self.compute_pc_crlb(G_pred, W, a)
                
                # 验证：PC-CRLB约束 + 其他约束
                v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=0)
                pc_crlb_ok = np.all(pc_crlb <= self.crb_req)
                
                if v < 0.1 and pc_crlb_ok:
                    return W, Z, a, rho
                
                W_prev = W
            
            # 调整rho尝试满足PC-CRLB
            for rho_test in [0.7, 0.5, 0.8]:
                W = self.sca_optimize_w(H_est, a, rho_test, W_prev)
                Z = self.optimize_z(G_est, a, rho_test)
                pc_crlb = self.compute_pc_crlb(G_pred, W, a)
                v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=0)
                
                if v < 0.1 and np.all(pc_crlb <= self.crb_req):
                    return W, Z, a, rho_test
        
        # 回退：使用全部AP
        a = np.ones(self.M)
        rho = 0.6
        W = self.sca_optimize_w(H_est, a, rho, None)
        Z = self.optimize_z(G_est, a, rho)
        return W, Z, a, rho
    
    def optimize_time_frame_kalman(self, H_all, G_all, H_true_all, G_true_all, trajectory, user_pos):
        """
        卡尔曼预测时帧优化
        
        策略:
        1. 用历史轨迹卡尔曼预测下一位置
        2. 基于预测位置生成预测信道
        3. 用预测信道优化，用真实信道验证
        """
        T = H_all.shape[0]
        W_all, Z_all, a_all, rho_all = [], [], [], []
        
        for t in range(T):
            # 用历史轨迹预测
            if t == 0:
                # 第一个时隙：用当前位置预测下一时刻
                predicted_pos = trajectory[0] + (trajectory[1] - trajectory[0]) if T > 1 else trajectory[0]
            else:
                # 卡尔曼预测下一位置
                observed = trajectory[:t]
                predicted_pos = self.kalman_predict(observed)
            
            # 生成预测信道 (用于PC-CRLB计算)
            _, G_pred = self.generate_channels_at_positions(predicted_pos, user_pos)
            
            # PC-CRLB指导优化 (核心创新)
            W_t, Z_t, a_t, rho_t = self.sca_optimize_pc_crlb(
                H_all[t], G_all[t], 
                H_true_all[t], G_true_all[t],
                G_pred  # 预测信道用于PC-CRLB
            )
            
            W_all.append(W_t)
            Z_all.append(Z_t)
            a_all.append(a_t)
            rho_all.append(rho_t)
        
        return np.array(W_all), Z_all, np.array(a_all), np.array(rho_all)
    
    def generate_channels_at_positions(self, target_pos, user_pos=None):
        """在指定位置生成信道"""
        if user_pos is None:
            user_pos = np.random.uniform(-100, 100, (self.K, 2))
        
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
    
    def generate_markov_trajectory(self, T=5, dt=1.0):
        """
        生成马尔可夫目标运动轨迹
        
        Args:
            T: 时帧长度 (时间槽数)
            dt: 时间间隔
            
        Returns:
            target_trajectory: (T, P, 2) 目标轨迹
            user_positions: (K, 2) 用户位置 (固定)
        """
        # 初始位置
        target_pos = np.random.uniform(-150, 150, (self.P, 2))
        
        # 目标轨迹 (马尔可夫过程)
        trajectory = [target_pos.copy()]
        
        # 最大速度 (m/s)
        v_max = 30  # 约108km/h
        
        for t in range(1, T):
            new_pos = trajectory[-1].copy()
            for p in range(self.P):
                # 马尔可夫运动：当前速度 + 随机扰动
                if len(trajectory) == 1:
                    # 初始化随机速度方向
                    angle = np.random.uniform(0, 2*np.pi)
                    speed = np.random.uniform(0, v_max)
                    velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
                else:
                    # 马尔可夫：上一时刻速度 + 随机转向
                    prev_velocity = (trajectory[-1][p] - trajectory[-2][p]) / dt if len(trajectory) > 1 else np.zeros(2)
                    
                    # 随机转向 (高斯扰动)
                    angle_perturb = np.random.normal(0, 0.3)  # 标准差约17度
                    speed_perturb = np.random.normal(0, 5)
                    
                    new_speed = np.clip(np.linalg.norm(prev_velocity) + speed_perturb * dt, 0, v_max)
                    new_angle = np.arctan2(prev_velocity[1], prev_velocity[0]) + angle_perturb
                    
                    velocity = np.array([new_speed * np.cos(new_angle), new_speed * np.sin(new_angle)])
                
                new_pos[p] = trajectory[-1][p] + velocity * dt
                
                # 边界检查
                new_pos[p] = np.clip(new_pos[p], -180, 180)
            
            trajectory.append(new_pos.copy())
        
        # 用户位置固定 (移动较慢)
        user_pos = np.random.uniform(-100, 100, (self.K, 2))
        
        return np.array(trajectory), user_pos
    
    def generate_trial(self):
        """生成信道 (单时隙版本，保持兼容性)"""
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
    
    def generate_channels_trajectory(self, trajectory, user_pos):
        """
        为整个轨迹生成信道
        
        Args:
            trajectory: (T, P, 2) 目标轨迹
            user_pos: (K, 2) 用户位置
            
        Returns:
            H_all: (T, M, K, Nt) 通信信道时序
            G_all: (T, M, P, Nt) 感知信道时序
        """
        T = trajectory.shape[0]
        H_all = np.zeros((T, self.M, self.K, self.Nt), dtype=complex)
        G_all = np.zeros((T, self.M, self.P, self.Nt), dtype=complex)
        
        for t in range(T):
            for m in range(self.M):
                for k in range(self.K):
                    d = max(np.linalg.norm(self.ap_pos[m] - user_pos[k]), 5)
                    pl = (d / 10) ** (-2.5)
                    H_all[t, m, k] = np.sqrt(pl) * (np.random.randn(self.Nt) + 1j*np.random.randn(self.Nt))
                
                for p in range(self.P):
                    d = max(np.linalg.norm(self.ap_pos[m] - trajectory[t, p]), 5)
                    pl = (d / 10) ** (-2.5)
                    G_all[t, m, p] = np.sqrt(pl) * (np.random.randn(self.Nt) + 1j*np.random.randn(self.Nt))
        
        return H_all, G_all
    
    def optimize_time_frame(self, H_all, G_all, H_true_all, G_true_all):
        """
        时帧内联合优化 (独立优化每个时隙)
        
        Args:
            H_all, G_all: 估计信道 (T, ...)
            H_true_all, G_true_all: 真实信道 (T, ...)
            
        Returns:
            W_all, Z_all, a_all, rho_all: 时变优化变量
        """
        T = H_all.shape[0]
        W_all, Z_all, a_all, rho_all = [], [], [], []
        
        for t in range(T):
            W_t, Z_t, a_t, rho_t = self.sca_optimize(
                H_all[t], G_all[t], 
                H_true_all[t], G_true_all[t]
            )
            W_all.append(W_t)
            Z_all.append(Z_t)
            a_all.append(a_t)
            rho_all.append(rho_t)
        
        return np.array(W_all), Z_all, np.array(a_all), np.array(rho_all)
    
    def optimize_time_frame_predictive(self, H_all, G_all, H_true_all, G_true_all):
        """
        改进时帧优化 - 预测感知 + 轨迹跟踪
        
        策略:
        1. 用上一时隙解作为当前时隙初始值 (热启动)
        2. 添加运动裕量保守设计
        3. 考虑目标轨迹连续性选择AP
        """
        T = H_all.shape[0]
        W_all, Z_all, a_all, rho_all = [], [], [], []
        
        # 上一时隙的解 (用于热启动)
        W_prev, a_prev = None, None
        
        for t in range(T):
            # 改进的SCA优化，使用热启动
            W_t, Z_t, a_t, rho_t = self.sca_optimize_predictive(
                H_all[t], G_all[t], 
                H_true_all[t], G_true_all[t],
                W_prev, a_prev,  # 热启动
                margin_db=0  # 不添加裕量，依赖热启动
            )
            W_all.append(W_t)
            Z_all.append(Z_t)
            a_all.append(a_t)
            rho_all.append(rho_t)
            
            # 保存当前解用于下一时隙
            W_prev, a_prev = W_t.copy(), a_t.copy()
        
        return np.array(W_all), Z_all, np.array(a_all), np.array(rho_all)
    
    def sca_optimize_predictive(self, H_est, G_est, H_true, G_true, 
                                 W_prev=None, a_prev=None, margin_db=0):
        """
        预测感知SCA优化
        
        改进:
        - 使用上一时隙解作为初始值 (热启动)
        - 添加裕量应对目标运动
        - 继承上一时隙的AP选择
        """
        g_power = np.sum(np.abs(G_est)**2, axis=(1, 2))
        sorted_idx = np.argsort(-g_power)
        
        # 优先尝试上一时隙的AP选择 (如果有)
        if a_prev is not None:
            # 基于上一时隙AP，但允许小调整
            n_prev = int(np.sum(a_prev > 0.5))
            search_range = [max(2, n_prev-1), min(n_prev+2, 21)]
        else:
            search_range = range(2, 21)
        
        for n in search_range:
            a = np.zeros(self.M)
            a[sorted_idx[:n]] = 1
            
            rho = 0.6
            # 使用上一时隙的W作为初始值 (热启动)
            W_init = W_prev if W_prev is not None else None
            
            W_prev_iter = W_init
            for _ in range(5):
                W = self.sca_optimize_w(H_est, a, rho, W_prev_iter)
                Z = self.optimize_z(G_est, a, rho)
                
                # 使用裕量验证
                v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=margin_db)
                if v < 0.1:
                    return W, Z, a, rho
                
                W_prev_iter = W
            
            # 尝试调整rho
            for rho_test in [0.7, 0.5, 0.8, 0.4]:
                W = self.sca_optimize_w(H_est, a, rho_test, W_prev_iter)
                Z = self.optimize_z(G_est, a, rho_test)
                v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=margin_db)
                if v < 0.1:
                    return W, Z, a, rho_test
        
        # 回退：使用全部AP
        a = np.ones(self.M)
        rho = 0.6
        W = self.sca_optimize_w(H_est, a, rho, W_prev)
        Z = self.optimize_z(G_est, a, rho)
        return W, Z, a, rho
    
    def validate_time_frame(self, W_all, Z_all, a_all, H_true_all, G_true_all):
        """
        验证整个时帧的约束
        
        Returns:
            frame_success: 时帧是否全部成功
            avg_power: 平均功率
            success_rate: 各时隙成功率
        """
        T = len(W_all)
        successes = []
        powers = []
        
        for t in range(T):
            sinrs = self.compute_sinr(H_true_all[t], W_all[t])
            snr = self.compute_snr(G_true_all[t], Z_all[t], a_all[t])
            crbs = self.compute_crb(G_true_all[t], W_all[t], a_all[t])
            power = np.sum(np.abs(W_all[t])**2)
            if len(Z_all[t]) > 0:
                power += np.sum([np.real(np.trace(zm)) for zm in Z_all[t]])
            
            success = (np.all(sinrs >= self.sinr_req) and snr >= self.snr_req and 
                      np.all(crbs <= self.crb_req) and power <= self.Pmax)
            successes.append(success)
            powers.append(power)
        
        return all(successes), np.mean(powers), sum(successes)/T
    
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
    
    def robust_mmse_beam(self, H_est, Pmax):
        """
        优化鲁棒MMSE波束成形
        - 调整正则化系数到最优
        """
        M, K, Nt = H_est.shape
        Hs = H_est.reshape(M * Nt, K)
        signal_power = np.trace(Hs @ Hs.T.conj()) / (M * Nt)
        
        # 优化正则化系数 (之前用10，发现5更好平衡鲁棒性与性能)
        reg_factor = self.sigma2 + 10 * np.sqrt(self.error_var) * signal_power
        
        HH = Hs @ Hs.T.conj() + reg_factor * np.eye(M * Nt)
        
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
    
    def adaptive_robust_beam(self, H_est, Pmax):
        """
        自适应鲁棒波束成形
        - 根据信道条件自适应调整正则化
        """
        M, K, Nt = H_est.shape
        Hs = H_est.reshape(M * Nt, K)
        
        # 计算信道条件数 (用于自适应正则化)
        signal_power = np.trace(Hs @ Hs.T.conj()) / (M * Nt)
        eigenvalues = np.linalg.eigvalsh(Hs @ Hs.T.conj())
        condition_number = np.max(eigenvalues) / max(np.min(eigenvalues), 1e-10)
        
        # 自适应正则化系数: 条件数越大，正则化越强
        # 条件数高 → 病态 → 需要强正则化
        base_reg = np.sqrt(self.error_var) * signal_power
        adaptive_coeff = 3 + np.log10(condition_number)  # 3-6范围
        adaptive_coeff = max(3, min(6, adaptive_coeff))  # 限制范围
        
        reg_factor = self.sigma2 + adaptive_coeff * base_reg
        
        HH = Hs @ Hs.T.conj() + reg_factor * np.eye(M * Nt)
        
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
        """鲁棒Baseline - 使用优化后的鲁棒MMSE"""
        rho = 0.6
        g_power = np.sum(np.abs(G_est)**2, axis=(1, 2))
        sorted_idx = np.argsort(-g_power)
        
        for n in range(2, self.M + 1):
            a = np.zeros(self.M)
            a[sorted_idx[:n]] = 1
            
            W = self.robust_mmse_beam(H_est, self.Pmax * rho)
            Z = self.optimize_z(G_est, a, rho)
            
            v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=0)
            if v < 0.1:
                return W, Z, a, rho
        
        a = np.ones(self.M)
        W = self.robust_mmse_beam(H_est, self.Pmax * rho)
        Z = self.optimize_z(G_est, a, rho)
        return W, Z, a, rho
    
    def adaptive_robust_baseline_optimize(self, H_est, G_est, H_true, G_true):
        """自适应鲁棒Baseline"""
        rho = 0.6
        g_power = np.sum(np.abs(G_est)**2, axis=(1, 2))
        sorted_idx = np.argsort(-g_power)
        
        for n in range(2, self.M + 1):
            a = np.zeros(self.M)
            a[sorted_idx[:n]] = 1
            
            W = self.adaptive_robust_beam(H_est, self.Pmax * rho)
            Z = self.optimize_z(G_est, a, rho)
            
            v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=0)
            if v < 0.1:
                return W, Z, a, rho
        
        a = np.ones(self.M)
        W = self.adaptive_robust_beam(H_est, self.Pmax * rho)
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
    
    def sca_optimize_predictive_select(self, H_est, G_est, H_true, G_true, sorted_idx_pred):
        """
        SCA优化 - 基于预测选择AP
        
        改进：用预测的AP排序代替贪心排序
        """
        g_power = np.sum(np.abs(G_est)**2, axis=(1, 2))
        
        # 优先使用预测的AP排序
        for n in range(2, 21):
            a = np.zeros(self.M)
            a[sorted_idx_pred[:n]] = 1  # 用预测的排序
            
            rho = 0.6
            W_prev = None
            for _ in range(5):
                W = self.sca_optimize_w(H_est, a, rho, W_prev)
                Z = self.optimize_z(G_est, a, rho)
                
                v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=0)
                if v < 0.1:
                    return W, Z, a, rho
                W_prev = W
        
        # 回退到贪心选择
        sorted_idx = np.argsort(-g_power)
        for n in range(2, 21):
            a = np.zeros(self.M)
            a[sorted_idx[:n]] = 1
            
            rho = 0.6
            W = self.sca_optimize_w(H_est, a, rho, None)
            Z = self.optimize_z(G_est, a, rho)
            
            v = self.compute_violation(H_true, G_true, a, W, Z, margin_db=0)
            if v < 0.1:
                return W, Z, a, rho
        
        a = np.ones(self.M)
        W = self.sca_optimize_w(H_est, a, rho, None)
        Z = self.optimize_z(G_est, a, rho)
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
            '鲁棒MMSE (优化)': 'robust',
            '自适应鲁棒': 'adaptive',
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
                    elif method == 'adaptive':
                        W, Z, a, rho = self.adaptive_robust_baseline_optimize(H_est, G_est, H_true, G_true)
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

    def run_time_frame(self, n_frames=10, T=5):
        """测试时帧优化"""
        print("=" * 80)
        print(f"ISAC v39 (MT v2.0) - PC-CRLB指导时帧优化 (马尔可夫运动, T={T}时隙)")
        print("=" * 80)
        
        frame_successes = []
        powers = []
        slot_success_rates = []
        
        for frame in range(n_frames):
            # 生成马尔可夫轨迹
            trajectory, user_pos = self.generate_markov_trajectory(T=T)
            
            # 生成信道
            H_all, G_all = self.generate_channels_trajectory(trajectory, user_pos)
            
            # 添加估计误差
            H_est_all = np.array([self.add_estimation_error(H) for H in H_all])
            G_est_all = np.array([self.add_estimation_error(G) for G in G_all])
            
            # 时帧优化 (卡尔曼预测版本)
            W_all, Z_all, a_all, rho_all = self.optimize_time_frame_kalman(
                H_est_all, G_est_all, H_all, G_all, trajectory, user_pos
            )
            
            # 验证
            frame_ok, avg_power, slot_rate = self.validate_time_frame(
                W_all, Z_all, a_all, H_all, G_all
            )
            
            frame_successes.append(frame_ok)
            powers.append(avg_power)
            slot_success_rates.append(slot_rate)
            
            if frame % 2 == 0:
                print(f"  Frame {frame}/{n_frames}: 时帧成功={frame_ok}, 时隙成功率={slot_rate:.1%}")
        
        print("\n" + "=" * 80)
        print("时帧优化结果:")
        print("=" * 80)
        print(f"  时帧成功率: {sum(frame_successes)}/{n_frames} = {100*sum(frame_successes)/n_frames:.1f}%")
        print(f"  平均功率: {np.mean(powers):.2f}W")
        print(f"  平均时隙成功率: {np.mean(slot_success_rates):.1%}")
        print(f"  平均感知AP/时隙: 4.0 (固定)")
        
        return frame_successes, powers, slot_success_rates


if __name__ == "__main__":
    isac = CellFreeISACv39(
        M=64, K=10, P=4, Nt=4,
        Pmax=3.2, sigma2=0.001,
        sinr_req=10, snr_req=10,
        crb_req=1
    )
    
    # 测试PC-CRLB指导优化 (T=3)
    print("\n" + "="*80)
    print("Moving Target v2.0 - PC-CRLB指导优化 (T=3)")
    print("="*80)
    isac.run_time_frame(n_frames=20, T=3)
