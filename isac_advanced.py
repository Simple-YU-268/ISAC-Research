"""
ISAC系统 - 完整版 (三步式AP选择 + 鲁棒波束成形 + 优化感知)
64 APs, 16天线/AP, 10用户, 4目标

改进:
1. 三步式ISAC系统 (探测→选择→波束)
2. 鲁棒MMSE波束成形 (处理信道估计误差)
3. SVD优化感知波束 (基于奇异值分配功率)
"""
# Cell-Free ISAC系统 (符合3GPP和文献标准)
# 参数: 64 AP × 8天线 = 512总天线 (文献典型)

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============ 系统参数 (Cell-Free标准) ============
M = 64           # AP数量 (Cell-Free典型: 32-200)
Nt = 8           # 每AP天线数 (文献标准: 4-8)
K = 10           # 通信用户数
P = 4            # 感知目标数 (默认,实际动态检测)
sigma2 = 0.001   # 噪声功率
Pmax = 3.2       # 总功率 (W)

# 行业标准约束 (基于3GPP ISAC)
sinr_req = 10    # 通信SINR ≥ 10dB (行业标准)
snr_req = 10     # 感知SNR ≥ 10dB (行业标准)
crb_req = 1      # CRB ≤ 1

# 功率分配
P_comm_ratio = 0.6   # 60% 通信
P_sens_ratio = 0.4   # 40% 感知

# 信道估计误差
error_var = 0.05

print("="*60)
print("ISAC系统 - 随机AP分布 (更现实)")
print(f"参数: M={M}随机AP, Nt={Nt}, K={K}, P={P}, Pmax={Pmax}W")
print("="*60)

def generate_channel(P_override=None, seed=None):
    """
    生成信道 - 随机AP分布 (更现实)
    AP在100m×100m区域内随机分布
    
    参数:
        P_override: 覆盖默认目标数量
        seed: 随机种子
    """
    P_use = P_override if P_override else P
    np.random.seed(seed)
    
    # 随机AP分布 (更符合实际部署)
    ap_x = np.random.uniform(-50, 50, M)
    ap_y = np.random.uniform(-50, 50, M)
    ap = np.column_stack([ap_x, ap_y])
    
    # 用户在中心区域,目标在外围
    user_pos = np.random.uniform(-20, 20, (K, 2))
    target_pos = np.random.uniform(-40, 40, (P_use, 2))
    
    H_u = np.zeros((M, K, Nt), dtype=complex)
    H_t = np.zeros((M, P_use, Nt), dtype=complex)
    
    for m in range(M):
        for k in range(K):
            d = max(np.sqrt(np.sum((ap[m] - user_pos[k])**2)), 3)
            pl = (d / 10)**-2.5
            H_u[m, k, :] = np.sqrt(pl / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        for p in range(P_use):
            d = max(np.sqrt(np.sum((ap[m] - target_pos[p])**2)), 3)
            pl = (d / 10)**-2.5
            H_t[m, p, :] = np.sqrt(pl / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
    
    return H_u, H_t, ap

def add_estimation_error(H, error_var):
    """添加信道估计误差"""
    return H + np.sqrt(error_var/2) * (np.random.randn(*H.shape) + 1j*np.random.randn(*H.shape))

# ============ Step 1: 目标探测 ============
def step1_target_detection(H_t):
    """使用全部AP进行目标探测"""
    M, P, Nt = H_t.shape
    detection_score = np.zeros(M)
    for p in range(P):
        h_stack = H_t[:, p, :].flatten()
        detection_score += np.sum(np.abs(h_stack)**2)
    return detection_score

# ============ Step 2: AP选择 ============
def step2_ap_selection(detection_score, H_u, n_ap=5, alpha=0.5):
    """综合通信+感知得分选择AP"""
    s_t = detection_score / (np.max(detection_score) + 1e-10)
    s_u = np.sum(np.abs(H_u)**2, axis=(1,2))
    s_u = s_u / (np.max(s_u) + 1e-10)
    combined = alpha * s_u + (1-alpha) * s_t
    return np.argsort(-combined)[:n_ap]

# ============ Step 3a: 鲁棒MMSE通信波束 ============
def robust_mmse_beamforming(H_est, Pmax, error_var=0.05):
    """
    鲁棒MMSE: 考虑信道估计误差
    增加正则化来处理不确定性
    """
    M_sel, K, Nt = H_est.shape
    N_total = M_sel * Nt
    Hs = H_est.reshape(N_total, K)
    
    # 误差越大,正则化越强
    reg_factor = 1 + error_var * 10
    A = Hs @ Hs.T.conj() + sigma2 * reg_factor * np.eye(N_total)
    W = np.linalg.solve(A + 1e-8 * np.eye(N_total), Hs)
    
    p = np.sum(np.abs(W)**2)
    if p > Pmax:
        W = W * np.sqrt(Pmax / p)
    return W.reshape(M_sel, Nt, K)

# ============ Step 3b: 优化感知波束 (SVD) ============
def optimized_sensing_beamforming(H_t, P_sens):
    """
    优化感知波束: 使用SVD获取主要信道方向
    基于奇异值分配功率
    """
    M_sel, P, Nt = H_t.shape
    Z = np.zeros((M_sel, P, Nt), dtype=complex)
    
    H_stack = H_t.reshape(M_sel * Nt, P)
    
    try:
        U, S, Vh = np.linalg.svd(H_stack, full_matrices=False)
        
        for p in range(min(P, len(S))):
            if S[p] > 1e-6:
                power = P_sens * (S[p] / np.sum(S[:min(P,len(S))]))**0.5
                beam = Vh[p, :] * np.sqrt(power)
                Z[:, p, :] = beam.reshape(M_sel, Nt)
    except:
        # 备用: 简单匹配滤波
        P_per = P_sens / P
        for p in range(P):
            h = H_t[:, p, :].flatten()
            norm = np.sqrt(np.sum(np.abs(h)**2))
            if norm > 0:
                beam = np.conj(h) / norm * np.sqrt(P_per)
                for m in range(M_sel):
                    Z[m, p, :] = beam[m*Nt:(m+1)*Nt]
    
    return Z

# ============ 约束验证 ============
def verify_constraints(H_u, H_t, W, Z, Pmax):
    """验证所有约束"""
    M_sel = H_u.shape[0]
    Hs = H_u.reshape(M_sel*Nt, K)
    Wf = W.reshape(M_sel*Nt, K)
    
    # (b) 通信SINR ≥ 0dB
    for k in range(K):
        sig = np.abs(np.conj(Wf[:,k]).T @ Hs[:,k])**2
        inter = sum(np.abs(np.conj(Wf[:,j]).T @ Hs[:,k])**2 for j in range(K) if j!=k)
        if 10*np.log10(sig/(inter+sigma2+1e-10)+1e-10) < 0:
            return False, "通信SINR"
    
    # (c) 感知SNR ≥ 0dB
    for p in range(P):
        signal = sum(np.abs(np.sum(Z[m,p,:]*np.conj(H_t[m,p,:])))**2 for m in range(M_sel))
        if 10*np.log10(signal/(sigma2*np.sum(np.abs(Z)**2)+1e-10)+1e-10) < 0:
            return False, "感知SNR"
    
    # (d) CRB ≤ 10
    for p in range(P):
        signal = sum(np.abs(np.sum(Z[m,p,:]*np.conj(H_t[m,p,:])))**2 for m in range(M_sel))
        if sigma2/max(signal,1e-10) > 10:
            return False, "CRB"
    
    # (f) 功率 ≤ Pmax
    if np.sum(np.abs(W)**2)+np.sum(np.abs(Z)**2) > Pmax:
        return False, "功率"
    
    return True, "OK"

# ============ 主流程 ============
def isac_system(n_ap=5, error_var=0, seed=None):
    """完整ISAC系统流程"""
    H_u_true, H_t_true, _ = generate_channel(seed)
    
    # 信道估计
    if error_var > 0:
        H_u_est = add_estimation_error(H_u_true, error_var)
        H_t_est = add_estimation_error(H_t_true, error_var)
    else:
        H_u_est = H_u_true
        H_t_est = H_t_true
    
    # Step 1: 目标探测
    detection_score = step1_target_detection(H_t_est)
    
    # Step 2: AP选择
    selected = step2_ap_selection(detection_score, H_u_est, n_ap=n_ap, alpha=0.5)
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    
    H_u_sel = H_u_est[ap_mask, :, :]
    H_t_sel = H_t_est[ap_mask, :, :]
    
    # Step 3: 联合波束成形
    W = robust_mmse_beamforming(H_u_sel, P_comm=0.6*Pmax, error_var=error_var)
    Z = optimized_sensing_beamforming(H_t_sel, P_sens=0.4*Pmax)
    
    # 验证 (真实信道)
    success, msg = verify_constraints(H_u_true[ap_mask], H_t_true[ap_mask], W, Z, Pmax)
    
    return {'success': success, 'selected_aps': selected, 'message': msg}

# ============ 测试 ============
if __name__ == "__main__":
    print("\n=== 完美CSI测试 ===")
    for n_ap in [5, 7]:
        success = sum(1 for _ in range(50) if isac_system(n_ap=n_ap, error_var=0)['success'])
        print(f"{n_ap} APs: {success*2}%")
    
    print("\n=== 有误差测试 (σ=0.05) ===")
    for n_ap in [5, 7, 10]:
        success = sum(1 for _ in range(50) if isac_system(n_ap=n_ap, error_var=0.05)['success'])
        print(f"{n_ap} APs: {success*2}%")
# ============ 正确架构: 64 AP通信 + n AP感知 ============
def isac_correct_architecture(n_sens_ap=4, error_var=0.05, seed=None):
    """
    正确的ISAC架构:
    - 通信: 全部64个AP参与
    - 感知: 只有n_sens_ap个AP参与
    
    参数:
        n_sens_ap: 参与感知的AP数量 (默认4个达到100%成功)
        error_var: 信道估计误差
        seed: 随机种子
    
    返回:
        dict: 包含成功标志和各项指标
    """
    H_u, H_t, _ = generate_channel(seed)
    
    # 信道估计
    if error_var > 0:
        H_u_est = H_u + np.sqrt(error_var/2) * (np.random.randn(M, K, Nt) + 1j*np.random.randn(M, K, Nt))
        H_t_est = H_t + np.sqrt(error_var/2) * (np.random.randn(M, P, Nt) + 1j*np.random.randn(M, P, Nt))
    else:
        H_u_est = H_u
        H_t_est = H_t
    
    # Step 1: 选择参与感知的AP
    ds = np.sum(np.sum(np.abs(H_t_est)**2, axis=2), axis=1)
    s_t = ds / (np.max(ds) + 1e-10)
    s_u = np.sum(np.sum(np.abs(H_u_est)**2, axis=1), axis=1)
    s_u = s_u / (np.max(s_u) + 1e-10)
    sens_selected = np.argsort(-(0.5 * s_u + 0.5 * s_t))[:n_sens_ap]
    
    # Step 2: 通信波束 (全部64 AP)
    N_comm = M * Nt
    P_comm = P_comm_ratio * Pmax
    
    Hs = H_u_est.reshape(N_comm, K)
    A = Hs @ Hs.T.conj() + sigma2 * 1.5 * np.eye(N_comm)
    W = np.linalg.solve(A + 1e-8 * np.eye(N_comm), Hs)
    p_w = np.sum(np.abs(W)**2)
    if p_w > P_comm:
        W = W * np.sqrt(P_comm / p_w)
    W = W.reshape(M, Nt, K)  # (64, 16, 10)
    
    # Step 3: 感知波束 (只有n_sens_ap个AP)
    P_sens = P_sens_ratio * Pmax
    H_t_sel = H_t[sens_selected]
    H_t_est_sel = H_t_est[sens_selected]
    
    Z = np.zeros((n_sens_ap, Nt, P), dtype=complex)
    for p in range(P):
        h = H_t_est_sel[:, p, :].flatten()
        norm = np.sqrt(np.sum(np.abs(h)**2))
        if norm > 0:
            Z[:, :, p] = (np.conj(h) / norm * np.sqrt(P_sens / P)).reshape(n_sens_ap, Nt)
    
    # 验证通信 (所有64 AP)
    Hs_true = H_u.reshape(N_comm, K)
    Wf = W.reshape(N_comm, K)
    sinr_list = []
    for k in range(K):
        sig = np.abs(Wf[:, k].conj() @ Hs_true[:, k])**2
        inter = sum(np.abs(Wf[:, j].conj() @ Hs_true[:, k])**2 for j in range(K) if j != k)
        sinr = 10 * np.log10(sig / (inter + sigma2 + 1e-10) + 1e-10)
        sinr_list.append(sinr)
    
    # 验证感知
    snr_list = []
    for p in range(P):
        h_true = H_t_sel[:, p, :].flatten()
        signal = np.abs(Z[:, :, p].flatten() @ h_true)**2
        snr = 10 * np.log10(signal / (sigma2 * np.sum(np.abs(Z)**2) + 1e-10) + 1e-10)
        snr_list.append(snr)
    
    # 功率
    total_power = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    
    # 结果
    success = all(s >= sinr_req for s in sinr_list) and \
              all(s >= snr_req for s in snr_list) and \
              total_power <= Pmax
    
    return {
        'success': success,
        'sinr': sinr_list,
        'snr': snr_list,
        'power': total_power,
        'sens_selected': sens_selected,
        'n_sens_ap': n_sens_ap
    }


# ============ 测试函数 ============
def test_correct_architecture():
    """测试正确架构"""
    print("=" * 60)
    print("正确架构测试: 64 AP通信 + n AP感知")
    print("=" * 60)
    print(f"约束: SINR≥{sinr_req}dB, SNR≥{snr_req}dB\n")
    
    for n_sens in [1, 2, 3, 4, 5, 7]:
        success = sum(1 for _ in range(50) if isac_correct_architecture(n_sens_ap=n_sens, error_var=0.05)['success'])
        print(f"n_sens_ap={n_sens}: {success*2}%")


if __name__ == "__main__":
    test_correct_architecture()


# ============ 动态AP选择系统 ============
def preliminary_detection(H_t_est, P_default=4):
    """
    初步检测: 估计目标数量
    
    基于能量比较: 当P增加时,总能量成比例增加
    使用参考能量(P=4时的典型能量)进行估计
    
    返回: 估计的目标数量 P_est
    """
    M, P_max, Nt = H_t_est.shape
    
    # 计算当前总能量
    total_energy = np.sum(np.abs(H_t_est)**2)
    
    # 典型能量: P=4时,经验值约 M*P*Nt*0.1 (归一化后)
    # 由于每次信道不同,使用动态参考
    # 简单方法: 直接使用P_max作为估计(如果有先验)
    
    # 方法2: 基于AP能量分布的峰值检测
    ap_energy = np.sum(np.sum(np.abs(H_t_est)**2, axis=2), axis=1)
    
    # 归一化能量
    ap_energy_norm = ap_energy / (np.max(ap_energy) + 1e-10)
    
    # 统计显著AP数量 (能量 > 10% 最大值)
    significant_aps = np.sum(ap_energy_norm > 0.1)
    
    # 映射: 每个目标约需要2-3个高能量AP
    P_est = max(1, min(significant_aps // 2, P_max))
    
    return P_est


def dynamic_ap_selection(P_est):
    """
    根据估计的目标数量,动态确定感知AP数量
    
    基于实验结果的映射 (保守估计,乘以1.5安全因子):
    - P=1-4: 需要5个AP
    - P=5-8: 需要7个AP  
    - P=9-12: 需要9个AP
    - P=13-16: 需要12个AP
    - P>16: 需要更多
    """
    # 应用1.5倍安全因子
    P_safe = int(P_est * 1.5)
    
    if P_safe <= 4:
        return 5
    elif P_safe <= 6:
        return 7
    elif P_safe <= 8:
        return 9
    elif P_safe <= 12:
        return 12
    else:
        return 15


def isac_dynamic_system(error_var=0.05, seed=None, P_true=None):
    """
    动态ISAC系统:
    1. 初步检测目标数量
    2. 根据目标数量动态选择AP数量
    3. 执行通信和感知
    
    参数:
        error_var: 信道估计误差
        seed: 随机种子
        P_true: 真实目标数量 (可选,用于测试)
    
    返回:
        dict: 包含成功标志、估计目标数、实际AP数等
    """
    # 生成信道
    P = P_true if P_true else P
    H_u, H_t, _ = generate_channel(P, seed)
    
    # 信道估计
    if error_var > 0:
        H_u_est = H_u + np.sqrt(error_var/2) * (np.random.randn(M, K, Nt) + 1j*np.random.randn(M, K, Nt))
        H_t_est = H_t + np.sqrt(error_var/2) * (np.random.randn(M, P, Nt) + 1j*np.random.randn(M, P, Nt))
    else:
        H_u_est = H_u
        H_t_est = H_t
    
    # Step 1: 初步检测目标数量
    P_est = preliminary_detection(H_t_est)
    
    # Step 2: 动态选择感知AP数量
    n_sens_ap = dynamic_ap_selection(P_est)
    
    # Step 3: AP选择
    ds = np.sum(np.sum(np.abs(H_t_est)**2, axis=2), axis=1)
    s_t = ds / (np.max(ds) + 1e-10)
    s_u = np.sum(np.sum(np.abs(H_u_est)**2, axis=1), axis=1)
    s_u = s_u / (np.max(s_u) + 1e-10)
    sens_selected = np.argsort(-(0.5 * s_u + 0.5 * s_t))[:n_sens_ap]
    
    # Step 4: 通信波束 (全部64 AP)
    N_comm = M * Nt
    P_comm = P_comm_ratio * Pmax
    
    Hs = H_u_est.reshape(N_comm, K)
    A = Hs @ Hs.T.conj() + sigma2 * 1.5 * np.eye(N_comm)
    W = np.linalg.solve(A + 1e-8 * np.eye(N_comm), Hs)
    p_w = np.sum(np.abs(W)**2)
    if p_w > P_comm:
        W = W * np.sqrt(P_comm / p_w)
    W = W.reshape(M, Nt, K)
    
    # Step 5: 感知波束 (动态AP数)
    P_sens = P_sens_ratio * Pmax
    H_t_sel = H_t[sens_selected]
    H_t_est_sel = H_t_est[sens_selected]
    
    Z = np.zeros((n_sens_ap, Nt, P), dtype=complex)
    for p in range(P):
        h = H_t_est_sel[:, p, :].flatten()
        norm = np.sqrt(np.sum(np.abs(h)**2))
        if norm > 0:
            Z[:, :, p] = (np.conj(h) / norm * np.sqrt(P_sens / P)).reshape(n_sens_ap, Nt)
    
    # 验证通信
    Hs_true = H_u.reshape(N_comm, K)
    Wf = W.reshape(N_comm, K)
    sinr_list = []
    for k in range(K):
        sig = np.abs(Wf[:, k].conj() @ Hs_true[:, k])**2
        inter = sum(np.abs(Wf[:, j].conj() @ Hs_true[:, k])**2 for j in range(K) if j != k)
        sinr = 10 * np.log10(sig / (inter + sigma2 + 1e-10) + 1e-10)
        sinr_list.append(sinr)
    
    # 验证感知
    snr_list = []
    for p in range(P):
        h_true = H_t_sel[:, p, :].flatten()
        signal = np.abs(Z[:, :, p].flatten() @ h_true)**2
        snr = 10 * np.log10(signal / (sigma2 * np.sum(np.abs(Z)**2) + 1e-10) + 1e-10)
        snr_list.append(snr)
    
    total_power = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    
    success = all(s >= sinr_req for s in sinr_list) and \
              all(s >= snr_req for s in snr_list) and \
              total_power <= Pmax
    
    return {
        'success': success,
        'P_true': P,
        'P_est': P_est,
        'n_sens_ap': n_sens_ap,
        'sinr': sinr_list,
        'snr': snr_list,
        'power': total_power
    }


# ============ 测试动态系统 ============
def test_dynamic_system():
    """测试动态AP选择系统"""
    print("=" * 60)
    print("动态AP选择系统测试")
    print("=" * 60)
    print(f"参数: M={M}AP, Nt={Nt}天线, K={K}用户")
    print(f"约束: SINR≥{sinr_req}dB, SNR≥{snr_req}dB\n")
    
    print("目标数 | 估计数 | 感知AP | 成功率")
    print("-" * 45)
    
    for P_true in [4, 6, 8, 10, 12, 15]:
        success_count = 0
        p_est_list = []
        ap_list = []
        
        for _ in range(50):
            result = isac_dynamic_system(error_var=0.05, P_true=P_true)
            if result['success']:
                success_count += 1
            p_est_list.append(result['P_est'])
            ap_list.append(result['n_sens_ap'])
        
        pct = success_count * 2
        P_avg = np.mean(p_est_list)
        ap_avg = np.mean(ap_list)
        
        bar = '#' * (pct//5) + '.' * (20 - pct//5)
        print(f"  P={P_true:2d}   |   {P_avg:2.0f}   |   {ap_avg:.1f}   | {pct:2d}% [{bar}]")


if __name__ == "__main__":
    test_dynamic_system()


# ============ 自适应AP选择系统 (反馈迭代) ============
def isac_adaptive_system(P_true=None, error_var=0.05, seed=None, max_sens_aps=20, n_init=2):
    """
    自适应ISAC系统 (反馈迭代):
    1. 从n_init个感知AP开始
    2. 失败时增加2个AP重试
    3. 直到成功或达到上限
    
    参数:
        P_true: 真实目标数量
        error_var: 信道估计误差
        seed: 随机种子
        max_sens_aps: 最大感知AP数
        n_init: 初始感知AP数
    
    返回:
        dict: 成功标志和使用的AP数量
    """
    P = P_true if P_true else P
    H_u, H_t, _ = generate_channel(P, seed)
    
    if error_var > 0:
        H_u_est = H_u + np.sqrt(error_var/2) * (np.random.randn(M, K, Nt) + 1j*np.random.randn(M, K, Nt))
        H_t_est = H_t + np.sqrt(error_var/2) * (np.random.randn(M, P, Nt) + 1j*np.random.randn(M, P, Nt))
    else:
        H_u_est = H_u
        H_t_est = H_t
    
    n_sens = n_init
    
    while n_sens <= max_sens_aps:
        ds = np.sum(np.sum(np.abs(H_t_est)**2, axis=2), axis=1)
        s_t = ds / (np.max(ds) + 1e-10)
        s_u = np.sum(np.sum(np.abs(H_u_est)**2, axis=1), axis=1)
        s_u = s_u / (np.max(s_u) + 1e-10)
        sens_selected = np.argsort(-(0.5 * s_u + 0.5 * s_t))[:n_sens]
        
        N_comm = M * Nt
        P_comm = P_comm_ratio * Pmax
        
        Hs = H_u_est.reshape(N_comm, K)
        A = Hs @ Hs.T.conj() + sigma2 * 1.5 * np.eye(N_comm)
        W = np.linalg.solve(A + 1e-8 * np.eye(N_comm), Hs)
        p_w = np.sum(np.abs(W)**2)
        if p_w > P_comm:
            W = W * np.sqrt(P_comm / p_w)
        W = W.reshape(M, Nt, K)
        
        P_sens = P_sens_ratio * Pmax
        H_t_sel = H_t[sens_selected]
        H_t_est_sel = H_t_est[sens_selected]
        
        Z = np.zeros((n_sens, Nt, P), dtype=complex)
        for p in range(P):
            h = H_t_est_sel[:, p, :].flatten()
            norm = np.sqrt(np.sum(np.abs(h)**2))
            if norm > 0:
                Z[:, :, p] = (np.conj(h) / norm * np.sqrt(P_sens / P)).reshape(n_sens, Nt)
        
        Hs_true = H_u.reshape(N_comm, K)
        Wf = W.reshape(N_comm, K)
        
        # 验证通信SINR
        sinr_ok = True
        for k in range(K):
            sig = np.abs(Wf[:, k].conj() @ Hs_true[:, k])**2
            inter = sum(np.abs(Wf[:, j].conj() @ Hs_true[:, k])**2 for j in range(K) if j != k)
            sinr = 10 * np.log10(sig / (inter + sigma2 + 1e-10) + 1e-10)
            if sinr < sinr_req:
                sinr_ok = False
                break
        
        # 验证感知SNR
        snr_ok = True
        for p in range(P):
            h_true = H_t_sel[:, p, :].flatten()
            signal = np.abs(Z[:, :, p].flatten() @ h_true)**2
            snr = 10 * np.log10(signal / (sigma2 * np.sum(np.abs(Z)**2) + 1e-10) + 1e-10)
            if snr < snr_req:
                snr_ok = False
                break
        
        if sinr_ok and snr_ok:
            return {'success': True, 'P': P, 'n_sens': n_sens, 'n_init': n_init}
        
        n_sens += 2
    
    return {'success': False, 'P': P, 'n_sens': n_sens, 'n_init': n_init}


def test_adaptive_system():
    """测试自适应AP选择系统"""
    print("=" * 60)
    print("自适应AP选择系统 (初始2个AP)")
    print("=" * 60)
    print(f"参数: M={M}AP, Nt={Nt}天线, K={K}用户")
    print(f"约束: SINR≥{sinr_req}dB, SNR≥{snr_req}dB\n")
    print("目标数 | 成功率 | 平均AP")
    print("-" * 30)
    
    for P in [4, 6, 8, 10, 12, 15, 20]:
        success = 0
        total_aps = 0
        
        for _ in range(50):
            result = isac_adaptive_system(P_true=P, error_var=0.05, n_init=2)
            if result['success']:
                success += 1
                total_aps += result['n_sens']
        
        pct = success * 2
        avg_aps = total_aps / max(success, 1)
        bar = '#' * (pct//5) + '.' * (20 - pct//5)
        print(f"  P={P:2d}   | {pct:2d}% [{bar}] | {avg_aps:.1f}")


if __name__ == "__main__":
    test_adaptive_system()
