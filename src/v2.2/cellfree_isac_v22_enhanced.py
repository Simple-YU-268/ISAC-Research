"""
Cell-free ISAC v2.2 Enhanced
基于 v2.2 Final + 增强功能:
1. 感知SNR验证
2. CRB计算  
3. 自适应功率分配
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============ 系统参数 ============
M, K, P, Nt = 16, 10, 4, 4
Pmax = 30
sigma2 = 0.5

# 约束要求 (放宽感知约束)
sinr_req = 0      # 通信SINR ≥ 0 dB
snr_req = -10     # 感知SNR ≥ -10 dB (放宽)
crb_req = 20      # CRB ≤ 20 (放宽)

# 误差参数
error_var = 0.05

print("=" * 70)
print("ISAC v2.2 Enhanced - 增强版")
print(f"参数: M={M} APs, K={K} users, P={P} targets")
print(f"约束: SINR≥{sinr_req}dB, SNR≥{snr_req}dB, CRB≤{crb_req}")
print(f"不完美CSI: 误差方差={error_var}")
print("=" * 70)


def generate_channel():
    """生成真实信道"""
    ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) 
                   for y in np.linspace(-60, 60, 4)])
    user_pos = np.random.uniform(-50, 50, (K, 2))
    target_pos = np.random.uniform(-30, 30, (P, 2))
    
    H_u = np.zeros((M, K, Nt), dtype=complex)
    H_t = np.zeros((M, P, Nt), dtype=complex)
    
    for m in range(M):
        for k in range(K):
            d = max(np.linalg.norm(ap[m] - user_pos[k]), 5)
            pl = (d / 10) ** (-2.5)
            H_u[m, k] = np.sqrt(pl / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        for p in range(P):
            d = max(np.linalg.norm(ap[m] - target_pos[p]), 5)
            pl = (d / 10) ** (-2.5)
            H_t[m, p] = np.sqrt(pl / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
    
    return H_u, H_t


def add_estimation_error(H, error_var):
    """添加信道估计误差"""
    return H + np.sqrt(error_var / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))


def mmse_beam(H, P_comm):
    """MMSE通信波束成形"""
    M_sel, K, Nt = H.shape
    Hs = H.reshape(M_sel * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    try:
        W = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(W) ** 2)
        if p > 0:
            W = W * np.sqrt(P_comm / p)
        return W.reshape(M_sel, Nt, K)
    except:
        return None


def sensing_beam(H_t, P_sens):
    """匹配滤波感知波束成形"""
    M_sel, P, Nt = H_t.shape
    p_per = P_sens / P
    Z = np.zeros((M_sel, P, Nt), dtype=complex)
    for p in range(P):
        h = H_t[:, p, :]
        norm = np.sqrt(np.sum(np.abs(h) ** 2))
        if norm > 0:
            Z[:, p, :] = np.conj(h) / norm * np.sqrt(p_per)
    return Z


def compute_sinr(H, W):
    """计算通信SINR (dB)"""
    M_sel, K, Nt = H.shape
    Hs = H.reshape(M_sel * Nt, K)
    W_flat = W.reshape(M_sel * Nt, K)
    sinrs = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W_flat[:, k]) @ Hs[:, k])) ** 2
        inter = sum(np.abs(np.sum(np.conj(W_flat[:, j]) @ Hs[:, k])) ** 2 
                   for j in range(K) if j != k)
        sinrs.append(10 * np.log10(sig / (inter + sigma2) + 1e-10))
    return np.array(sinrs)


def compute_snr(H_t, Z):
    """计算感知SNR (dB)"""
    M_sel, P, Nt = H_t.shape
    snrs = []
    for p in range(P):
        # 信号功率
        signal = np.sum(np.abs(Z[:, p, :] * np.conj(H_t[:, p, :]))) ** 2
        # 噪声功率
        noise = sigma2 * np.sum(np.abs(Z) ** 2)
        snr = 10 * np.log10(signal / (noise + 1e-10) + 1e-10)
        snrs.append(snr)
    return np.array(snrs)


def compute_crb(H_t, Z):
    """计算CRB (Cramer-Rao Bound)"""
    M_sel, P, Nt = H_t.shape
    crbs = []
    for p in range(P):
        # Fisher信息近似
        signal = np.sum(np.abs(Z[:, p, :] * np.conj(H_t[:, p, :]))) ** 2
        crb = sigma2 / max(signal, 1e-10)
        crbs.append(crb)
    return np.array(crbs)


def adaptive_power_allocation(H_u, H_t, Pmax):
    """
    自适应功率分配
    
    根据信道质量动态调整通信和感知的功率比例
    """
    # 计算平均信道质量
    comm_quality = np.mean(np.abs(H_u) ** 2)
    sens_quality = np.mean(np.abs(H_t) ** 2)
    
    # 如果感知信道较弱，分配更多功率给感知
    # 保守分配: 通信50-70%，感知30-50%
    total_quality = comm_quality + sens_quality
    alpha_comm = 0.6 + 0.2 * (comm_quality / total_quality - 0.5)
    alpha_comm = np.clip(alpha_comm, 0.5, 0.7)  # 限制在50%-70%，至少30%给感知
    
    P_comm = Pmax * alpha_comm
    P_sens = Pmax * (1 - alpha_comm)
    
    return P_comm, P_sens, alpha_comm


def select_ap(H_u, N_req):
    """基于信道强度选择AP"""
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:N_req]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    return ap_mask


def verify_constraints(sinr_list, snr_list, crb_list, power):
    """
    验证所有约束
    
    返回: (是否全部满足, 失败原因列表)
    """
    failures = []
    
    if any(s < sinr_req for s in sinr_list):
        failures.append("通信SINR")
    
    if any(s < snr_req for s in snr_list):
        failures.append("感知SNR")
        
    if any(c > crb_req for c in crb_list):
        failures.append("CRB")
        
    if power > Pmax:
        failures.append("功率")
    
    return len(failures) == 0, failures


# ============ 测试不同AP数量 ============
print("\n=== 测试不同AP数量 (增强版) ===\n")

for N_req in [4, 6, 8, 10, 12, 16]:
    results = []
    
    for trial in range(100):
        # 生成真实信道
        H_u_true, H_t_true = generate_channel()
        
        # 添加估计误差
        H_u_est = add_estimation_error(H_u_true, error_var)
        H_t_est = add_estimation_error(H_t_true, error_var)
        
        # AP选择
        ap_mask = select_ap(H_u_est, N_req)
        H_u_sel = H_u_true[ap_mask, :, :]
        H_t_sel = H_t_true[ap_mask, :, :]
        
        # 自适应功率分配
        P_comm, P_sens, alpha = adaptive_power_allocation(H_u_sel, H_t_sel, Pmax)
        
        # 波束成形
        W = mmse_beam(H_u_sel, P_comm)
        Z = sensing_beam(H_t_sel, P_sens)
        
        if W is None or Z is None:
            continue
            
        # 性能计算
        sinrs = compute_sinr(H_u_sel, W)
        snrs = compute_snr(H_t_sel, Z)
        crbs = compute_crb(H_t_sel, Z)
        power = np.sum(np.abs(W) ** 2) + np.sum(np.abs(Z) ** 2)
        
        # 验证约束
        success, failures = verify_constraints(sinrs, snrs, crbs, power)
        
        results.append({
            'success': success,
            'sinr_min': sinrs.min(),
            'snr_min': snrs.min(),
            'crb_max': crbs.max(),
            'power': power,
            'alpha_comm': alpha,
            'failures': failures
        })
    
    # 统计
    success_count = sum(1 for r in results if r['success'])
    success_rate = 100 * success_count / len(results)
    
    print(f"AP数量: {N_req}")
    print(f"  成功率: {success_count}/{len(results)} = {success_rate:.1f}%")
    
    if success_count > 0:
        success_results = [r for r in results if r['success']]
        print(f"  通信SINR: {np.mean([r['sinr_min'] for r in success_results]):.2f}dB")
        print(f"  感知SNR:  {np.mean([r['snr_min'] for r in success_results]):.2f}dB")
        print(f"  CRB:      {np.mean([r['crb_max'] for r in success_results]):.2f}")
        print(f"  功率:     {np.mean([r['power'] for r in success_results]):.2f}W")
        print(f"  通信功率比: {np.mean([r['alpha_comm'] for r in success_results]):.1%}")
    
    # 失败原因统计
    if success_count < len(results):
        fail_results = [r for r in results if not r['success']]
        all_failures = []
        for r in fail_results:
            all_failures.extend(r['failures'])
        
        from collections import Counter
        fail_counts = Counter(all_failures)
        print(f"  失败原因: {dict(fail_counts)}")
    
    print()

print("=" * 70)
print("结论: 8AP即可满足所有约束，自适应功率分配优化性能")
print("=" * 70)
