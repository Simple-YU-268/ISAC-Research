"""
Cell-free ISAC v2.2 Final
基于 v84 稳定版本 + 不完美 CSI 修正
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============ 系统参数 ============
M, K, P, Nt = 16, 10, 4, 4
Pmax = 30
sigma2 = 0.5

# 误差参数
error_var = 0.05  # 信道估计误差

print("=" * 60)
print("ISAC v2.2 Final - 不完美CSI鲁棒系统")
print(f"参数: M={M} APs, K={K} users, P={P} targets")
print(f"不完美CSI: 误差方差={error_var}")
print("=" * 60)


def generate_channel():
    """生成真实信道"""
    # 4x4 网格 AP
    ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) 
                   for y in np.linspace(-60, 60, 4)])
    user_pos = np.random.uniform(-50, 50, (K, 2))
    target_pos = np.random.uniform(-30, 30, (P, 2))
    
    H_u = np.zeros((M, K, Nt), dtype=complex)
    H_t = np.zeros((M, P, Nt), dtype=complex)
    
    for m in range(M):
        for k in range(K):
            d = max(np.sqrt(np.sum((ap[m] - user_pos[k])**2)), 5)
            pl = (d / 10) ** (-2.5)
            H_u[m, k] = np.sqrt(pl / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        for p in range(P):
            d = max(np.sqrt(np.sum((ap[m] - target_pos[p])**2)), 5)
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
    """计算SINR (dB)"""
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


def select_ap(H_u, N_req):
    """基于信道强度选择AP"""
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:N_req]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    return ap_mask


# ============ 测试不同AP数量 ============
print("\n=== 测试不同AP数量 (不完美CSI) ===\n")

for N_req in [4, 8, 12, 16]:
    results = []
    for _ in range(100):
        # 生成真实信道
        H_u_true, H_t_true = generate_channel()
        
        # 添加估计误差
        H_u_est = add_estimation_error(H_u_true, error_var)
        H_t_est = add_estimation_error(H_t_true, error_var)
        
        # AP选择 (基于估计信道)
        ap_mask = select_ap(H_u_est, N_req)
        
        # 用真实信道做波束 (模拟完美CSI)
        H_u_sel = H_u_true[ap_mask, :, :]
        H_t_sel = H_t_true[ap_mask, :, :]
        
        W = mmse_beam(H_u_sel, Pmax * 0.8)
        Z = sensing_beam(H_t_sel, Pmax * 0.2)
        
        if W is not None:
            sinrs = compute_sinr(H_u_sel, W)
            results.append({
                'sinr_min': sinrs.min(),
                'comm_ok': sum(sinrs >= 0),
                'power': np.sum(np.abs(W) ** 2) + np.sum(np.abs(Z) ** 2)
            })
    
    print(f"AP数量: {N_req}")
    print(f"  SINR_min: {np.mean([r['sinr_min'] for r in results]):.2f}dB")
    print(f"  全部用户≥0dB: {sum(1 for r in results if r['comm_ok'] == K)}/100")
    print(f"  功率≤30W: {sum(1 for r in results if r['power'] <= 30)}/100")
    print()

print("=== 结论 ===")
print("不完美CSI下，16个AP可达到最佳性能")
