"""针对3个AP的精细优化"""
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import random

M, K, P, Nt = 16, 10, 4, 4
Pmax = 30
sigma2 = 0.5

def mmse_beam(H, Pmax):
    M_sel, K, Nt = H.shape
    Hs = H.reshape(M_sel * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    try:
        W = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(W)**2)
        W = W * np.sqrt(Pmax / p)
        return W.reshape(M_sel, Nt, K)
    except:
        return None

def compute_sinr(H, W):
    M_sel, K, Nt = H.shape
    Hs = H.reshape(M_sel * Nt, K)
    W_flat = W.reshape(M_sel * Nt, K)
    sinrs = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W_flat[:, k]) @ Hs[:, k]))**2
        inter = sum(np.abs(np.sum(np.conj(W_flat[:, j]) @ Hs[:, k]))**2 for j in range(K) if j != k)
        sinrs.append(10 * np.log10(sig / (inter + sigma2) + 1e-10))
    return np.array(sinrs)

def generate_channel():
    ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
    # 用户位置更集中一些，便于3个AP覆盖
    user_pos = np.random.uniform(-30, 30, (K, 2))
    target_pos = np.random.uniform(-30, 30, (P, 2))
    
    H_u = np.zeros((M, K, Nt), dtype=complex)
    H_t = np.zeros((M, P, Nt), dtype=complex)
    
    for m in range(M):
        for k in range(K):
            d = max(np.sqrt(np.sum((ap[m] - user_pos[k])**2)), 5)
            pl = (d / 10)**-2.5
            H_u[m, k, :] = np.sqrt(pl / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        for p in range(P):
            d = max(np.sqrt(np.sum((ap[m] - target_pos[p])**2)), 5)
            pl = (d / 10)**-2.5
            H_t[m, p, :] = np.sqrt(pl / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
    
    return H_u, H_t, ap, user_pos

# 方法: 联合优化AP选择 + 功率分配
def joint_optimize(H_u, N_req, P_comm):
    """联合优化AP选择和功率分配"""
    all_combos = list(combinations(range(M), N_req))
    
    best_sinr = -100
    best_combo = None
    best_pcomm = None
    
    for combo in all_combos:
        H_sel = H_u[list(combo), :, :]
        
        # 尝试不同功率比例
        for p_ratio in [0.7, 0.8, 0.9, 1.0]:
            W = mmse_beam(H_sel, Pmax * p_ratio)
            if W is not None:
                sinrs = compute_sinr(H_sel, W)
                min_sinr = sinrs.min()
                
                if min_sinr > best_sinr:
                    best_sinr = min_sinr
                    best_combo = combo
                    best_pcomm = p_ratio
    
    return best_combo, best_pcomm

print("=== 3个AP精细优化测试 (用户集中在30m内) ===\n")

# 方法1: 联合优化
results1 = []
for _ in range(100):
    H_u, H_t, ap, user_pos = generate_channel()
    
    combo, p_ratio = joint_optimize(H_u, 3, Pmax)
    
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[list(combo)] = True
    H_sel = H_u[ap_mask, :, :]
    
    W = mmse_beam(H_sel, Pmax * p_ratio)
    Z = np.zeros((3, P, Nt), dtype=complex)  # 简化感知
    
    sinrs = compute_sinr(H_sel, W)
    total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    
    results1.append({'sinr_min': sinrs.min(), 'comm_ok': sum(sinrs >= 0), 'power': total_pwr})

# 方法2: 遍历所有组合
results2 = []
for _ in range(100):
    H_u, H_t, ap, user_pos = generate_channel()
    
    all_combos = list(combinations(range(M), 3))
    
    best_sinr = -100
    best_combo = None
    
    for combo in all_combos:
        H_sel = H_u[list(combo), :, :]
        W = mmse_beam(H_sel, Pmax * 0.8)
        if W is not None:
            sinrs = compute_sinr(H_sel, W)
            if sinrs.min() > best_sinr:
                best_sinr = sinrs.min()
                best_combo = combo
    
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[list(best_combo)] = True
    H_sel = H_u[ap_mask, :, :]
    W = mmse_beam(H_sel, Pmax * 0.8)
    sinrs = compute_sinr(H_sel, W)
    
    results2.append({'sinr_min': sinrs.min(), 'comm_ok': sum(sinrs >= 0)})

print(f"联合优化 (功率可调):")
print(f"  SINR_min: 最小{np.min([r['sinr_min'] for r in results1]):.2f}dB, 平均{np.mean([r['sinr_min'] for r in results1]):.2f}dB")
print(f"  全部10用户≥0dB: {sum(1 for r in results1 if r['comm_ok']==K)}/100")
print(f"  功率≤30W: {sum(1 for r in results1 if r['power']<=30)}/100")

print(f"\n固定功率0.8:")
print(f"  SINR_min: 最小{np.min([r['sinr_min'] for r in results2]):.2f}dB, 平均{np.mean([r['sinr_min'] for r in results2]):.2f}dB")
print(f"  全部10用户≥0dB: {sum(1 for r in results2 if r['comm_ok']==K)}/100")

# 方法3: 尝试更少用户
print("\n=== 减少用户数测试 (3个AP) ===")
for K_test in [6, 8, 10]:
    results = []
    for _ in range(50):
        ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        user_pos = np.random.uniform(-30, 30, (K_test, 2))
        
        H_u = np.zeros((M, K_test, Nt), dtype=complex)
        for m in range(M):
            for k in range(K_test):
                d = max(np.sqrt(np.sum((ap[m] - user_pos[k])**2)), 5)
                pl = (d / 10)**-2.5
                H_u[m, k, :] = np.sqrt(pl / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        
        all_combos = list(combinations(range(M), 3))
        best_sinr = -100
        best_combo = None
        for combo in all_combos:
            H_sel = H_u[list(combo), :, :]
            W = mmse_beam(H_sel, Pmax * 0.8)
            if W is not None:
                sinrs = compute_sinr(H_sel, W)
                if sinrs.min() > best_sinr:
                    best_sinr = sinrs.min()
                    best_combo = combo
        
        ap_mask = np.zeros(M, dtype=bool)
        ap_mask[list(best_combo)] = True
        H_sel = H_u[ap_mask, :, :]
        W = mmse_beam(H_sel, Pmax * 0.8)
        sinrs = compute_sinr(H_sel, W)
        
        results.append({'sinr_min': sinrs.min(), 'comm_ok': sum(sinrs >= 0)})
    
    print(f"K={K_test}: SINR_min平均={np.mean([r['sinr_min'] for r in results]):.2f}dB, 全部≥0dB: {sum(1 for r in results if r['comm_ok']==K_test)}/50")
