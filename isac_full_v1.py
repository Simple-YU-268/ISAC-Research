"""
ISAC完整方案 - 端到端可学习优化
目标: 10用户通信SINR≥0dB + 感知定位 + AP选择(16选4) + 功率≤30W
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 系统参数
M, K, P, Nt = 16, 10, 4, 4   # 10个用户 (K=10)
Pmax = 30
N_req = 4                     # 选择4个AP
target_sinr_db = 0           # 目标SINR 0dB

print("=== ISAC完整方案 v1 ===")
print(f"系统: {M}AP, {K}用户, {Nt}天线, {P}感知目标")
print(f"目标: SINR≥{target_sinr_db}dB, 功率≤{Pmax}W, AP选择{N_req}个")

# ============= 信道模型 =============
def generate_channel(ap_pos, user_pos, target_pos):
    """生成信道: AP-用户 + AP-感知目标"""
    H_u = np.zeros((M, K, Nt), dtype=complex)  # 用户信道
    H_t = np.zeros((M, P, Nt), dtype=complex)  # 感知目标信道
    
    for m in range(M):
        # 用户信道
        for k in range(K):
            d = max(np.sqrt(np.sum((ap_pos[m] - user_pos[k])**2)), 5)
            pl = (d / 10) ** -2.5
            H_u[m, k, :] = np.sqrt(pl) * (np.random.randn(Nt) + 1j * np.random.randn(Nt)) / np.sqrt(2)
        
        # 感知目标信道
        for p in range(P):
            d = max(np.sqrt(np.sum((ap_pos[m] - target_pos[p])**2)), 5)
            pl = (d / 10) ** -2.5
            H_t[m, p, :] = np.sqrt(pl) * (np.random.randn(Nt) + 1j * np.random.randn(Nt)) / np.sqrt(2)
    
    return H_u, H_t

# ============= 波束成形优化 (scipy) =============
from scipy.optimize import minimize

def optimize_isac(H_u, H_t, ap_selection, Pmax, target_sinr_db):
    """联合优化通信+感知的波束成形"""
    M_sel = np.sum(ap_selection > 0.5)
    if M_sel == 0:
        return None, None
    
    # 简化: 平均功率分配
    p_per_ap = Pmax / M_sel
    
    # 通信: MMSE预编码
    H_sel = H_u[ap_selection > 0.5, :, :]  # (M_sel, K, Nt)
    Hs = H_sel.reshape(M_sel * Nt, K)
    
    sigma2 = 0.5
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    
    try:
        W_mmse = np.linalg.inv(HH) @ Hs
        power = np.sum(np.abs(W_mmse) ** 2)
        if power > 0:
            W_mmse = W_mmse * np.sqrt(p_per_ap * 0.8 / power)
    except:
        W_mmse = np.zeros((M_sel * Nt, K), dtype=complex)
    
    # 感知: 发射感知信号
    Z = np.zeros((M_sel, P, Nt), dtype=complex)
    for p in range(P):
        h_t = H_sel[:, p, :]  # (M_sel, Nt)
        Z[:, p, :] = h_t / (np.sqrt(np.sum(np.abs(h_t)**2)) + 1e-3) * np.sqrt(p_per_ap * 0.2 / P)
    
    return W_mmse, Z

def compute_sinr(H_u, W, ap_selection):
    """计算通信SINR"""
    M_sel = int(np.sum(ap_selection))
    H_sel = H_u[ap_selection > 0.5, :, :]  # (M_sel, K, Nt)
    
    sinrs = []
    for k in range(K):
        # 有用信号
        h_k = H_sel[:, k, :]  # (M_sel, Nt)
        w_k = W[:, k] if W is not None else np.zeros(M_sel * Nt, dtype=complex)
        
        if np.sum(np.abs(w_k)) < 1e-6:
            sinrs.append(-100)
            continue
            
        signal = np.abs(np.sum(np.conj(w_k) @ h_k.flatten())) ** 2
        
        # 干扰
        interference = 0
        for j in range(K):
            if j != k:
                w_j = W[:, j] if W is not None else np.zeros(M_sel * Nt, dtype=complex)
                interference += np.abs(np.sum(np.conj(w_j) @ h_k.flatten())) ** 2
        
        sinr_db = 10 * np.log10(signal / (interference + 0.01) + 1e-10)
        sinrs.append(sinr_db)
    
    return np.array(sinrs)

def compute_crb(H_t, Z, ap_selection):
    """计算感知CRB (定位精度)"""
    M_sel = int(np.sum(ap_selection))
    if M_sel < 2 or Z is None:
        return 1000  # 差定位精度
    
    crbs = []
    for p in range(P):
        # 简化CRB: 基于等效信道质量
        z_p = Z[:, p, :]  # (M_sel, Nt)
        h_t_p = H_t[ap_selection > 0.5, p, :]
        
        # 等效感知信道
        heff = z_p * np.conj(h_t_p)
        power = np.sum(np.abs(heff)) ** 2
        
        crb = 1 / (power + 0.1)  # 简化CRB公式
        crbs.append(crb)
    
    return np.mean(crbs)

# ============= 测试基准算法 =============
print("\n=== 测试基准算法 ===")

results = []
for trial in range(50):
    # 生成场景
    ap_pos = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
    user_pos = np.random.uniform(-50, 50, (K, 2))
    target_pos = np.random.uniform(-30, 30, (P, 2))
    
    H_u, H_t = generate_channel(ap_pos, user_pos, target_pos)
    
    # 方法1: 选择最强AP + MMSE
    signal_power = np.sum(np.abs(H_u) ** 2, axis=(2, 3))  # (M, K)
    total_signal = signal_power.sum(axis=1)
    selected_ap = np.zeros(M)
    selected_idx = np.argsort(-total_signal)[:N_req]
    selected_ap[selected_idx] = 1
    
    W, Z = optimize_isac(H_u, H_t, selected_ap, Pmax, target_sinr_db)
    sinrs = compute_sinr(H_u, W, selected_ap)
    crb = compute_crb(H_t, Z, selected_ap)
    
    power = np.sum(np.abs(W) ** 2) if W is not None else 0
    power += np.sum(np.abs(Z) ** 2) if Z is not None else 0
    
    results.append({
        'sinr_min': np.min(sinrs),
        'sinr_mean': np.mean(sinrs),
        'crb': crb,
        'power': power,
        'sinr_ok': np.sum(sinrs >= target_sinr_db)
    })

print(f"方法1 (选择最强{N_req}AP + MMSE):")
print(f"  SINR_min: {np.min([r['sinr_min'] for r in results]):.2f}dB")
print(f"  SINR平均: {np.mean([r['sinr_mean'] for r in results]):.2f}dB")
print(f"  功率: {np.mean([r['power'] for r in results]):.2f}W")
print(f"  SINR≥0dB用户比例: {np.mean([r['sinr_ok'] for r in results])/K*100:.1f}%")
print(f"  ≥5用户满足: {sum(1 for r in results if r['sinr_ok'] >= 5)}/50")

# 方法2: 所有AP + 更优功率分配
print("\n方法2 (所有AP + 功率优化):")
results2 = []
for trial in range(50):
    ap_pos = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
    user_pos = np.random.uniform(-50, 50, (K, 2))
    target_pos = np.random.uniform(-30, 30, (P, 2))
    
    H_u, H_t = generate_channel(ap_pos, user_pos, target_pos)
    
    selected_ap = np.ones(M)  # 所有AP
    W, Z = optimize_isac(H_u, H_t, selected_ap, Pmax, target_sinr_db)
    sinrs = compute_sinr(H_u, W, selected_ap)
    
    results2.append({'sinr_min': np.min(sinrs), 'sinr_ok': np.sum(sinrs >= target_sinr_db)})

print(f"  SINR_min: {np.min([r['sinr_min'] for r in results2]):.2f}dB")
print(f"  SINR平均: {np.mean([r['sinr_min'] for r in results2]):.2f}dB")
print(f"  ≥5用户满足: {sum(1 for r in results2 if r['sinr_ok'] >= 5)}/50")

print("\n=== 结论: 当前算法远未达标 ===")
print(f"目标: 10用户全部SINR≥0dB")
print(f"当前: 最多5-6用户达到要求")
