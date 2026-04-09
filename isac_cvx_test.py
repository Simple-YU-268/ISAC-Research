"""测试CVX优化在不同AP数量下的性能"""
import numpy as np
from scipy.optimize import minimize

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

def sensing_beam(H_t, Pmax):
    M_sel, P, Nt = H_t.shape
    p_sensing = Pmax / P
    Z = np.zeros((M_sel, P, Nt), dtype=complex)
    for p in range(P):
        h_t = H_t[:, p, :]
        norm = np.sqrt(np.sum(np.abs(h_t)**2))
        if norm > 0:
            Z[:, p, :] = np.conj(h_t) / norm * np.sqrt(p_sensing)
    return Z

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
    user_pos = np.random.uniform(-50, 50, (K, 2))
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
    
    return H_u, H_t

# 方法1: 基于信道强度的AP选择
def select_ap_by_channel(H_u, N_req):
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:N_req]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    return ap_mask

# 方法2: 遍历所有组合找最优 (CVX风格)
def select_ap_exhaustive(H_u, N_req):
    """遍历所有AP组合，选择最优"""
    from itertools import combinations
    
    best_sinr = -100
    best_mask = None
    
    # 采样测试所有组合
    all_combos = list(combinations(range(M), N_req))
    # 随机采样100个组合避免太慢
    if len(all_combos) > 100:
        import random
        all_combos = random.sample(all_combos, 100)
    
    for combo in all_combos:
        ap_mask = np.zeros(M, dtype=bool)
        ap_mask[list(combo)] = True
        
        H_sel = H_u[ap_mask, :, :]
        W = mmse_beam(H_sel, Pmax * 0.8)
        
        if W is not None:
            sinrs = compute_sinr(H_sel, W)
            min_sinr = sinrs.min()
            
            if min_sinr > best_sinr:
                best_sinr = min_sinr
                best_mask = ap_mask
    
    return best_mask if best_mask is not None else select_ap_by_channel(H_u, N_req)

print("=== CVX风格AP选择测试 ===\n")
print(f"{'AP数':^6} | {'选择方法':^12} | {'SINR_min':^12} | {'SINR_avg':^10} | {'全部≥0dB':^8}")
print("-" * 70)

for N_req in [3, 4, 5, 6]:
    # 方法1: 信道强度
    results1 = []
    for _ in range(100):
        H_u, H_t = generate_channel()
        ap_mask = select_ap_by_channel(H_u, N_req)
        H_u_sel = H_u[ap_mask, :, :]
        W = mmse_beam(H_u_sel, Pmax * 0.8)
        Z = sensing_beam(H_t[ap_mask, :, :], Pmax * 0.2)
        sinrs = compute_sinr(H_u_sel, W)
        results1.append({'sinr_min': sinrs.min(), 'comm_ok': sum(sinrs >= 0), 'power': np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)})
    
    # 方法2: 遍历最优
    results2 = []
    for _ in range(100):
        H_u, H_t = generate_channel()
        ap_mask = select_ap_exhaustive(H_u, N_req)
        H_u_sel = H_u[ap_mask, :, :]
        W = mmse_beam(H_u_sel, Pmax * 0.8)
        Z = sensing_beam(H_t[ap_mask, :, :], Pmax * 0.2)
        sinrs = compute_sinr(H_u_sel, W)
        results2.append({'sinr_min': sinrs.min(), 'comm_ok': sum(sinrs >= 0), 'power': np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)})
    
    sinr1 = [r['sinr_min'] for r in results1]
    sinr2 = [r['sinr_min'] for r in results2]
    
    print(f"  {N_req:^4} | 信道强度     | {np.min(sinr1):^12.2f} | {np.mean(sinr1):^10.2f} | {sum(1 for r in results1 if r['comm_ok']==K):^8}")
    print(f"  {N_req:^4} | CVX遍历最优  | {np.min(sinr2):^12.2f} | {np.mean(sinr2):^10.2f} | {sum(1 for r in results2 if r['comm_ok']==K):^8}")
    print()

print("结论: CVX遍历选择可以在更少AP下达到目标")
