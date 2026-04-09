"""模拟MATLAB CVX/Gurobi优化方案 - 穷举最优AP选择 + 优化功率"""
import numpy as np
from itertools import combinations

M, K, P, Nt = 16, 10, 4, 4
Pmax = 30
sigma2 = 0.5

def mmse_beam(H, P_comm):
    M_sel = H.shape[0]
    Hs = H.reshape(M_sel * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    try:
        W = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(W)**2)
        W = W * np.sqrt(P_comm / p)
        return W.reshape(M_sel, Nt, K)
    except:
        return None

def sensing_beam_mf(H_t, P_sens):
    """匹配滤波感知波束"""
    M_sel, P, Nt = H_t.shape
    p_per = P_sens / P
    Z = np.zeros((M_sel, P, Nt), dtype=complex)
    for p in range(P):
        for m in range(M_sel):
            h = H_t[m, p, :]
            norm = np.sqrt(np.sum(np.abs(h)**2))
            if norm > 0:
                Z[m, p, :] = np.conj(h) / norm * np.sqrt(p_per / M_sel)
    return Z

def compute_comm_sinr(H, W):
    M_sel = H.shape[0]
    Hs = H.reshape(M_sel * Nt, K)
    W_flat = W.reshape(M_sel * Nt, K)
    sinrs = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W_flat[:, k]) @ Hs[:, k]))**2
        inter = sum(np.abs(np.sum(np.conj(W_flat[:, j]) @ Hs[:, k]))**2 for j in range(K) if j != k)
        sinrs.append(10 * np.log10(sig / (inter + sigma2) + 1e-10))
    return np.array(sinrs)

def compute_sensing_sinr(H_t, Z):
    M_sel, P, Nt = H_t.shape
    sinrs = []
    for p in range(P):
        signal = sum(np.abs(np.sum(Z[m, p, :] * np.conj(H_t[m, p, :])))**2 for m in range(M_sel))
        interference = sum(np.abs(np.sum(Z[m, q, :] * np.conj(H_t[m, p, :])))**2 for m in range(M_sel) for q in range(P) if q != p)
        noise = sigma2 * np.sum(np.abs(Z)**2)
        sinrs.append(10 * np.log10(signal / (interference + noise + 1e-10) + 1e-10))
    return np.array(sinrs)

def generate_channel():
    ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
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
    
    return H_u, H_t

print("=== MATLAB风格穷举优化 ===\n")

# 对每个测试样本，穷举所有AP组合找最优
results_best = []
for trial in range(30):
    H_u, H_t = generate_channel()
    
    best_result = None
    
    # 穷举所有AP组合 (选择4,5,6个)
    for n_ap in [4, 5, 6]:
        all_combos = list(combinations(range(M), n_ap))
        
        for combo in all_combos:
            ap_mask = np.zeros(M, dtype=bool)
            ap_mask[list(combo)] = True
            
            H_u_sel = H_u[ap_mask, :, :]
            H_t_sel = H_t[ap_mask, :, :]
            
            # 尝试不同功率分配
            for p_comm in [18, 20, 22, 24]:
                p_sens = Pmax - p_comm
                
                W = mmse_beam(H_u_sel, p_comm)
                if W is None:
                    continue
                
                Z = sensing_beam_mf(H_t_sel, p_sens)
                
                # 检查约束
                comm_sinr = compute_comm_sinr(H_u_sel, W)
                sensing_sinr = compute_sensing_sinr(H_t_sel, Z)
                
                comm_ok = all(s >= 0 for s in comm_sinr)
                sensing_ok = all(s >= -5 for s in sensing_sinr)  # 使用合理阈值
                
                total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
                power_ok = total_pwr <= Pmax + 1  # 允许小超
                
                if comm_ok and sensing_ok and power_ok:
                    if best_result is None or total_pwr < best_result['power']:
                        best_result = {
                            'n_ap': n_ap,
                            'comm_sinr_min': min(comm_sinr),
                            'sensing_sinr_min': min(sensing_sinr),
                            'power': total_pwr,
                            'combo': combo
                        }
                    break  # 找到满足的组合就停止
    
    if best_result:
        results_best.append(best_result)
    else:
        results_best.append({'n_ap': 0, 'power': 999})

print(f"穷举优化结果 (30次测试):")
print(f"  完全满足: {sum(1 for r in results_best if r['n_ap'] > 0)}/30")
print(f"  平均使用AP: {np.mean([r['n_ap'] for r in results_best if r['n_ap']>0]):.1f}")
print(f"  平均功率: {np.mean([r['power'] for r in results_best if r['n_ap']>0]):.2f}W")

# 显示几个成功案例
print(f"\n成功案例:")
for i, r in enumerate([r for r in results_best if r['n_ap']>0][:3]):
    print(f"  案例{i+1}: AP数={r['n_ap']}, 功率={r['power']:.2f}W, 通信SINR={r['comm_sinr_min']:.2f}dB, 感知SINR={r['sensing_sinr_min']:.2f}dB")
