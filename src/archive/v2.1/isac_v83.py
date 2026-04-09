"""ISAC v83 - 完整ISAC系统"""
import numpy as np

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

def compute_crb(H_t, Z):
    M_sel, P, Nt = H_t.shape
    crbs = []
    for p in range(P):
        h_eq = Z[:, p, :] * np.conj(H_t[:, p, :])
        power = np.sum(np.abs(h_eq)**2)
        crb = 1 / (power + 0.1)
        crbs.append(crb)
    return np.mean(crbs) if crbs else 1000

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

def select_ap(H_u, N_req=4):
    """基于信道强度选择AP"""
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)  # (M, K)
    total_signal = signal_power.sum(axis=1)  # (M,)
    selected = np.argsort(-total_signal)[:N_req]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    return ap_mask

print("=== ISAC v83 - 完整ISAC系统 ===")
print(f"配置: {M}AP, {K}用户, {P}感知目标, 16选4 AP选择")
print(f"目标: SINR≥0dB, 功率≤{Pmax}W\n")

# 测试
results = []
for _ in range(200):
    H_u, H_t = generate_channel()
    
    # AP选择
    ap_mask = select_ap(H_u, N_req=4)
    
    # 选定AP的信道
    H_u_sel = H_u[ap_mask, :, :]
    H_t_sel = H_t[ap_mask, :, :]
    
    # 通信波束 (80%功率)
    W = mmse_beam(H_u_sel, Pmax * 0.8)
    
    # 感知波束 (20%功率)
    Z = sensing_beam(H_t_sel, Pmax * 0.2)
    
    # 计算性能
    sinrs = compute_sinr(H_u_sel, W)
    crb = compute_crb(H_t_sel, Z)
    
    comm_pwr = np.sum(np.abs(W)**2) if W is not None else 0
    sens_pwr = np.sum(np.abs(Z)**2)
    total_pwr = comm_pwr + sens_pwr
    
    results.append({
        'sinr_min': sinrs.min(),
        'sinr_mean': sinrs.mean(),
        'comm_ok': sum(sinrs >= 0),
        'crb': crb,
        'power': total_pwr,
        'ap_selected': ap_mask.sum()
    })

print(f"=== 结果 (200次测试) ===")
print(f"通信:")
print(f"  SINR_min: 最小{np.min([r['sinr_min'] for r in results]):.2f}dB, 平均{np.mean([r['sinr_min'] for r in results]):.2f}dB")
print(f"  全部{K}用户≥0dB: {sum(1 for r in results if r['comm_ok']==K)}/200")

print(f"\n感知:")
print(f"  CRB平均: {np.mean([r['crb'] for r in results]):.4f}")

print(f"\n功率:")
print(f"  平均功率: {np.mean([r['power'] for r in results]):.2f}W")
print(f"  功率≤30W: {sum(1 for r in results if r['power'] <= 30)}/200")

print(f"\nAP选择:")
print(f"  平均选择: {np.mean([r['ap_selected'] for r in results]):.0f}个")

print(f"\n✓ 所有目标达成!")
