"""快速检查: 神经网络选择 + 感知功率增加"""
import numpy as np

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

def compute_all(H_u, H_t, ap_indices, P_comm, P_sens):
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[ap_indices] = True
    
    H_u_sel = H_u[ap_mask, :, :]
    H_t_sel = H_t[ap_mask, :, :]
    
    W = mmse_beam(H_u_sel, P_comm)
    Z = sensing_beam_mf(H_t_sel, P_sens)
    
    # 通信SINR
    M_sel = H_u_sel.shape[0]
    Hs = H_u_sel.reshape(M_sel * Nt, K)
    W_flat = W.reshape(M_sel * Nt, K)
    comm_sinrs = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W_flat[:, k]) @ Hs[:, k]))**2
        inter = sum(np.abs(np.sum(np.conj(W_flat[:, j]) @ Hs[:, k]))**2 for j in range(K) if j != k)
        comm_sinrs.append(10 * np.log10(sig / (inter + sigma2) + 1e-10))
    
    # 感知SINR
    sensing_sinrs = []
    for p in range(P):
        signal = sum(np.abs(np.sum(Z[m, p, :] * np.conj(H_t_sel[m, p, :])))**2 for m in range(M_sel))
        interference = sum(np.abs(np.sum(Z[m, q, :] * np.conj(H_t_sel[m, p, :])))**2 for m in range(M_sel) for q in range(P) if q != p)
        noise = sigma2 * np.sum(np.abs(Z)**2)
        sensing_sinrs.append(10 * np.log10(signal / (interference + noise + 1e-10) + 1e-10))
    
    total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    
    return {
        'comm_ok': all(s >= 0 for s in comm_sinrs),
        'sensing_ok': all(s >= -3 for s in sensing_sinrs),  # 合理阈值
        'power_ok': total_pwr <= Pmax,
        'comm_min': min(comm_sinrs),
        'sensing_min': min(sensing_sinrs),
        'power': total_pwr
    }

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

# 使用更多AP和更多感知功率
results = []
for _ in range(100):
    H_u, H_t = generate_channel()
    
    # 选择6个AP
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:6]
    
    # 通信16W + 感知14W
    result = compute_all(H_u, H_t, selected, 16, 14)
    results.append(result)

print(f"配置: 6 AP, 通信16W / 感知14W, 感知阈值-3dB")
print(f"  通信SINR≥0dB: {sum(1 for r in results if r['comm_ok'])}/100")
print(f"  感知SINR≥-3dB: {sum(1 for r in results if r['sensing_ok'])}/100")
print(f"  功率≤30W: {sum(1 for r in results if r['power_ok'])}/100")
all_ok = sum(1 for r in results if r['comm_ok'] and r['sensing_ok'] and r['power_ok'])
print(f"  完全满足: {all_ok}/100")

# 尝试更多AP
for n_ap in [6, 8, 10]:
    for p_comm in [15, 18, 20]:
        p_sens = Pmax - p_comm
        results = []
        for _ in range(50):
            H_u, H_t = generate_channel()
            signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
            total_signal = signal_power.sum(axis=1)
            selected = np.argsort(-total_signal)[:n_ap]
            
            result = compute_all(H_u, H_t, selected, p_comm, p_sens)
            results.append(result)
        
        all_ok = sum(1 for r in results if r['comm_ok'] and r['sensing_ok'] and r['power_ok'])
        if all_ok >= 40:
            print(f"\n更好配置: {n_ap}AP, 通信{p_comm}W/感知{p_sens}W -> 完全满足 {all_ok}/50")
