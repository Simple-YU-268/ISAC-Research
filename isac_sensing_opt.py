"""改进感知波束设计 + 功率优化"""
import numpy as np

M, K, P, Nt = 16, 10, 4, 4
Pmax = 30
sigma2 = 0.5

def mmse_comm_beam(H, P_comm):
    """通信MMSE波束"""
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
    """感知匹配滤波波束 - 更好的检测性能"""
    M_sel, P, Nt = H_t.shape
    p_per_target = P_sens / P
    
    Z = np.zeros((M_sel, P, Nt), dtype=complex)
    for p in range(P):
        # 匹配滤波: 对每个目标单独波束
        h_t = H_t[:, p, :]  # (M_sel, Nt)
        # 分布式波束: 每个AP单独优化
        for m in range(M_sel):
            h_m = h_t[m, :]
            norm = np.sqrt(np.sum(np.abs(h_m)**2))
            if norm > 0:
                Z[m, p, :] = np.conj(h_m) / norm * np.sqrt(p_per_target / M_sel)
    
    return Z

def sensing_beam_mimo(H_t, P_sens):
    """感知MIMO波束 - 联合处理"""
    M_sel, P, Nt = H_t.shape
    
    # 将感知目标视为虚拟用户
    Hs = H_t.reshape(M_sel * Nt, P)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    
    try:
        Z_mat = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(Z_mat)**2)
        Z_mat = Z_mat * np.sqrt(P_sens / p)
        return Z_mat.reshape(M_sel, Nt, P).transpose(0, 2, 1)
    except:
        return sensing_beam_mf(H_t, P_sens)

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
    """感知SINR (公式c)"""
    M_sel, P, Nt = H_t.shape
    sinrs = []
    for p in range(P):
        h_eq = Z[:, p, :] * np.conj(H_t[:, p, :])
        signal = np.sum(np.abs(h_eq)**2)
        # 包括其他目标的干扰
        interference = 0
        for q in range(P):
            if q != p:
                h_inter = Z[:, q, :] * np.conj(H_t[:, p, :])
                interference += np.sum(np.abs(h_inter)**2)
        noise = sigma2 * np.sum(np.abs(Z[:, p, :])**2)
        sinrs.append(10 * np.log10(signal / (interference + noise + 1e-10) + 1e-10))
    return np.array(sinrs)

def compute_crb(H_t, Z):
    M_sel, P, Nt = H_t.shape
    crbs = []
    for p in range(P):
        h_eq = Z[:, p, :] * np.conj(H_t[:, p, :])
        power = np.sum(np.abs(h_eq)**2)
        crb = 1 / (power + 0.1)
        crbs.append(crb)
    return np.array(crbs)

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

print("=== 感知波束优化测试 ===\n")

# 测试不同功率分配
for P_comm, P_sens in [(24, 6), (21, 9), (18, 12), (15, 15)]:
    results = []
    for _ in range(100):
        H_u, H_t = generate_channel()
        
        # AP选择 (4个)
        signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
        total_signal = signal_power.sum(axis=1)
        selected = np.argsort(-total_signal)[:4]
        ap_mask = np.zeros(M, dtype=bool)
        ap_mask[selected] = True
        
        H_u_sel = H_u[ap_mask, :, :]
        H_t_sel = H_t[ap_mask, :, :]
        
        # 通信 + 感知波束
        W = mmse_comm_beam(H_u_sel, P_comm)
        Z = sensing_beam_mimo(H_t_sel, P_sens)
        
        comm_sinr = compute_comm_sinr(H_u_sel, W)
        sensing_sinr = compute_sensing_sinr(H_t_sel, Z)
        crb = compute_crb(H_t_sel, Z)
        
        total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
        
        results.append({
            'comm_ok': all(s >= 0 for s in comm_sinr),
            'sensing_ok': all(s >= 3 for s in sensing_sinr),
            'crb_ok': all(c < 10 for c in crb),
            'power': total_pwr
        })
    
    print(f"功率分配 通信{P_comm}W / 感知{P_sens}W:")
    print(f"  通信SINR≥0dB: {sum(1 for r in results if r['comm_ok'])}/100")
    print(f"  感知SINR≥3dB: {sum(1 for r in results if r['sensing_ok'])}/100")
    print(f"  CRB<10: {sum(1 for r in results if r['crb_ok'])}/100")
    print(f"  完全满足: {sum(1 for r in results if r['comm_ok'] and r['sensing_ok'] and r['crb_ok'] and r['power']<=30)}/100")
    print()

# 尝试更多AP
print("=== 更多AP + 感知功率 ===")
for N_ap in [5, 6, 8]:
    results = []
    for _ in range(100):
        H_u, H_t = generate_channel()
        
        signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
        total_signal = signal_power.sum(axis=1)
        selected = np.argsort(-total_signal)[:N_ap]
        ap_mask = np.zeros(M, dtype=bool)
        ap_mask[selected] = True
        
        H_u_sel = H_u[ap_mask, :, :]
        H_t_sel = H_t[ap_mask, :, :]
        
        W = mmse_comm_beam(H_u_sel, 18)
        Z = sensing_beam_mimo(H_t_sel, 12)
        
        comm_sinr = compute_comm_sinr(H_u_sel, W)
        sensing_sinr = compute_sensing_sinr(H_t_sel, Z)
        crb = compute_crb(H_t_sel, Z)
        
        results.append({
            'comm_ok': all(s >= 0 for s in comm_sinr),
            'sensing_ok': all(s >= 3 for s in sensing_sinr),
            'crb_ok': all(c < 10 for c in crb),
            'power': np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
        })
    
    print(f"N_ap={N_ap} + 18/12功率:")
    print(f"  完全满足: {sum(1 for r in results if r['comm_ok'] and r['sensing_ok'] and r['crb_ok'] and r['power']<=30)}/100")
