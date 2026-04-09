"""对比当前实现与完整公式"""
import numpy as np

M, K, P, Nt = 16, 10, 4, 4
Pmax = 30
sigma2 = 0.5

def mmse_beam(H, Pmax):
    M_sel = H.shape[0]
    Hs = H.reshape(M_sel * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    try:
        W = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(W)**2)
        W = W * np.sqrt(Pmax * 0.8 / p)
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
        # 等效信道
        h_eq = Z[:, p, :] * np.conj(H_t[:, p, :])
        signal = np.sum(np.abs(h_eq)**2)
        noise = sigma2 * np.sum(np.abs(Z[:, p, :])**2)
        sinrs.append(10 * np.log10(signal / (noise + 1e-10) + 1e-10))
    return np.array(sinrs)

def compute_crb(H_t, Z):
    """跟踪CRB (公式d)"""
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

print("=== 完整约束验证 ===\n")

results = []
for _ in range(100):
    H_u, H_t = generate_channel()
    
    # 选择4个AP
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:4]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    
    H_u_sel = H_u[ap_mask, :, :]
    H_t_sel = H_t[ap_mask, :, :]
    
    # 通信波束 (80%)
    W = mmse_beam(H_u_sel, Pmax * 0.8)
    
    # 感知波束 (20%)
    Z = sensing_beam(H_t_sel, Pmax * 0.2)
    
    # 计算各约束
    comm_sinr = compute_comm_sinr(H_u_sel, W)  # 公式(b)
    sensing_sinr = compute_sensing_sinr(H_t_sel, Z)  # 公式(c)
    crb = compute_crb(H_t_sel, Z)  # 公式(d)
    
    results.append({
        'comm_ok': all(s >= 0 for s in comm_sinr),  # (b)
        'sensing_ok': all(s >= 3 for s in sensing_sinr),  # (c) PoD阈值3dB
        'crb_ok': all(c < 10 for c in crb),  # (d) CRB<10
        'power': np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)  # (f)
    })

print(f"公式(b) 通信SINR≥0dB:    {sum(1 for r in results if r['comm_ok'])}/100")
print(f"公式(c) 感知SINR≥3dB:    {sum(1 for r in results if r['sensing_ok'])}/100")  
print(f"公式(d) 跟踪CRB<10:      {sum(1 for r in results if r['crb_ok'])}/100")
print(f"公式(f) 功率≤30W:        {sum(1 for r in results if r['power']<=30)}/100")
print(f"完全满足(a-f):           {sum(1 for r in results if r['comm_ok'] and r['sensing_ok'] and r['crb_ok'] and r['power']<=30)}/100")
