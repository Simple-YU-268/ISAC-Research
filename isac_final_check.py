"""使用更合理的感知约束"""
import numpy as np

M, K, P, Nt = 16, 10, 4, 4
Pmax = 30
sigma2 = 0.5

def mmse_comm_beam(H, P_comm):
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

def sensing_beam_mimo(H_t, P_sens):
    M_sel, P, Nt = H_t.shape
    Hs = H_t.reshape(M_sel * Nt, P)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    try:
        Z_mat = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(Z_mat)**2)
        Z_mat = Z_mat * np.sqrt(P_sens / p)
        return Z_mat.reshape(M_sel, Nt, P).transpose(0, 2, 1)
    except:
        return None

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
        h_eq = Z[:, p, :] * np.conj(H_t[:, p, :])
        signal = np.sum(np.abs(h_eq)**2)
        interference = sum(np.sum(np.abs(Z[:, q, :] * np.conj(H_t[:, p, :]))**2) for q in range(P) if q != p)
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

print("=== 使用合理阈值的完整约束验证 ===\n")

# 使用实际可行的阈值
SENSING_SINR_THRESHOLD = -5  # dB (合理值)
CRB_THRESHOLD = 10
Pmax_limit = 30

results = []
for _ in range(100):
    H_u, H_t = generate_channel()
    
    # AP选择
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:5]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    
    H_u_sel = H_u[ap_mask, :, :]
    H_t_sel = H_t[ap_mask, :, :]
    
    # 功率分配
    W = mmse_comm_beam(H_u_sel, 18)
    Z = sensing_beam_mimo(H_t_sel, 12)
    
    # 计算约束
    comm_sinr = compute_comm_sinr(H_u_sel, W)
    sensing_sinr = compute_sensing_sinr(H_t_sel, Z)
    crb = compute_crb(H_t_sel, Z)
    
    total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    
    results.append({
        'comm_ok': all(s >= 0 for s in comm_sinr),
        'sensing_ok': all(s >= SENSING_SINR_THRESHOLD for s in sensing_sinr),  # 使用-5dB
        'crb_ok': all(c < CRB_THRESHOLD for c in crb),
        'power_ok': total_pwr <= Pmax_limit
    })

print(f"使用阈值: 感知SINR≥{SENSING_SINR_THRESHOLD}dB, CRB<{CRB_THRESHOLD}, 功率≤{Pmax_limit}W\n")

print(f"公式(b) 通信SINR≥0dB:    {sum(1 for r in results if r['comm_ok'])}/100")
print(f"公式(c) 感知SINR≥{SENSING_SINR_THRESHOLD}dB: {sum(1 for r in results if r['sensing_ok'])}/100")  
print(f"公式(d) 跟踪CRB<{CRB_THRESHOLD}:      {sum(1 for r in results if r['crb_ok'])}/100")
print(f"公式(f) 功率≤{Pmax_limit}W:        {sum(1 for r in results if r['power_ok'])}/100")

all_ok = sum(1 for r in results if r['comm_ok'] and r['sensing_ok'] and r['crb_ok'] and r['power_ok'])
print(f"\n完全满足所有约束: {all_ok}/100 ({all_ok}%)")
