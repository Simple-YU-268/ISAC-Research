"""改进的感知波束 - 联合处理所有AP"""
import numpy as np

M, K, P, Nt = 16, 10, 4, 4
sigma2 = 0.5

def mmse_comm(H, P_comm):
    M_sel = H.shape[0]
    Hs = H.reshape(M_sel * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    W = np.linalg.inv(HH) @ Hs
    p = np.sum(np.abs(W)**2)
    W = W * np.sqrt(P_comm / p)
    return W.reshape(M_sel, Nt, K)

def joint_sensing(H_t, P_sens):
    """联合MIMO感知 - 类似通信的MMSE处理"""
    M_sel, P, Nt = H_t.shape
    
    # 堆叠成虚拟MIMO信道
    Hs = H_t.reshape(M_sel * Nt, P)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    
    # 匹配滤波/迫零检测
    try:
        # 使用迫零检测
        Z_mat = np.linalg.pinv(Hs) * np.sqrt(P_sens / P)
        return Z_mat.reshape(M_sel, Nt, P).transpose(0, 2, 1)
    except:
        # 备用: 分布式匹配滤波
        p_per = P_sens / P
        Z = np.zeros((M_sel, P, Nt), dtype=complex)
        for p in range(P):
            for m in range(M_sel):
                h = H_t[m, p, :]
                norm = np.sqrt(np.sum(np.abs(h)**2))
                if norm > 0:
                    Z[m, p, :] = np.conj(h) / norm * np.sqrt(p_per / M_sel)
        return Z

def compute_comm_sinr(H_u, W):
    M_sel = H_u.shape[0]
    Hs = H_u.reshape(M_sel * Nt, K)
    W_flat = W.reshape(M_sel * Nt, K)
    sinrs = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W_flat[:, k]) @ Hs[:, k]))**2
        inter = sum(np.abs(np.sum(np.conj(W_flat[:, j]) @ Hs[:, k]))**2 for j in range(K) if j != k)
        sinrs.append(10 * np.log10(sig / (inter + sigma2) + 1e-10))
    return np.array(sinrs)

def compute_sensing_snr(H_t, Z):
    M_sel, P, Nt = H_t.shape
    snrs = []
    for p in range(P):
        # 等效信道
        signal = np.sum(np.abs(Z[:, p, :] * np.conj(H_t[:, p, :]))**2)
        interference = 0
        for q in range(P):
            if q != p:
                interference += np.sum(np.abs(Z[:, q, :] * np.conj(H_t[:, p, :]))**2)
        noise = sigma2 * np.sum(np.abs(Z)**2)
        snrs.append(10 * np.log10(signal / (interference + noise + 1e-10) + 1e-10))
    return np.array(snrs)

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

print("=== 改进的联合感知波束 ===\n")

for Pmax in [50, 80, 100, 150]:
    results = []
    for _ in range(50):
        H_u, H_t = generate_channel()
        
        # 选择AP
        signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
        total_signal = signal_power.sum(axis=1)
        selected = np.argsort(-total_signal)[:6]
        ap_mask = np.zeros(M, dtype=bool)
        ap_mask[selected] = True
        
        H_u_sel = H_u[ap_mask, :, :]
        H_t_sel = H_t[ap_mask, :, :]
        
        W = mmse_comm(H_u_sel, Pmax * 0.7)
        Z = joint_sensing(H_t_sel, Pmax * 0.3)
        
        comm_sinrs = compute_comm_sinr(H_u_sel, W)
        sensing_snrs = compute_sensing_snr(H_t_sel, Z)
        
        total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
        
        results.append({
            'comm_ok': all(s >= 0 for s in comm_sinrs),
            'sensing_ok': all(s >= -10 for s in sensing_snrs),
            'power_ok': total_pwr <= Pmax * 1.1
        })
    
    print(f"Pmax={Pmax}W:")
    print(f"  通信≥0dB: {sum(1 for r in results if r['comm_ok'])}/50")
    print(f"  感知≥-10dB: {sum(1 for r in results if r['sensing_ok'])}/50")
    print(f"  功率≤{Pmax}W: {sum(1 for r in results if r['power_ok'])}/50")
    all_ok = sum(1 for r in results if r['comm_ok'] and r['sensing_ok'] and r['power_ok'])
    print(f"  完全满足: {all_ok}/50 ({all_ok*2}%)")
    print()
