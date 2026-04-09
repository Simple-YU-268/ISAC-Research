"""检查实际感知SINR值和阈值"""
import numpy as np

M, K, P, Nt = 16, 10, 4, 4
sigma2 = 0.5

def mmse_beam(H, Pmax):
    M_sel = H.shape[0]
    Hs = H.reshape(M_sel * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    W = np.linalg.inv(HH) @ Hs
    p = np.sum(np.abs(W)**2)
    W = W * np.sqrt(Pmax * 0.8 / p)
    return W.reshape(M_sel, Nt, K)

def sensing_beam(H_t, Pmax):
    M_sel, P, Nt = H_t.shape
    p_per = Pmax * 0.2 / P
    Z = np.zeros((M_sel, P, Nt), dtype=complex)
    for p in range(P):
        for m in range(M_sel):
            h = H_t[m, p, :]
            norm = np.sqrt(np.sum(np.abs(h)**2))
            if norm > 0:
                Z[m, p, :] = np.conj(h) / norm * np.sqrt(p_per / M_sel)
    return Z

def compute_sensing_sinr(H_t, Z):
    M_sel, P, Nt = H_t.shape
    sinrs = []
    for p in range(P):
        # 计算等效信噪比
        h_eq = np.sum(Z * np.conj(H_t), axis=2)  # (M_sel, P)
        signal = np.abs(h_eq[:, p])**2
        interference = np.sum(np.abs(h_eq[:, :]), axis=1) - signal
        snr = signal / (interference + sigma2)
        sinrs.append(10 * np.log10(np.mean(snr) + 1e-10))
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

# 测试不同功率下的感知SINR
print("=== 感知SINR vs 功率 (6 AP) ===\n")
for Pmax in [30, 50, 100]:
    sinrs_all = []
    for _ in range(100):
        H_u, H_t = generate_channel()
        
        signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
        total_signal = signal_power.sum(axis=1)
        selected = np.argsort(-total_signal)[:6]
        ap_mask = np.zeros(M, dtype=bool)
        ap_mask[selected] = True
        
        H_t_sel = H_t[ap_mask, :, :]
        Z = sensing_beam(H_t_sel, Pmax)
        
        sinrs = compute_sensing_sinr(H_t_sel, Z)
        sinrs_all.extend(sinrs)
    
    print(f"Pmax={Pmax}W:")
    print(f"  感知SINR: 最小={np.min(sinrs_all):.2f}dB, 平均={np.mean(sinrs_all):.2f}dB, 最大={np.max(sinrs_all):.2f}dB")
    print(f"  满足≥0dB: {sum(1 for s in sinrs_all if s>=0)}/{len(sinrs_all)}")
    print(f"  满足≥-5dB: {sum(1 for s in sinrs_all if s>=-5)}/{len(sinrs_all)}")
    print(f"  满足≥-10dB: {sum(1 for s in sinrs_all if s>=-10)}/{len(sinrs_all)}")
    print()
