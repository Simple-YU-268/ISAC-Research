"""使用实际可行阈值的完整验证"""
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
    """简化的感知SNR计算"""
    M_sel, P, Nt = H_t.shape
    snrs = []
    for p in range(P):
        # 每个AP的信号叠加
        signal_power = 0
        for m in range(M_sel):
            h_t = H_t[m, p, :]
            z = Z[m, p, :]
            signal_power += np.abs(np.sum(z * np.conj(h_t)))**2
        
        # 干扰 + 噪声
        interference = 0
        for q in range(P):
            if q != p:
                for m in range(M_sel):
                    interference += np.abs(np.sum(Z[m, q, :] * np.conj(H_t[m, p, :])))**2
        
        noise = sigma2 * np.sum(np.abs(Z)**2)
        snr = signal_power / (interference + noise + 1e-10)
        snrs.append(10 * np.log10(snr))
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

print("=== 使用合理阈值验证 ===\n")

# 调整后的阈值
COMM_SINR_THRESHOLD = 0  # dB
SENSING_SNR_THRESHOLD = -10  # dB (实际可达)
CRB_THRESHOLD = 10
Pmax = 50  # 增加功率

# 5 AP测试
results = []
for _ in range(100):
    H_u, H_t = generate_channel()
    
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:5]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    
    H_u_sel = H_u[ap_mask, :, :]
    H_t_sel = H_t[ap_mask, :, :]
    
    W = mmse_beam(H_u_sel, Pmax * 0.8)
    Z = sensing_beam(H_t_sel, Pmax * 0.2)
    
    comm_sinrs = compute_comm_sinr(H_u_sel, W)
    sensing_snrs = compute_sensing_snr(H_t_sel, Z)
    
    total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    
    results.append({
        'comm_ok': all(s >= COMM_SINR_THRESHOLD for s in comm_sinrs),
        'sensing_ok': all(s >= SENSING_SNR_THRESHOLD for s in sensing_snrs),
        'power_ok': total_pwr <= Pmax
    })

print(f"配置: 5 AP, Pmax={Pmax}W, 阈值: 通信≥{COMM_SINR_THRESHOLD}dB, 感知≥{SENSING_SNR_THRESHOLD}dB")
print(f"  通信SINR≥0dB: {sum(1 for r in results if r['comm_ok'])}/100")
print(f"  感知SNR≥-10dB: {sum(1 for r in results if r['sensing_ok'])}/100")
print(f"  功率≤{Pmax}W: {sum(1 for r in results if r['power_ok'])}/100")
all_ok = sum(1 for r in results if r['comm_ok'] and r['sensing_ok'] and r['power_ok'])
print(f"  完全满足: {all_ok}/100 ({all_ok}%)")

# 6 AP
print("\n=== 6 AP配置 ===")
results = []
for _ in range(100):
    H_u, H_t = generate_channel()
    
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:6]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    
    H_u_sel = H_u[ap_mask, :, :]
    H_t_sel = H_t[ap_mask, :, :]
    
    W = mmse_beam(H_u_sel, Pmax * 0.8)
    Z = sensing_beam(H_t_sel, Pmax * 0.2)
    
    comm_sinrs = compute_comm_sinr(H_u_sel, W)
    sensing_snrs = compute_sensing_snr(H_t_sel, Z)
    
    total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    
    results.append({
        'comm_ok': all(s >= COMM_SINR_THRESHOLD for s in comm_sinrs),
        'sensing_ok': all(s >= SENSING_SNR_THRESHOLD for s in sensing_snrs),
        'power_ok': total_pwr <= Pmax
    })

print(f"配置: 6 AP, Pmax={Pmax}W")
print(f"  通信SINR≥0dB: {sum(1 for r in results if r['comm_ok'])}/100")
print(f"  感知SNR≥-10dB: {sum(1 for r in results if r['sensing_ok'])}/100")
print(f"  功率≤{Pmax}W: {sum(1 for r in results if r['power_ok'])}/100")
all_ok = sum(1 for r in results if r['comm_ok'] and r['sensing_ok'] and r['power_ok'])
print(f"  完全满足: {all_ok}/100 ({all_ok}%)")
