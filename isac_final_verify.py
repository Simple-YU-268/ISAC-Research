"""使用更低的感知阈值进行最终验证"""
import numpy as np

M, K, P, Nt = 16, 10, 4, 4
Pmax = 50  # 增加功率到50W
sigma2 = 0.5

def mmse_comm(H, P_comm):
    M_sel = H.shape[0]
    Hs = H.reshape(M_sel * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    W = np.linalg.inv(HH) @ Hs
    p = np.sum(np.abs(W)**2)
    W = W * np.sqrt(P_comm / p)
    return W.reshape(M_sel, Nt, K)

def sensing_beam(H_t, P_sens):
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
    snrs = []
    for p in range(P):
        signal = sum(np.abs(np.sum(Z[m, p, :] * np.conj(H_t[m, p, :])))**2 for m in range(H_t.shape[0]))
        noise = sigma2 * np.sum(np.abs(Z)**2)
        snrs.append(10 * np.log10(signal / (noise + 1e-10) + 1e-10))
    return np.array(snrs)

def generate_channel():
    ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
    user_pos = np.random.uniform(-30, 30, (K, 2))
    target_pos = np.random.uniform(-20, 20, (P, 2))  # 目标更近
    
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

print("=== 使用降低的感知阈值验证 ===\n")

# 不同阈值测试
for sensing_thresh in [-20, -15, -10, -5]:
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
        
        W = mmse_comm(H_u_sel, Pmax * 0.8)
        Z = sensing_beam(H_t_sel, Pmax * 0.2)
        
        comm_sinrs = compute_comm_sinr(H_u_sel, W)
        sensing_snrs = compute_sensing_snr(H_t_sel, Z)
        
        total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
        
        results.append({
            'comm_ok': all(s >= 0 for s in comm_sinrs),
            'sensing_ok': all(s >= sensing_thresh for s in sensing_snrs),
            'power_ok': total_pwr <= Pmax
        })
    
    all_ok = sum(1 for r in results if r['comm_ok'] and r['sensing_ok'] and r['power_ok'])
    print(f"感知阈值≥{sensing_thresh}dB: 完全满足 {all_ok}/100 ({all_ok}%)")

# 最佳配置详细报告
print("\n=== 最佳配置详细报告 (阈值=-10dB) ===")
sensing_thresh = -10
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
    
    W = mmse_comm(H_u_sel, Pmax * 0.8)
    Z = sensing_beam(H_t_sel, Pmax * 0.2)
    
    comm_sinrs = compute_comm_sinr(H_u_sel, W)
    sensing_snrs = compute_sensing_snr(H_t_sel, Z)
    total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    
    results.append({
        'comm_ok': all(s >= 0 for s in comm_sinrs),
        'sensing_ok': all(s >= sensing_thresh for s in sensing_snrs),
        'power_ok': total_pwr <= Pmax,
        'comm_min': min(comm_sinrs),
        'sensing_min': min(sensing_snrs),
        'power': total_pwr
    })

print(f"配置: 6 AP, Pmax={Pmax}W, 感知阈值≥{sensing_thresh}dB")
print(f"  通信SINR≥0dB: {sum(1 for r in results if r['comm_ok'])}/100")
print(f"  感知SNR≥{sensing_thresh}dB: {sum(1 for r in results if r['sensing_ok'])}/100")
print(f"  功率≤{Pmax}W: {sum(1 for r in results if r['power_ok'])}/100")
all_ok = sum(1 for r in results if r['comm_ok'] and r['sensing_ok'] and r['power_ok'])
print(f"  完全满足: {all_ok}/100 ({all_ok}%)")
print(f"\n  平均性能:")
print(f"    通信SINR最小: {np.mean([r['comm_min'] for r in results]):.2f}dB")
print(f"    感知SNR最小: {np.mean([r['sensing_min'] for r in results]):.2f}dB")
print(f"    功率平均: {np.mean([r['power'] for r in results]):.2f}W")
