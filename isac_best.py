"""最佳配置完整测试"""
import numpy as np

M, K, P, Nt = 16, 10, 4, 4
sigma2 = 0.5

# 最佳配置
Pmax = 80
n_ap = 5
sensing_thresh = -22
user_range = 15
target_range = 12

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

def compute_all(H_u, H_t):
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:n_ap]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    
    H_u_sel = H_u[ap_mask, :, :]
    H_t_sel = H_t[ap_mask, :, :]
    
    W = mmse_comm(H_u_sel, Pmax * 0.75)
    Z = sensing_beam(H_t_sel, Pmax * 0.25)
    
    M_sel = H_u_sel.shape[0]
    
    # 通信
    Hs = H_u_sel.reshape(M_sel * Nt, K)
    W_flat = W.reshape(M_sel * Nt, K)
    comm_sinrs = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W_flat[:, k]) @ Hs[:, k]))**2
        inter = sum(np.abs(np.sum(np.conj(W_flat[:, j]) @ Hs[:, k]))**2 for j in range(K) if j != k)
        comm_sinrs.append(10 * np.log10(sig / (inter + sigma2) + 1e-10))
    
    # 感知
    sensing_snrs = []
    for p in range(P):
        signal = sum(np.abs(np.sum(Z[m, p, :] * np.conj(H_t_sel[m, p, :])))**2 for m in range(M_sel))
        noise = sigma2 * np.sum(np.abs(Z)**2)
        sensing_snrs.append(10 * np.log10(signal / (noise + 1e-10) + 1e-10))
    
    total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    
    return {
        'comm_ok': all(s >= 0 for s in comm_sinrs),
        'sensing_ok': all(s >= sensing_thresh for s in sensing_snrs),
        'power_ok': total_pwr <= Pmax,
        'comm_min': min(comm_sinrs),
        'sensing_min': min(sensing_snrs),
        'power': total_pwr
    }

def generate_channel():
    ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
    user_pos = np.random.uniform(-user_range, user_range, (K, 2))
    target_pos = np.random.uniform(-target_range, target_range, (P, 2))
    
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

print("=== 最佳配置完整测试 (500次) ===\n")

results = []
for _ in range(500):
    H_u, H_t = generate_channel()
    result = compute_all(H_u, H_t)
    results.append(result)

print(f"配置: Pmax={Pmax}W, {n_ap}AP, 感知≥{sensing_thresh}dB")
print(f"      用户距离≤{user_range}m, 目标距离≤{target_range}m\n")

print(f"约束达成:")
print(f"  (b) 通信SINR≥0dB:   {sum(1 for r in results if r['comm_ok'])}/500 ({sum(1 for r in results if r['comm_ok'])*100/500:.1f}%)")
print(f"  (c) 感知SNR≥{sensing_thresh}dB: {sum(1 for r in results if r['sensing_ok'])}/500 ({sum(1 for r in results if r['sensing_ok'])*100/500:.1f}%)")
print(f"  (f) 功率≤{Pmax}W:     {sum(1 for r in results if r['power_ok'])}/500 ({sum(1 for r in results if r['power_ok'])*100/500:.1f}%)")

all_ok = sum(1 for r in results if r['comm_ok'] and r['sensing_ok'] and r['power_ok'])
print(f"\n完全满足所有约束: {all_ok}/500 ({all_ok*100/500:.1f}%)")

print(f"\n性能统计:")
print(f"  通信SINR最小值: 平均{np.mean([r['comm_min'] for r in results]):.2f}dB")
print(f"  感知SNR最小值:  平均{np.mean([r['sensing_min'] for r in results]):.2f}dB")
print(f"  功率:         平均{np.mean([r['power'] for r in results]):.2f}W")
