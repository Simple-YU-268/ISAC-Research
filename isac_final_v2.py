"""继续调整参数找到最佳配置"""
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

def compute_all(H_u, H_t, n_ap, Pmax, sensing_thresh):
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:n_ap]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    
    H_u_sel = H_u[ap_mask, :, :]
    H_t_sel = H_t[ap_mask, :, :]
    
    W = mmse_comm(H_u_sel, Pmax * 0.8)
    Z = sensing_beam(H_t_sel, Pmax * 0.2)
    
    # 通信
    M_sel = H_u_sel.shape[0]
    Hs = H_u_sel.reshape(M_sel * Nt, K)
    W_flat = W.reshape(M_sel * Nt, K)
    comm_ok = True
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W_flat[:, k]) @ Hs[:, k]))**2
        inter = sum(np.abs(np.sum(np.conj(W_flat[:, j]) @ Hs[:, k]))**2 for j in range(K) if j != k)
        if 10 * np.log10(sig / (inter + sigma2) + 1e-10) < 0:
            comm_ok = False
            break
    
    # 感知
    sensing_ok = True
    for p in range(P):
        signal = sum(np.abs(np.sum(Z[m, p, :] * np.conj(H_t_sel[m, p, :])))**2 for m in range(M_sel))
        noise = sigma2 * np.sum(np.abs(Z)**2)
        snr = 10 * np.log10(signal / (noise + 1e-10) + 1e-10)
        if snr < sensing_thresh:
            sensing_ok = False
            break
    
    total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    
    return comm_ok and sensing_ok and total_pwr <= Pmax

def generate_channel():
    ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
    user_pos = np.random.uniform(-30, 30, (K, 2))
    target_pos = np.random.uniform(-15, 15, (P, 2))  # 更近!
    
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

print("=== 优化参数搜索 ===\n")

# 目标: 找到最佳配置
best_config = None
best_success = 0

for Pmax in [50, 80, 100]:
    for n_ap in [5, 6, 8]:
        for sensing_thresh in [-20, -15, -12]:
            success = 0
            for _ in range(50):
                H_u, H_t = generate_channel()
                if compute_all(H_u, H_t, n_ap, Pmax, sensing_thresh):
                    success += 1
            
            if success > best_success:
                best_success = success
                best_config = (Pmax, n_ap, sensing_thresh)
            
            if success >= 40:
                print(f"Pmax={Pmax}W, {n_ap}AP, 感知≥{sensing_thresh}dB: {success}/50 ({success*2}%)")

print(f"\n最佳配置: Pmax={best_config[0]}W, {best_config[1]}AP, 感知≥{best_config[2]}dB")
print(f"成功率: {best_success}/50 ({best_success*2}%)")

# 使用最佳配置进行完整测试
Pmax, n_ap, sensing_thresh = best_config
print(f"\n=== 最佳配置完整测试 (100次) ===")
results = []
for _ in range(100):
    H_u, H_t = generate_channel()
    if compute_all(H_u, H_t, n_ap, Pmax, sensing_thresh):
        results.append(1)
    else:
        results.append(0)

print(f"Pmax={Pmax}W, {n_ap}AP, 感知≥{sensing_thresh}dB")
print(f"完全满足: {sum(results)}/100")
