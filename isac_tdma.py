"""时分复用方案: 通信和感知分时进行"""
import numpy as np
import torch

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
        W = W * np.sqrt(Pmax / p)
        return W.reshape(M_sel, Nt, K)
    except:
        return None

def sensing_beam_single(H_t, P_sens):
    """只感知一个目标,分配所有功率"""
    M_sel = P = 1  # 单目标
    Nt = H_t.shape[2]
    
    # 分布式最大比合并
    Z = np.zeros((M_sel, 1, Nt), dtype=complex)
    for m in range(M_sel):
        h = H_t[m, 0, :]
        norm = np.sqrt(np.sum(np.abs(h)**2))
        if norm > 0:
            Z[m, 0, :] = np.conj(h) / norm * np.sqrt(P_sens / M_sel)
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

def compute_sensing_sinr_single(H_t, Z):
    """单目标感知SINR"""
    M_sel = 1
    signal = sum(np.abs(np.sum(Z[m, 0, :] * np.conj(H_t[m, 0, :])))**2 for m in range(M_sel))
    noise = sigma2 * np.sum(np.abs(Z)**2)
    return 10 * np.log10(signal / (noise + 1e-10) + 1e-10)

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

print("=== 时分复用方案 ===")
print("时隙1: 通信 (30W, 5 AP)")
print("时隙2: 感知单目标 (30W, 5 AP)")
print()

# 时隙1: 通信
results_comm = []
for _ in range(100):
    H_u, H_t = generate_channel()
    
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:5]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    
    H_u_sel = H_u[ap_mask, :, :]
    W = mmse_beam(H_u_sel, 30)
    
    comm_sinrs = compute_comm_sinr(H_u_sel, W)
    results_comm.append(all(s >= 0 for s in comm_sinrs))

print(f"通信时隙: 全部10用户≥0dB: {sum(results_comm)}/100")

# 时隙2: 感知 (每个目标依次)
results_sensing = []
for _ in range(100):
    H_u, H_t = generate_channel()
    
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:5]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    
    H_t_sel = H_t[ap_mask, :, :]
    
    # 依次感知4个目标
    sinrs_all = []
    for p in range(P):
        Z = sensing_beam_single(H_t_sel[:, p:p+1, :], 30)
        sinr = compute_sensing_sinr_single(H_t_sel[:, p:p+1, :], Z)
        sinrs_all.append(sinr)
    
    results_sensing.append(all(s >= 0 for s in sinrs_all))  # 单目标感知更容易

print(f"感知时隙: 每目标≥0dB: {sum(results_sensing)}/100")
print(f"\n如果允许时隙分别优化:")
print(f"  通信约束满足: {sum(results_comm)}%")
print(f"  感知约束满足: {sum(results_sensing)}%")
