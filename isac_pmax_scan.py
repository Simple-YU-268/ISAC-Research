"""扫描不同Pmax，找到满足所有约束的最小功率"""
import numpy as np
from itertools import combinations

M, K, P, Nt = 16, 10, 4, 4
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
    p_per = Pmax * 0.2 / P
    Z = np.zeros((M_sel, P, Nt), dtype=complex)
    for p in range(P):
        for m in range(M_sel):
            h = H_t[m, p, :]
            norm = np.sqrt(np.sum(np.abs(h)**2))
            if norm > 0:
                Z[m, p, :] = np.conj(h) / norm * np.sqrt(p_per / M_sel)
    return Z

def compute_all_constraints(H_u, H_t, ap_indices, Pmax):
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[ap_indices] = True
    
    H_u_sel = H_u[ap_mask, :, :]
    H_t_sel = H_t[ap_mask, :, :]
    
    W = mmse_beam(H_u_sel, Pmax * 0.8)
    Z = sensing_beam(H_t_sel, Pmax * 0.2)
    
    # 通信SINR
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
    
    # 感知SINR (简化)
    sensing_ok = True
    for p in range(P):
        h_eq = Z[:, p, :] * np.conj(H_t_sel[:, p, :])
        signal = np.sum(np.abs(h_eq)**2)
        if signal < 1:
            sensing_ok = False
            break
    
    total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    
    return comm_ok and sensing_ok and total_pwr <= Pmax * 1.1

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

print("=== 扫描最小Pmax (5 AP) ===\n")

for Pmax_test in [30, 40, 50, 60, 70, 80, 100]:
    success = 0
    for _ in range(50):
        H_u, H_t = generate_channel()
        
        # 选择5个AP
        signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
        total_signal = signal_power.sum(axis=1)
        selected = np.argsort(-total_signal)[:5]
        
        if compute_all_constraints(H_u, H_t, selected, Pmax_test):
            success += 1
    
    print(f"Pmax={Pmax_test}W: 满足 {success}/50 ({success*2}%)")

print("\n=== 扫描最小Pmax (6 AP) ===")
for Pmax_test in [30, 40, 50, 60]:
    success = 0
    for _ in range(50):
        H_u, H_t = generate_channel()
        
        signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
        total_signal = signal_power.sum(axis=1)
        selected = np.argsort(-total_signal)[:6]
        
        if compute_all_constraints(H_u, H_t, selected, Pmax_test):
            success += 1
    
    print(f"Pmax={Pmax_test}W (6 AP): 满足 {success}/50 ({success*2}%)")
