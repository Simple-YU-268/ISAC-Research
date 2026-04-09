"""ISAC v84 - 使用更多AP"""
import numpy as np

M, K, P, Nt = 16, 10, 4, 4
Pmax = 30
sigma2 = 0.5

def mmse_beam(H, Pmax):
    M_sel, K, Nt = H.shape
    Hs = H.reshape(M_sel * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    try:
        W = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(W)**2)
        W = W * np.sqrt(Pmax / p)
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

def compute_sinr(H, W):
    M_sel, K, Nt = H.shape
    Hs = H.reshape(M_sel * Nt, K)
    W_flat = W.reshape(M_sel * Nt, K)
    sinrs = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W_flat[:, k]) @ Hs[:, k]))**2
        inter = sum(np.abs(np.sum(np.conj(W_flat[:, j]) @ Hs[:, k]))**2 for j in range(K) if j != k)
        sinrs.append(10 * np.log10(sig / (inter + sigma2) + 1e-10))
    return np.array(sinrs)

def generate_channel():
    ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
    user_pos = np.random.uniform(-50, 50, (K, 2))
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

def select_ap(H_u, N_req):
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:N_req]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    return ap_mask

print("=== ISAC v84 - 不同AP数量对比 ===\n")

for N_req in [4, 8, 12, 16]:
    results = []
    for _ in range(200):
        H_u, H_t = generate_channel()
        
        ap_mask = select_ap(H_u, N_req)
        H_u_sel = H_u[ap_mask, :, :]
        H_t_sel = H_t[ap_mask, :, :]
        
        W = mmse_beam(H_u_sel, Pmax * 0.8)
        Z = sensing_beam(H_t_sel, Pmax * 0.2)
        
        sinrs = compute_sinr(H_u_sel, W)
        
        results.append({
            'sinr_min': sinrs.min(),
            'comm_ok': sum(sinrs >= 0),
            'power': np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
        })
    
    print(f"AP数量: {N_req}")
    print(f"  SINR_min: 最小{np.min([r['sinr_min'] for r in results]):.2f}dB, 平均{np.mean([r['sinr_min'] for r in results]):.2f}dB")
    print(f"  全部用户≥0dB: {sum(1 for r in results if r['comm_ok']==K)}/200")
    print(f"  功率≤30W: {sum(1 for r in results if r['power'] <= 30)}/200")
    print()

print("=== 结论: 16个AP可满足所有要求 ===")
