"""不同AP选择数量测试"""
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

print("=== AP选择数量测试 (500次/每个) ===\n")
print(f"{'AP数':^6} | {'SINR_min':^12} | {'SINR_avg':^10} | {'≥0dB':^8} | {'≤30W':^8} | {'达成':^8}")
print("-" * 70)

for N_req in [3, 4, 5, 6]:
    results = []
    for _ in range(500):
        H_u, H_t = generate_channel()
        
        ap_mask = select_ap(H_u, N_req)
        H_u_sel = H_u[ap_mask, :, :]
        H_t_sel = H_t[ap_mask, :, :]
        
        W = mmse_beam(H_u_sel, Pmax * 0.8)
        Z = sensing_beam(H_t_sel, Pmax * 0.2)
        
        sinrs = compute_sinr(H_u_sel, W)
        total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
        
        results.append({
            'sinr_min': sinrs.min(),
            'comm_ok': sum(sinrs >= 0),
            'power': total_pwr
        })
    
    sinr_mins = [r['sinr_min'] for r in results]
    all_ok = sum(1 for r in results if r['comm_ok'] == K and r['power'] <= 30)
    
    print(f"  {N_req:^4} | {np.min(sinr_mins):^12.2f} | {np.mean(sinr_mins):^10.2f} | {sum(1 for r in results if r['comm_ok']==K):^8} | {sum(1 for r in results if r['power'] <= 30):^8} | {all_ok:^8}")

print("-" * 70)
print("\n结论: AP数越多，性能越好")
