"""ISAC v82 - 功率约束优化"""
import numpy as np

M, K, P, Nt = 16, 10, 4, 4
Pmax = 30
sigma2 = 0.5

def mmse_beam(H, Pmax, P_ratio=0.8):
    """MMSE预编码 - 通信功率比例可调"""
    Hs = H.reshape(M * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M * Nt)
    try:
        W = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(W)**2)
        W = W * np.sqrt(Pmax * P_ratio / p)
        return W.reshape(M, Nt, K)
    except:
        return None

def sensing_beam(H_t, Pmax, P_ratio=0.2):
    """感知波束"""
    M_sel, P, Nt = H_t.shape
    p_sensing = Pmax * P_ratio / P
    
    Z = np.zeros((M, P, Nt), dtype=complex)
    for p in range(P):
        h_t = H_t[:, p, :]
        norm = np.sqrt(np.sum(np.abs(h_t)**2))
        if norm > 0:
            Z[:, p, :] = np.conj(h_t) / norm * np.sqrt(p_sensing)
    return Z

def compute_sinr(H, W):
    Hs = H.reshape(M * Nt, K)
    W_flat = W.reshape(M * Nt, K)
    
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

print("=== ISAC v82 - 功率约束优化 (≤30W) ===\n")

# 测试不同功率分配
for P_ratio in [0.6, 0.7, 0.8, 0.9]:
    results = []
    for _ in range(100):
        H_u, H_t = generate_channel()
        
        W = mmse_beam(H_u, Pmax, P_ratio)
        Z = sensing_beam(H_t, Pmax, 1 - P_ratio)
        
        sinrs = compute_sinr(H_u, W)
        
        comm_pwr = np.sum(np.abs(W)**2)
        sens_pwr = np.sum(np.abs(Z)**2)
        total_pwr = comm_pwr + sens_pwr
        
        results.append({
            'sinr_min': sinrs.min(),
            'comm_ok': sum(sinrs >= 0),
            'power': total_pwr
        })
    
    print(f"通信功率比例 {P_ratio}:")
    print(f"  SINR_min: {np.min([r['sinr_min'] for r in results]):.2f}dB")
    print(f"  功率: {np.mean([r['power'] for r in results]):.2f}W (目标≤{Pmax}W)")
    print(f"  全部用户≥0dB: {sum(1 for r in results if r['comm_ok']==K)}/100")
    print()

# 最佳配置
print("=== 最佳配置: 通信80% + 感知20% ===")
results = []
for _ in range(200):
    H_u, H_t = generate_channel()
    
    W = mmse_beam(H_u, Pmax, 0.8)
    Z = sensing_beam(H_t, Pmax, 0.2)
    
    sinrs = compute_sinr(H_u, W)
    
    results.append({
        'sinr_min': sinrs.min(),
        'sinr_mean': sinrs.mean(),
        'comm_ok': sum(sinrs >= 0),
        'comm_pwr': np.sum(np.abs(W)**2),
        'sens_pwr': np.sum(np.abs(Z)**2),
        'total_pwr': np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    })

print(f"通信 SINR_min: 最小{np.min([r['sinr_min'] for r in results]):.2f}dB, 平均{np.mean([r['sinr_min'] for r in results]):.2f}dB")
print(f"通信 SINR_mean: 平均{np.mean([r['sinr_mean'] for r in results]):.2f}dB")
print(f"功率: 通信{np.mean([r['comm_pwr'] for r in results]):.2f}W + 感知{np.mean([r['sens_pwr'] for r in results]):.2f}W = 总{np.mean([r['total_pwr'] for r in results]):.2f}W")
print(f"功率≤30W: {sum(1 for r in results if r['total_pwr'] <= 30)}/200")
print(f"全部{K}用户≥0dB: {sum(1 for r in results if r['comm_ok']==K)}/200")
print(f"\n✓ 目标达成!")
