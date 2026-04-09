"""简单感知波束 - 每个目标分配固定功率"""
import numpy as np

M, K, P, Nt = 16, 10, 4, 4
Pmax = 30

def mmse_comm_beam(H, P_comm):
    M_sel = H.shape[0]
    sigma2 = 0.5
    Hs = H.reshape(M_sel * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    try:
        W = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(W)**2)
        W = W * np.sqrt(P_comm / p)
        return W.reshape(M_sel, Nt, K)
    except:
        return None

def sensing_beam_simple(H_t, P_sens):
    """简单方法: 每个AP对每个目标发射最大功率波束"""
    M_sel, P, Nt = H_t.shape
    p_per_target = P_sens / P
    p_per_ap_target = p_per_target / M_sel
    
    Z = np.zeros((M_sel, P, Nt), dtype=complex)
    for p in range(P):
        for m in range(M_sel):
            h = H_t[m, p, :]
            norm = np.sqrt(np.sum(np.abs(h)**2))
            if norm > 1e-6:
                Z[m, p, :] = np.conj(h) / norm * np.sqrt(p_per_ap_target)
    return Z

def compute_sensing_sinr(H_t, Z):
    M_sel, P, Nt = H_t.shape
    sinrs = []
    sigma2 = 0.5
    for p in range(P):
        # 每个AP的贡献叠加
        signal = 0
        for m in range(M_sel):
            h_t = H_t[m, p, :]
            z_m = Z[m, p, :]
            signal += np.abs(np.sum(z_m * np.conj(h_t)))**2
        
        # 干扰: 来自其他目标的波束
        interference = 0
        for q in range(P):
            if q != p:
                for m in range(M_sel):
                    h_t = H_t[m, p, :]
                    z_m = Z[m, q, :]
                    interference += np.abs(np.sum(z_m * np.conj(h_t)))**2
        
        noise = sigma2 * np.sum(np.abs(Z)**2)
        sinrs.append(10 * np.log10(signal / (interference + noise + 1e-10) + 1e-10))
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

print("=== 简单感知波束测试 ===\n")

# 测试不同功率
for P_comm, P_sens in [(18, 12), (15, 15), (12, 18)]:
    sinrs_all = []
    for _ in range(50):
        H_u, H_t = generate_channel()
        
        signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
        total_signal = signal_power.sum(axis=1)
        selected = np.argsort(-total_signal)[:5]
        ap_mask = np.zeros(M, dtype=bool)
        ap_mask[selected] = True
        
        H_u_sel = H_u[ap_mask, :, :]
        H_t_sel = H_t[ap_mask, :, :]
        
        W = mmse_comm_beam(H_u_sel, P_comm)
        Z = sensing_beam_simple(H_t_sel, P_sens)
        
        sinrs = compute_sensing_sinr(H_t_sel, Z)
        sinrs_all.extend(sinrs)
    
    print(f"功率: 通信{P_comm}W / 感知{P_sens}W")
    print(f"  感知SINR: 最小{np.min(sinrs_all):.2f}dB, 平均{np.mean(sinrs_all):.2f}dB, 最大{np.max(sinrs_all):.2f}dB")
    print(f"  满足≥-5dB: {sum(1 for s in sinrs_all if s >= -5)}/{len(sinrs_all)}")
    print()

# 尝试单目标感知(简化问题)
print("=== 单目标感知测试 ===")
for target_power in [5, 10, 15]:
    sinrs_all = []
    for _ in range(50):
        H_u, H_t = generate_channel()
        
        signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
        total_signal = signal_power.sum(axis=1)
        selected = np.argsort(-total_signal)[:5]
        ap_mask = np.zeros(M, dtype=bool)
        ap_mask[selected] = True
        
        H_u_sel = H_u[ap_mask, :, :]
        H_t_sel = H_t[ap_mask, :, :]
        
        W = mmse_comm_beam(H_u_sel, 30 - target_power)
        
        # 只对第一个目标感知
        Z = np.zeros((5, P, Nt), dtype=complex)
        for m in range(5):
            h = H_t_sel[m, 0, :]
            norm = np.sqrt(np.sum(np.abs(h)**2))
            if norm > 0:
                Z[m, 0, :] = np.conj(h) / norm * np.sqrt(target_power / 5)
        
        # 只计算第一个目标的SINR
        p = 0
        h_t = H_t_sel[:, p, :]
        signal = 0
        for m in range(5):
            signal += np.abs(np.sum(Z[m, p, :] * np.conj(h_t[m, :])))**2
        
        noise = 0.5 * np.sum(np.abs(Z)**2)
        sinr = 10 * np.log10(signal / (noise + 1e-10) + 1e-10)
        sinrs_all.append(sinr)
    
    print(f"单目标功率{target_power}W: SINR平均{np.mean(sinrs_all):.2f}dB, 最小{np.min(sinrs_all):.2f}dB")
