"""完整优化方案 - 使用更高效的通信和感知波束"""
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

M, K, P, Nt = 16, 10, 4, 4
sigma2 = 0.5

print("="*60)
print("完整优化 - 寻找30W下的最优解")
print("="*60)

def generate_channel():
    ap = np.array([[x,y] for x in np.linspace(-60,60,4) for y in np.linspace(-60,60,4)])
    user_pos = np.random.uniform(-20,20,(K,2))
    target_pos = np.random.uniform(-15,15,(P,2))
    
    H_u = np.zeros((M,K,Nt), dtype=complex)
    H_t = np.zeros((M,P,Nt), dtype=complex)
    
    for m in range(M):
        for k in range(K):
            d = max(np.sqrt(np.sum((ap[m]-user_pos[k])**2)), 3)
            pl = (d/10)**-2.5
            H_u[m,k,:] = np.sqrt(pl/2)*(np.random.randn(Nt)+1j*np.random.randn(Nt))
        for p in range(P):
            d = max(np.sqrt(np.sum((ap[m]-target_pos[p])**2)), 3)
            pl = (d/10)**-2.5
            H_t[m,p,:] = np.sqrt(pl/2)*(np.random.randn(Nt)+1j*np.random.randn(Nt))
    
    return H_u, H_t

def zero_forcing(H, Pmax):
    """迫零波束 - 更高效率"""
    M_sel = H.shape[0]
    Hs = H.reshape(M_sel*Nt, K)
    try:
        W = np.linalg.pinv(Hs) * np.sqrt(Pmax)
        p = np.sum(np.abs(W)**2)
        return (W * np.sqrt(Pmax/p)).reshape(M_sel, Nt, K)
    except:
        return None

def mmse_beam(H, Pmax):
    """MMSE波束"""
    M_sel = H.shape[0]
    Hs = H.reshape(M_sel*Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel*Nt)
    W = np.linalg.inv(HH) @ Hs
    p = np.sum(np.abs(W)**2)
    return W.reshape(M_sel,Nt,K) * np.sqrt(Pmax/p)

def joint_sensing(H_t, P_sens):
    """联合感知 - 所有AP联合处理"""
    M_sel, P, Nt = H_t.shape
    Hs = H_t.reshape(M_sel*Nt, P)
    try:
        # 迫零检测
        Z = np.linalg.pinv(Hs) * np.sqrt(P_sens/P)
        return Z.reshape(M_sel, Nt, P).transpose(0,2,1)
    except:
        # 备用
        p_per = P_sens/P
        Z = np.zeros((M_sel,P,Nt), dtype=complex)
        for p in range(P):
            h_all = H_t[:,p,:].flatten()
            norm = np.sqrt(np.sum(np.abs(h_all)**2))
            if norm > 0:
                beam = np.conj(h_all)/norm * np.sqrt(p_per)
                for m in range(M_sel):
                    Z[m,p,:] = beam[m*Nt:(m+1)*Nt]
        return Z

def check_constraints(H_u, H_t, ap_mask, Pmax, comm_pct=0.75):
    """检查所有约束"""
    H_u_sel = H_u[ap_mask,:,:]
    H_t_sel = H_t[ap_mask,:,:]
    n_ap = np.sum(ap_mask)
    
    P_comm = Pmax * comm_pct
    P_sens = Pmax * (1 - comm_pct)
    
    # 通信波束
    W = mmse_beam(H_u_sel, P_comm)
    
    # 感知波束
    Z = joint_sensing(H_t_sel, P_sens)
    
    # 通信SINR
    M_sel = H_u_sel.shape[0]
    Hs = H_u_sel.reshape(M_sel*Nt, K)
    Wf = W.reshape(M_sel*Nt, K)
    comm_ok = True
    for k in range(K):
        sig = np.abs(np.sum(np.conj(Wf[:,k]) @ Hs[:,k]))**2
        inter = sum(np.abs(np.sum(np.conj(Wf[:,j]) @ Hs[:,k]))**2 for j in range(K) if j!=k)
        if 10*np.log10(sig/(inter+sigma2)+1e-10) < 0:
            comm_ok = False
            break
    
    # 感知SNR
    sensing_ok = True
    for p in range(P):
        signal = sum(np.abs(np.sum(Z[m,p,:]*np.conj(H_t_sel[m,p,:])))**2 for m in range(M_sel))
        noise = sigma2 * np.sum(np.abs(Z)**2)
        if 10*np.log10(signal/(noise+1e-10)+1e-10) < -5:
            sensing_ok = False
            break
    
    total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    power_ok = total_pwr <= Pmax
    
    return {
        'comm_ok': comm_ok,
        'sensing_ok': sensing_ok,
        'power_ok': power_ok,
        'all_ok': comm_ok and sensing_ok and power_ok,
        'comm_min': min([10*np.log10(np.abs(np.sum(np.conj(Wf[:,k]) @ Hs[:,k]))**2/(sum(np.abs(np.sum(np.conj(Wf[:,j]) @ Hs[:,k]))**2 for j in range(K) if j!=k)+sigma2)+1e-10) for k in range(K)]),
        'sensing_min': min([10*np.log10(sum(np.abs(np.sum(Z[m,p,:]*np.conj(H_t_sel[m,p,:])))**2 for m in range(M_sel))/(sigma2*np.sum(np.abs(Z)**2)+1e-10)+1e-10) for p in range(P)])
    }

# 测试不同配置
print("\n配置搜索...")

configs = []
for Pmax in [30, 35, 40]:
    for n_ap in [6, 8, 10]:
        for comm_pct in [0.7, 0.75, 0.8]:
            configs.append((Pmax, n_ap, comm_pct))

results_by_config = {}

for Pmax, n_ap, comm_pct in configs:
    success = 0
    for _ in range(30):
        H_u, H_t = generate_channel()
        
        # 选择AP
        sp = np.sum(np.abs(H_u)**2, axis=(2,3))
        ts = sp.sum(axis=1)
        selected = np.argsort(-ts)[:n_ap]
        ap_mask = np.zeros(M, dtype=bool)
        ap_mask[selected] = True
        
        result = check_constraints(H_u, H_t, ap_mask, Pmax, comm_pct)
        if result['all_ok']:
            success += 1
    
    if success > 15:
        results_by_config[(Pmax, n_ap, comm_pct)] = success
        print(f"Pmax={Pmax}W, {n_ap}AP, 通信{comm_pct*100:.0f}%: {success}/30")

# 最佳配置深度测试
if results_by_config:
    best_config = max(results_by_config.items(), key=lambda x: x[1])[0]
    Pmax, n_ap, comm_pct = best_config
    
    print(f"\n最佳配置: Pmax={Pmax}W, {n_ap}AP, 通信{comm_pct*100:.0f}%")
    print("深度测试...")
    
    all_ok = 0
    comm_ok = 0
    sensing_ok = 0
    power_ok = 0
    
    for _ in range(200):
        H_u, H_t = generate_channel()
        sp = np.sum(np.abs(H_u)**2, axis=(2,3))
        ts = sp.sum(axis=1)
        selected = np.argsort(-ts)[:n_ap]
        ap_mask = np.zeros(M, dtype=bool)
        ap_mask[selected] = True
        
        result = check_constraints(H_u, H_t, ap_mask, Pmax, comm_pct)
        
        if result['comm_ok']: comm_ok += 1
        if result['sensing_ok']: sensing_ok += 1
        if result['power_ok']: power_ok += 1
        if result['all_ok']: all_ok += 1
    
    print(f"\n200次测试结果:")
    print(f"  通信≥0dB: {comm_ok}/200")
    print(f"  感知≥-5dB: {sensing_ok}/200")
    print(f"  功率≤{Pmax}W: {power_ok}/200")
    print(f"  完全满足: {all_ok}/200 ({all_ok*100/200:.1f}%)")
else:
    print("\n30W下无法满足约束，需要更大功率或降低阈值")
    
print("\n" + "="*60)
