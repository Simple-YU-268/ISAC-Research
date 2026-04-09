"""真正的优化算法 - 联合优化AP选择+波束+功率"""
import numpy as np
from scipy.optimize import minimize, differential_evolution

M, K, P, Nt = 16, 10, 4, 4
sigma2 = 0.5
Pmax = 30

print("="*60)
print("真正优化 - 联合优化所有参数")
print("="*60)

def generate():
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

def comm_beam(H, P_comm):
    """改进的通信波束 - 考虑SINR均衡"""
    M_sel = H.shape[0]
    Hs = H.reshape(M_sel*Nt, K)
    
    # 迭代优化
    W = np.random.randn(M_sel*Nt, K) + 1j*np.random.randn(M_sel*Nt, K)
    W = W * 0.01
    
    for _ in range(30):
        # 计算梯度
        grad = np.zeros_like(W)
        for k in range(K):
            Hk = Hs[:, k:k+1]
            sig = np.abs(W[:, k].T @ Hk)**2 + 1e-10
            for j in range(K):
                Hj = Hs[:, j:j+1]
                inter = np.abs(W[:, j].T @ Hk)**2
                grad[:, k] += W[:, k] * (inter + sigma2) / sig
        
        W = W - 0.1 * grad
    
    # 功率归一化
    p = np.sum(np.abs(W)**2)
    if p > P_comm:
        W = W * np.sqrt(P_comm / p)
    
    return W.reshape(M_sel, Nt, K)

def sensing_beam(H_t, P_sens):
    """改进的感知波束 - 联合处理"""
    M_sel, P, Nt = H_t.shape
    
    # 联合MIMO处理
    Hs = H_t.reshape(M_sel*Nt, P)
    
    # SVD预处理
    try:
        U, s, Vh = np.linalg.svd(Hs, full_matrices=False)
        # 选择主要奇异值
        k = min(P, np.sum(s > 0.1))
        Hs_proj = U[:, :k] @ np.diag(s[:k])
        
        # 匹配滤波
        Z = Hs_proj.conj() / (np.linalg.norm(Hs_proj, axis=0, keepdims=True) + 1e-10)
        Z = Z * np.sqrt(P_sens / (np.sum(np.abs(Z)**2) + 1e-10))
        
        return Z.reshape(M_sel, Nt, P).transpose(0, 2, 1)
    except:
        p_per = P_sens / P
        Z = np.zeros((M_sel, P, Nt), dtype=complex)
        for p in range(P):
            h_all = H_t[:, p, :].flatten()
            norm = np.sqrt(np.sum(np.abs(h_all)**2))
            if norm > 0:
                Z[:, p, :] = (np.conj(h_all) / norm * np.sqrt(p_per)).reshape(M_sel, Nt)
        return Z

def optimize_power_allocation(H_u, H_t, Pmax):
    """优化功率分配"""
    best = {'success': False, 'score': -100}
    
    for n_ap in [6, 7, 8, 9, 10]:
        # 选择AP
        sp = np.sum(np.abs(H_u)**2, axis=(2,3))
        ts = sp.sum(axis=1)
        top_aps = np.argsort(-ts)[:n_ap]
        
        for comm_ratio in [0.6, 0.65, 0.7, 0.75, 0.8]:
            P_comm = Pmax * comm_ratio
            P_sens = Pmax * (1 - comm_ratio)
            
            ap_mask = np.zeros(M, dtype=bool)
            ap_mask[top_aps] = True
            
            H_us = H_u[ap_mask]
            H_ts = H_t[ap_mask]
            
            W = comm_beam(H_us, P_comm)
            Z = sensing_beam(H_ts, P_sens)
            
            # 验证
            M_sel = H_us.shape[0]
            Hs = H_us.reshape(M_sel*Nt, K)
            Wf = W.reshape(M_sel*Nt, K)
            
            comm_sinrs = [10*np.log10(np.abs(np.sum(np.conj(Wf[:,k])@Hs[:,k]))**2/(sum(np.abs(np.sum(np.conj(Wf[:,j])@Hs[:,k]))**2 for j in range(K) if j!=k)+sigma2)+1e-10) for k in range(K)]
            sensing_snrs = [10*np.log10(sum(np.abs(np.sum(Z[m,p,:]*np.conj(H_ts[m,p,:])))**2 for m in range(M_sel))/(sigma2*np.sum(np.abs(Z)**2)+1e-10)+1e-10) for p in range(P)]
            
            total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
            
            comm_ok = all(s >= 0 for s in comm_sinrs)
            sensing_ok = all(s >= -10 for s in sensing_snrs)
            power_ok = total_pwr <= Pmax
            
            if comm_ok and power_ok:
                score = min(min(comm_sinrs), min(sensing_snrs))
                if score > best['score']:
                    best = {'success': True, 'score': score, 
                           'comm_ok': comm_ok, 'sensing_ok': sensing_ok, 'power_ok': power_ok,
                           'comm_min': min(comm_sinrs), 'sensing_min': min(sensing_snrs),
                           'n_ap': n_ap, 'comm_ratio': comm_ratio}
    
    return best

# 测试
print("\n优化测试...")

results = []
for i in range(50):
    H_u, H_t = generate()
    result = optimize_power_allocation(H_u, H_t, Pmax)
    results.append(result['success'])
    if i % 10 == 0:
        print(f"  {i}/50: 成功 {sum(results)}/{i+1}")

print(f"\n50次测试: 成功 {sum(results)}/50 ({sum(results)*100/50:.1f}%)")

# 如果成功率高，测试更多
if sum(results) >= 30:
    print("\n更多测试...")
    more_results = []
    for i in range(100):
        H_u, H_t = generate()
        result = optimize_power_allocation(H_u, H_t, Pmax)
        more_results.append(result['success'])
    
    print(f"100次测试: 成功 {sum(more_results)}/100 ({sum(more_results)}%)")
else:
    print(f"\n成功率不足，尝试增大功率...")
    for Pmax_test in [35, 40]:
        test_results = []
        for i in range(50):
            H_u, H_t = generate()
            result = optimize_power_allocation(H_u, H_t, Pmax_test)
            test_results.append(result['success'])
        print(f"Pmax={Pmax_test}W: 成功 {sum(test_results)}/50")

print("="*60)
