"""
使用迫零预编码 (Zero Forcing) 消除用户间干扰
"""

import numpy as np
from scipy.optimize import minimize

M, K, Nt = 16, 4, 4
Pmax = 30

def generate_channel(n_samples):
    """生成真实信道"""
    X_data = []
    for _ in range(n_samples):
        # 简单的大尺度衰落模型
        ap_x = np.linspace(-60, 60, 4)
        ap_y = np.linspace(-60, 60, 4)
        AP_pos = np.array([[x, y] for x in ap_x for y in ap_y])
        
        user_pos = np.random.uniform(-50, 50, (K, 2))
        
        H = np.zeros((M, K, Nt*2), dtype=np.float32)
        for m in range(M):
            for k in range(K):
                d = np.sqrt(np.sum((AP_pos[m] - user_pos[k])**2))
                d = max(d, 5)
                pl = (d / 10) ** -2  # 简化的路径损耗
                
                h = np.sqrt(pl/2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
                H[m, k, :Nt] = np.real(h)
                H[m, k, Nt:] = np.imag(h)
        
        X_data.append(H.flatten())
    
    return np.array(X_data)

def zf_precoding(H, Pmax):
    """迫零预编码"""
    H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]  # (M, K, Nt)
    
    # 将信道堆叠成 M*Nt x K 的矩阵
    H_stack = np.vstack([H_complex[m, :, :] for m in range(M)])  # (M*Nt, K)
    
    # 迫零: W_zf = H^+ (伪逆)
    try:
        H_pinv = np.linalg.pinv(H_stack)  # (K, M*Nt)
    except:
        H_pinv = np.linalg.lstsq(H_stack, np.eye(K), rcond=None)[0]
    
    # 功率分配
    total_power = Pmax * 0.7
    
    # 等功率分配
    w_zf = H_pinv * np.sqrt(total_power / K)
    
    # 重塑为波束矩阵
    w = np.zeros((M, K, Nt), dtype=complex)
    for m in range(M):
        for k in range(K):
            w[m, k, :] = w_zf[k, m*Nt:(m+1)*Nt]
    
    # 归一化
    w = w / (np.sqrt(np.sum(np.abs(w)**2)) + 1e-8) * np.sqrt(Pmax * 0.7)
    
    return w

def compute_sinr(H, w):
    H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
    sinrs = []
    for k in range(K):
        signal = np.abs(np.sum(np.conj(w[:, k, :]) * H_complex[:, k, :])) ** 2
        interference = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H_complex[:, k, :])) ** 2 for j in range(K) if j != k)
        noise = 0.01
        sinrs.append(10 * np.log10(signal / (interference + noise) + 1e-8))
    return np.array(sinrs)

# 测试ZF预编码
print("测试迫零预编码:")
results = []

X = generate_channel(30)

for i in range(30):
    H = X[i].reshape(M, K, Nt*2)
    
    # ZF预编码
    w_zf = zf_precoding(H, Pmax)
    
    power = np.sum(np.abs(w_zf) ** 2)
    sinr_db = compute_sinr(H, w_zf)
    
    results.append({'power': power, 'sinr_min': sinr_db.min(), 'sinr_mean': sinr_db.mean()})
    print(f"  样本{i+1}: 功率={power:.2f}W, SINR_min={sinr_db.min():.2f}dB, SINR_mean={sinr_db.mean():.2f}dB")

print(f"\n平均功率: {np.mean([r['power'] for r in results]):.2f}W")
print(f"平均SINR_min: {np.mean([r['sinr_min'] for r in results]):.2f}dB")
print(f"正SINR比例: {sum(1 for r in results if r['sinr_min'] > 0)}/30")

# 尝试优化功率分配
print("\n尝试功率优化:")

def optimize_with_zf(H, Pmax):
    H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
    H_stack = np.vstack([H_complex[m, :, :] for m in range(M)])
    
    try:
        H_pinv = np.linalg.pinv(H_stack)
    except:
        H_pinv = np.linalg.lstsq(H_stack, np.eye(K), rcond=None)[0]
    
    def objective(p_db):
        power = 10 ** (p_db / 10)
        w_zf = H_pinv * np.sqrt(power / K)
        
        w = np.zeros((M, K, Nt), dtype=complex)
        for m in range(M):
            for k in range(K):
                w[m, k, :] = w_zf[k, m*Nt:(m+1)*Nt]
        w = w / (np.sqrt(np.sum(np.abs(w)**2)) + 1e-8) * np.sqrt(power)
        
        sinr_db = compute_sinr(H, w)
        min_sinr = min(sinr_db)
        
        # 目标: 在满足SINR>=10dB的前提下最小化功率
        if min_sinr < 10:
            return power + 100 * (10 - min_sinr)
        return power
    
    result = minimize(objective, 15, method='BFGS', options={'maxiter': 100})
    return 10 ** (result.x[0] / 10), result.fun

results2 = []
for i in range(10):
    H = X[i].reshape(M, K, Nt*2)
    power, sinr = optimize_with_zf(H, Pmax)
    w_zf = np.linalg.pinv(np.vstack([H[:, :, :Nt] + 1j*H[:, :, Nt:] for m in range(M)])) * np.sqrt(power / K)
    w = np.zeros((M, K, Nt), dtype=complex)
    for m in range(M):
        for k in range(K):
            w[m, k, :] = w_zf[k, m*Nt:(m+1)*Nt]
    w = w / (np.sqrt(np.sum(np.abs(w)**2)) + 1e-8) * np.sqrt(power)
    sinr_db = compute_sinr(H, w)
    results2.append({'power': power, 'sinr_min': sinr_db.min()})
    print(f"  样本{i+1}: 功率={power:.2f}W, SINR_min={sinr_db.min():.2f}dB")

print(f"\n优化后正SINR比例: {sum(1 for r in results2 if r['sinr_min'] > 0)}/10")
