"""
ISAC算法分析 - 为什么无法达到10用户 SINR≥0dB
"""

import numpy as np
from scipy.optimize import minimize

M, K, Nt = 16, 10, 4   # 10个用户
Pmax = 30

print("=== ISAC算法问题分析 ===")
print(f"场景: {M}AP, {K}用户, {Nt}天线, 总功率{Pmax}W")
print(f"目标: 每用户SINR≥0dB\n")

# 问题1: 自由度分析
print("【问题1: 自由度不匹配】")
total_dof = M * Nt  # 16*4 = 64 发射天线
user_dof = K       # 10 用户
print(f"  发射天线自由度: {total_dof}")
print(f"  用户数: {K}")
print(f"  理论上: 64 > 10，应该可以")
print(f"  但: 每个用户需要独立数据流，实际是{K}个独立信道")

# 问题2: 功率分配
print("\n【问题2: 功率分配】")
p_per_user = Pmax / K
print(f"  总功率: {Pmax}W")
print(f"  平均每用户: {p_per_user:.2f}W")
print(f"  在噪声 0.01W 下: {10*np.log10(p_per_user/0.01):.2f}dB")
print(f"  这是理想情况，实际还要考虑干扰")

# 问题3: 信道条件
print("\n【问题3: 信道条件测试】")

def gen_channel():
    ap = np.array([[x, y] for x in np.linspace(-60,60,4) for y in np.linspace(-60,60,4)])
    u = np.random.uniform(-50,50,(K,2))
    H = np.zeros((M,K,Nt), dtype=complex)
    for m in range(M):
        for k in range(K):
            d = max(np.sqrt(np.sum((ap[m]-u[k])**2)), 5)
            pl = (d/10)**-2.5
            H[m,k,:] = np.sqrt(pl)*(np.random.randn(Nt)+1j*np.random.randn(Nt))/np.sqrt(2)
    return H

def compute_sinr(H, W):
    Hs = H.reshape(M*Nt, K)
    sinrs = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W[:,k]) @ Hs[:,k]))**2
        inter = sum(np.abs(np.sum(np.conj(W[:,j]) @ Hs[:,k]))**2 for j in range(K) if j!=k)
        sinrs.append(10*np.log10(sig/(inter+0.01)+1e-10))
    return np.array(sinrs)

# 理论最优: 脏编码 (TDM/FDM)
print("\n【理论最优: 时分复用】")
results = []
for _ in range(20):
    H = gen_channel()
    Hs = H.reshape(M*Nt, K)
    
    # 每次只服务1个用户
    best_sinrs = []
    for k in range(K):
        h = Hs[:, k]
        w = h / (np.linalg.norm(h) + 1e-3) * np.sqrt(Pmax)
        sig = np.abs(np.sum(np.conj(w) @ h))**2
        best_sinrs.append(10*np.log10(sig/0.01 + 1e-10))
    
    results.append(np.max(best_sinrs))

print(f"  单用户最优SINR: 平均{np.mean(results):.2f}dB, 最差{np.min(results):.2f}dB")

# 多用户MMSE
print("\n【多用户MMSE】")
results_mmse = []
for _ in range(20):
    H = gen_channel()
    Hs = H.reshape(M*Nt, K)
    
    HH = Hs @ Hs.T.conj() + 0.5*np.eye(M*Nt)
    W = np.linalg.inv(HH) @ Hs
    p = np.sum(np.abs(W)**2)
    W = W * np.sqrt(Pmax * 0.9 / p)
    
    sinrs = compute_sinr(H, W)
    results_mmse.append({'min': np.min(sinrs), 'mean': np.mean(sinrs), 'ok': sum(sinrs>=0)})

print(f"  SINR_min: 平均{np.mean([r['min'] for r in results_mmse]):.2f}dB")
print(f"  SINR_mean: 平均{np.mean([r['mean'] for r in results_mmse]):.2f}dB")
print(f"  达到0dB用户: 平均{np.mean([r['ok'] for r in results_mmse]):.1f}/{K}")

print("\n" + "="*50)
print("【根本问题】")
print("1. 10用户共享16AP功率，干扰严重")
print("2. 随机信道导致部分用户信道条件差")
print("3. 30W功率不足以支撑10用户同时通信")
print("\n【解决方案】")
print("- 降低用户数或增加功率")
print("- 使用更先进的预编码(脏编码/非线性)")
print("- 引入用户调度")
print("="*50)
