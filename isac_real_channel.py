"""
ISAC with Realistic Channel Model
真实信道模型:
1. 路径损耗 (自由空间 + 实际衰减)
2. 阴影衰落 (对数正态分布)
3. 小尺度衰落 (瑞利/莱斯)
4. 角度扩展 (基于几何的信道模型)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize

# 系统参数
M = 16  # AP数量
K = 4   # 用户数
P = 4   # 感知目标数
Nt = 4  # 天线数
Pmax = 30
N_req = 4

# 真实信道参数
fc = 3e9  # 载频 3GHz
c = 3e8   # 光速
d_bs = 50  # AP间距离 (m)
d_user_max = 100  # 用户最大距离 (m)

def generate_realistic_channel(n_samples, seed=None):
    """生成真实信道模型"""
    if seed is not None:
        np.random.seed(seed)
    
    X_data = []
    channel_data = []  # 存储真实信道
    
    for _ in range(n_samples):
        # 1. AP布局 (方形覆盖区域)
        ap_x = np.linspace(-(M//4)*d_bs, (M//4)*d_bs, 4)
        ap_y = np.linspace(-(M//4)*d_bs, (M//4)*d_bs, 4)
        AP_pos = np.array([[x, y] for x in ap_x for y in ap_y])  # 16个AP
        
        # 2. 用户位置 (随机分布在覆盖区域内)
        user_pos = np.random.uniform(-80, 80, (K, 2))
        
        # 3. 感知目标位置
        target_pos = np.random.uniform(-50, 50, (P, 2))
        
        # 4. 计算信道
        H_real = np.zeros((M, K, Nt*2), dtype=np.float32)
        
        for m in range(M):
            for k in range(K):
                # 距离
                d = np.sqrt(np.sum((AP_pos[m] - user_pos[k])**2))
                d = max(d, 1)  # 避免除零
                
                # 路径损耗 (自由空间 + 实际模型)
                # PL = 20*log10(d) + 20*log10(fc) - 147.55
                pl_db = 20 * np.log10(d) + 20 * np.log10(fc/1e9) - 147.55
                pl_linear = 10 ** (-pl_db / 10)
                
                # 阴影衰落 (对数正态, sigma=8dB)
                shadow_db = np.random.normal(0, 8)
                shadow_linear = 10 ** (-shadow_db / 10)
                
                # 小尺度衰落 (瑞利信道)
                # 生成多个路径的叠加 (3-5条)
                n_paths = np.random.randint(3, 6)
                alpha = np.sqrt(pl_linear * shadow_linear / n_paths)
                
                h = np.zeros(Nt, dtype=complex)
                for p in range(n_paths):
                    # 随机延迟扩展
                    tau = np.random.uniform(0, 1e-6)
                    # 多普勒扩展 (低速场景)
                    fd = 10  # 最大多普勒频移
                    # 相位
                    phi = np.random.uniform(0, 2*np.pi)
                    
                    # 简化的频率选择信道
                    h += alpha * np.exp(1j * phi) * np.exp(-1j * 2 * np.pi * fc * tau)
                
                # 转换为实数表示
                H_real[m, k, :Nt] = np.real(h) / (np.linalg.norm(np.real(h)) + 1e-8)
                H_real[m, k, Nt:] = np.imag(h) / (np.linalg.norm(np.imag(h)) + 1e-8)
        
        # 归一化
        H_real = H_real / (np.linalg.norm(H_real, axis=(-1,-2), keepdims=True) + 1e-8)
        
        X_data.append(H_real.flatten())
        channel_data.append({
            'AP_pos': AP_pos,
            'user_pos': user_pos,
            'target_pos': target_pos,
            'H': H_real
        })
    
    return np.array(X_data), channel_data

def compute_sinr_real(H, w):
    """真实SINR计算"""
    H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
    sinrs = []
    
    for k in range(K):
        signal = np.abs(np.sum(np.conj(w[:, k, :]) * H_complex[:, k, :])) ** 2
        interference = 0
        for j in range(K):
            if j != k:
                interference += np.abs(np.sum(np.conj(w[:, j, :]) * H_complex[:, k, :])) ** 2
        
        sinr = signal / (interference + 0.001)
        sinrs.append(10 * np.log10(sinr + 1e-8))
    
    return np.array(sinrs)

# 测试真实信道模型
print("生成真实信道数据...")
X_real, channels = generate_realistic_channel(100, seed=42)

print(f"生成完成: {len(X_real)} 样本")
print(f"信道形状: {channels[0]['H'].shape}")

# 测试优化器在真实信道下的性能
print("\n测试优化器在真实信道下的SINR:")

def optimize_comm(H, Pmax):
    H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
    
    def objective(p_vec):
        p = p_vec.reshape(M, K)
        power = np.sum(p)
        if power > Pmax * 0.8:
            return 1e6
        
        # 波束
        w = np.sqrt(p[:, :, None]) * (np.random.randn(M, K, Nt) + 1j * np.random.randn(M, K, Nt))
        w = w / np.sqrt(M * Nt)
        
        sinrs = []
        for k in range(K):
            signal = np.abs(np.sum(np.conj(w[:, k, :]) * H_complex[:, k, :])) ** 2
            interference = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H_complex[:, k, :])) ** 2 for j in range(K) if j != k)
            sinrs.append(signal / (interference + 0.01))
        
        min_sinr = min(sinrs)
        if min_sinr < 5:
            return power + 20 * (5 - min_sinr)
        return power
    
    p0 = np.random.rand(M, K) * 1
    result = minimize(objective, p0.flatten(), method='SLSQP', options={'maxiter': 300})
    p_opt = result.x.reshape(M, K)
    
    # 归一化功率
    if np.sum(p_opt) > 0:
        p_opt = p_opt * Pmax * 0.7 / np.sum(p_opt)
    
    # 构建波束
    w = np.sqrt(p_opt[:, :, None]) * (np.random.randn(M, K, Nt) + 1j * np.random.randn(M, K, Nt))
    w = w / np.sqrt(M * Nt)
    
    return np.sum(p_opt), compute_sinr_real(H, w)

success = 0
for i in range(20):
    H = channels[i]['H']
    power, sinr_db = optimize_comm(H, Pmax)
    
    print(f"  样本{i+1}: 功率={power:.2f}W, SINR_min={sinr_db.min():.2f}dB, SINR_mean={sinr_db.mean():.2f}dB")
    if sinr_db.min() > 0:
        success += 1

print(f"\n正SINR比例: {success}/20")
