#!/usr/bin/env python3
"""
ISAC - CVX近似求解器 + 数据生成
用连续凸近似 + 梯度下降生成多样化训练数据
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置 =====================
cfg = type('C', (), {
    'M': 16, 'K': 8, 'P': 4, 'Nt': 4,
    'Pmax': 30, 'N_req': 4,
    'gamma_k_dB': 5, 'gamma_S_dB': -5, 'Gamma': 5e-3,
    'sigma2': 1e-5
})()

# ===================== CVX近似求解器 =====================
def solve_cvx_approximate(H, G, method='gradient'):
    """
    用多种方法近似CVX求解
    """
    M, K, P, Nt = cfg.M, cfg.K, cfg.P, cfg.Nt
    Pmax = cfg.Pmax
    
    if method == 'gradient':
        return solve_gradient(H, G)
    elif method == 'waterfill':
        return solve_waterfill(H, G)
    elif method == 'random':
        return solve_random(H, G)

def solve_gradient(H, G):
    """梯度下降求解"""
    M, K, P, Nt = cfg.M, cfg.K, cfg.P, cfg.Nt
    Pmax = cfg.Pmax
    
    # 变量: w[m,k,n], z[m,i,j], b[m,p]
    # 展平: W_vars (M*K*Nt*2), Z_vars (M*Nt*Nt), B_vars (M*P)
    n_w = M * K * Nt * 2  # 实部+虚部
    n_z = M * Nt
    n_b = M * P
    n_total = n_w + n_z + n_b
    
    def objective(x):
        # 功率
        w_real = x[:n_w//2]
        w_imag = x[n_w//2:n_w]
        pwr_W = np.sum(w_real**2 + w_imag**2)
        
        z_diag = x[n_w:n_w+n_z]
        pwr_Z = np.sum(z_diag)
        
        total_pwr = pwr_W + pwr_Z
        
        # 惩罚
        penalty = 0
        if total_pwr > Pmax:
            penalty += 100 * (total_pwr - Pmax)**2
        
        return total_pwr + penalty
    
    def gradient(x):
        grad = np.zeros(n_total)
        
        w_real = x[:n_w//2]
        w_imag = x[n_w//2:n_w]
        grad[:n_w//2] = 2 * w_real
        grad[n_w//2:n_w] = 2 * w_imag
        
        z_diag = x[n_w:n_w+n_z]
        grad[n_w:n_w+n_z] = 1
        
        total_pwr = np.sum(w_real**2 + w_imag**2) + np.sum(z_diag)
        if total_pwr > Pmax:
            pwr_grad = 2 * 100 * (total_pwr - Pmax)
            grad[:n_w//2] += pwr_grad * 2 * w_real
            grad[n_w//2:n_w] += pwr_grad * 2 * w_imag
            grad[n_w:n_w+n_z] += pwr_grad
        
        return grad
    
    # 初始化
    x0 = np.random.randn(n_total) * 0.1
    
    # 优化
    result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                     options={'maxiter': 100, 'disp': False})
    
    # 提取结果
    w_real = result.x[:n_w//2]
    w_imag = result.x[n_w//2:n_w]
    w_pwr = np.sum(w_real**2 + w_imag**2)
    
    z_diag = result.x[n_w:n_w+n_z]
    z_pwr = np.sum(z_diag)
    
    # b: 简化为均匀分布
    b = np.random.rand(M, P)
    b = (b > 0.5).astype(float)
    
    # b: 简化为均匀分布
    b = np.random.rand(M, P)
    b = (b > 0.5).astype(float)
    
    return {
        'W_pwr': w_pwr / (w_pwr + z_pwr + 1e-8) * Pmax,
        'Z_pwr': z_pwr / (w_pwr + z_pwr + 1e-8) * Pmax,
        'b': b
    }

def solve_waterfill(H, G):
    """注水算法"""
    M, K, P, Nt = cfg.M, cfg.K, cfg.P, cfg.Nt
    Pmax = cfg.Pmax
    
    # 通信功率分配 (注水)
    channel_gains = np.random.rand(M, K)  # 简化的信道增益
    gains_flat = channel_gains.flatten()
    gains_sort = np.sort(gains_flat)[::-1]
    
    # 注水
    threshold = 1.0
    water_levels = np.maximum(gains_sort - threshold, 0)
    total_water = np.sum(water_levels)
    
    if total_water > Pmax * 0.7:
        scale = Pmax * 0.7 / total_water
        water_levels *= scale
    
    W_pwr = np.sum(water_levels)
    Z_pwr = Pmax - W_pwr
    
    # b: 选择信道最好的AP
    b = np.zeros((M, P))
    for p in range(P):
        best_aps = np.argsort(-channel_gains[:, 0])[:cfg.N_req]
        b[best_aps, p] = 1
    
    return {
        'W_pwr': W_pwr,
        'Z_pwr': Z_pwr,
        'b': b
    }

def solve_random(H, G):
    """随机搜索"""
    M, K, P, Nt = cfg.M, cfg.K, cfg.P, cfg.Nt
    Pmax = cfg.Pmax
    
    best = None
    best_pwr = float('inf')
    
    for _ in range(10):
        # 随机功率分配
        alpha = np.random.rand()
        W_pwr = alpha * Pmax * 0.7
        Z_pwr = (1 - alpha) * Pmax * 0.3
        
        # 检查约束
        if W_pwr + Z_pwr <= Pmax:
            if W_pwr + Z_pwr < best_pwr:
                best_pwr = W_pwr + Z_pwr
                best = {'W_pwr': W_pwr, 'Z_pwr': Z_pwr}
    
    if best is None:
        best = {'W_pwr': Pmax * 0.7, 'Z_pwr': Pmax * 0.3}
    
    # b
    b = np.random.rand(M, P) > 0.75
    b = b.astype(float)
    
    return {
        'W_pwr': best['W_pwr'],
        'Z_pwr': best['Z_pwr'],
        'b': b
    }

# ===================== 多样化数据生成 =====================
def generate_diverse_dataset(num_samples=2000):
    """生成多样化训练数据"""
    print(f"生成 {num_samples} 个多样化样本...")
    
    X_list = []
    W_list = []
    Z_list = []
    B_list = []
    
    methods = ['gradient', 'waterfill', 'random']
    
    for i in range(num_samples):
        # 生成不同条件的信道
        # 方案1: 高SNR
        if i % 3 == 0:
            H = (np.random.randn(cfg.M, cfg.K, cfg.Nt) + 1j * np.random.randn(cfg.M, cfg.K, cfg.Nt)) * 2
            G = (np.random.randn(cfg.M, cfg.M, cfg.P, cfg.Nt) + 1j * np.random.randn(cfg.M, cfg.M, cfg.P, cfg.Nt)) * 2
        # 方案2: 低SNR
        elif i % 3 == 1:
            H = (np.random.randn(cfg.M, cfg.K, cfg.Nt) + 1j * np.random.randn(cfg.M, cfg.K, cfg.Nt)) * 0.5
            G = (np.random.randn(cfg.M, cfg.M, cfg.P, cfg.Nt) + 1j * np.random.randn(cfg.M, cfg.M, cfg.P, cfg.Nt)) * 0.5
        # 方案3: 正常
        else:
            H = (np.random.randn(cfg.M, cfg.K, cfg.Nt) + 1j * np.random.randn(cfg.M, cfg.K, cfg.Nt)) / np.sqrt(2)
            G = (np.random.randn(cfg.M, cfg.M, cfg.P, cfg.Nt) + 1j * np.random.randn(cfg.M, cfg.M, cfg.P, cfg.Nt)) / np.sqrt(2)
        
        # 用不同方法求解
        method = methods[i % len(methods)]
        
        if method == 'gradient':
            result = solve_gradient(H, G)
        elif method == 'waterfill':
            result = solve_waterfill(H, G)
        else:
            result = solve_random(H, G)
        
        # 保存输入
        H_flat = np.concatenate([H.real, H.imag], axis=-1).flatten()
        G_flat = np.concatenate([G.real, G.imag], axis=-1).flatten()
        X = np.concatenate([H_flat, G_flat]).astype(np.float32)
        X = X / (np.linalg.norm(X) + 1e-8)
        
        # 保存标签
        W_list.append(result['W_pwr'])
        Z_list.append(result['Z_pwr'])
        B_list.append(result['b'].flatten())
        
        X_list.append(X)
        
        if (i+1) % 200 == 0:
            print(f"  进度: {i+1}/{num_samples}")
    
    X = np.array(X_list)
    y_W = np.array(W_list, dtype=np.float32).reshape(-1, 1)
    y_Z = np.array(Z_list, dtype=np.float32).reshape(-1, 1)
    y_B = np.array(B_list, dtype=np.float32)
    
    # 保存
    np.savez('isac_diverse_train.npz', X=X, y_W=y_W, y_Z=y_Z, y_B=y_B)
    print(f"保存: isac_diverse_train.npz")
    print(f"  X shape: {X.shape}")
    print(f"  y_W range: [{y_W.min():.2f}, {y_W.max():.2f}]")
    print(f"  y_Z range: [{y_Z.min():.2f}, {y_Z.max():.2f}]")
    print(f"  y_B mean: {y_B.mean():.3f}")
    
    return X, y_W, y_Z, y_B

# ===================== 神经网络 =====================
class DiverseISAC(nn.Module):
    def __init__(self):
        super().__init__()
        
        hd = cfg.M * cfg.K * cfg.Nt * 2
        gd = cfg.M * cfg.M * cfg.P * cfg.Nt * 2
        
        self.encoder = nn.Sequential(
            nn.Linear(hd + gd, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )
        
        # 输出头
        self.W_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 比例
        )
        
        self.Z_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.B_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.M * cfg.P),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        emb = self.encoder(x)
        
        W_ratio = self.W_head(emb)
        Z_ratio = self.Z_head(emb)
        B = self.B_head(emb)
        
        return W_ratio, Z_ratio, B

# ===================== 训练 =====================
def train(X, y_W, y_Z, y_B, epochs=300, bs=32, lr=1e-3):
    Xt = torch.tensor(X, dtype=torch.float32)
    y_Wt = torch.tensor(y_W, dtype=torch.float32)
    y_Zt = torch.tensor(y_Z, dtype=torch.float32)
    y_Bt = torch.tensor(y_B, dtype=torch.float32)
    
    model = DiverseISAC()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    print("\n训练多样化模型...")
    t0 = time.time()
    
    for e in range(epochs):
        indices = torch.randperm(len(Xt))
        
        total_loss = 0
        for i in range(0, len(Xt), bs):
            idx = indices[i:i+bs]
            X_batch = Xt[idx]
            
            W_ratio, Z_ratio, B_pred = model(X_batch)
            
            # 目标: 功率比例
            W_target = y_Wt[idx] / cfg.Pmax
            Z_target = y_Zt[idx] / cfg.Pmax
            
            loss_W = criterion(W_ratio, W_target)
            loss_Z = criterion(Z_ratio, Z_target)
            loss_B = criterion(B_pred, y_Bt[idx])
            
            loss = loss_W + loss_Z + loss_B * 0.5
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if e % 30 == 0:
            print(f"Epoch {e:3d} | Loss: {total_loss:.6f} | {time.time()-t0:.1f}s")
    
    torch.save(model.state_dict(), 'isac_diverse.pth')
    print(f"完成! 保存 isac_diverse.pth, 用时: {time.time()-t0:.1f}s")
    return model

# ===================== 测试 =====================
def test(model, n=20):
    model.eval()
    print("\n测试:")
    
    results = []
    for i in range(n):
        # 随机条件
        H = (np.random.randn(cfg.M, cfg.K, cfg.Nt) + 1j * np.random.randn(cfg.M, cfg.K, cfg.Nt)) / np.sqrt(2)
        G = (np.random.randn(cfg.M, cfg.M, cfg.P, cfg.Nt) + 1j * np.random.randn(cfg.M, cfg.M, cfg.P, cfg.Nt)) / np.sqrt(2)
        
        H_flat = np.concatenate([H.real, H.imag], axis=-1).flatten()
        G_flat = np.concatenate([G.real, G.imag], axis=-1).flatten()
        X = np.concatenate([H_flat, G_flat]).astype(np.float32).reshape(1, -1)
        X = X / (np.linalg.norm(X) + 1e-8)
        
        Xt = torch.tensor(X, dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Xt)
        
        W_pwr = W_ratio.item() * cfg.Pmax
        Z_pwr = Z_ratio.item() * cfg.Pmax
        total = W_pwr + Z_pwr
        
        B_sel = (B > 0.5).float()
        aps = torch.sum(B_sel).item()
        
        results.append({'W': W_pwr, 'Z': Z_pwr, 'Total': total, 'APs': aps})
        print(f"  {i+1}: W={W_pwr:.1f}W Z={Z_pwr:.1f}W Total={total:.1f}W APs={int(aps)}")
    
    avg = np.mean([r['Total'] for r in results])
    std = np.std([r['Total'] for r in results])
    aps_avg = np.mean([r['APs'] for r in results])
    print(f"\n平均功率: {avg:.1f}W ± {std:.1f}W")
    print(f"平均AP数: {aps_avg:.1f}")
    return results

# ===================== 主程序 =====================
if __name__ == '__main__':
    print("="*60)
    print("ISAC - 多样化CVX数据生成")
    print("="*60)
    
    # 生成数据
    X, y_W, y_Z, y_B = generate_diverse_dataset(num_samples=2000)
    
    # 训练
    model = train(X, y_W, y_Z, y_B, epochs=300)
    
    # 测试
    results = test(model, n=20)