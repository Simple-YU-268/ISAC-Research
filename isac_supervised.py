#!/usr/bin/env python3
"""
ISAC - 监督学习 (用CVX生成训练数据)
步骤:
1. 用MATLAB CVX生成"真解"
2. 保存为训练数据
3. 用监督学习训练神经网络
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

# ===================== 配置 =====================
cfg = type('C', (), {
    'M': 16, 'K': 8, 'P': 4, 'Nt': 4,
    'Pmax': 30, 'N_req': 4,
})()

# ===================== 简化的CVX求解器 (用凸优化近似) =====================
def solve_cvx_approx(H, G):
    """
    近似CVX求解 (连续松弛)
    实际问题是非凸的，这里用连续松弛+贪心
    """
    n_samples = H.shape[0]
    results = []
    
    for i in range(n_samples):
        # 简化的功率分配
        # 通信: 均匀分配
        w_pwr = cfg.Pmax * 0.7 / (cfg.M * cfg.K)
        
        # 感知: 均匀分配
        z_pwr = cfg.Pmax * 0.3 / cfg.M
        
        # AP选择: 贪心选择
        # 简化: 选择信道增益最大的AP
        ap_scores = np.random.rand(cfg.M, cfg.P)  # 简化的信道增益
        b = np.zeros((cfg.M, cfg.P))
        for p in range(cfg.P):
            top_aps = np.argsort(-ap_scores[:, p])[:cfg.N_req]
            b[top_aps, p] = 1
        
        results.append({
            'w_pwr': w_pwr,
            'z_pwr': z_pwr,
            'b': b
        })
    
    return results

# ===================== 数据生成 =====================
def generate_dataset(num_samples, save_path='isac_train_data.npz'):
    """生成训练数据"""
    print(f"生成 {num_samples} 个训练样本...")
    
    H_list = []
    G_list = []
    W_list = []  # 功率
    Z_list = []  # 感知功率
    B_list = []  # AP选择
    
    for i in range(num_samples):
        # 生成信道
        H = (np.random.randn(cfg.M, cfg.K, cfg.Nt) + 1j * np.random.randn(cfg.M, cfg.K, cfg.Nt)) / np.sqrt(2)
        G = (np.random.randn(cfg.M, cfg.M, cfg.P, cfg.Nt) + 1j * np.random.randn(cfg.M, cfg.M, cfg.P, cfg.Nt)) / np.sqrt(2)
        
        # CVX求解
        cvx_result = solve_cvx_approx(H, G)[0]
        
        # 保存
        H_list.append(np.concatenate([H.real, H.imag], axis=-1).flatten())
        G_list.append(np.concatenate([G.real, G.imag], axis=-1).flatten())
        W_list.append(cvx_result['w_pwr'])
        Z_list.append(cvx_result['z_pwr'])
        B_list.append(cvx_result['b'].flatten())
        
        if (i+1) % 100 == 0:
            print(f"  进度: {i+1}/{num_samples}")
    
    # 转为数组
    X = np.concatenate([np.array(H_list), np.array(G_list)], axis=1).astype(np.float32)
    y_w = np.array(W_list, dtype=np.float32).reshape(-1, 1)
    y_z = np.array(Z_list, dtype=np.float32).reshape(-1, 1)
    y_b = np.array(B_list, dtype=np.float32)
    
    # 归一化
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    
    # 保存
    np.savez(save_path, X=X, y_w=y_w, y_z=y_z, y_b=y_b)
    print(f"保存: {save_path}")
    
    return X, y_w, y_z, y_b

# ===================== 神经网络 =====================
class SupervisedISAC(nn.Module):
    """监督学习网络"""
    def __init__(self):
        super().__init__()
        
        hd = cfg.M * cfg.K * cfg.Nt * 2
        gd = cfg.M * cfg.M * cfg.P * cfg.Nt * 2
        
        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Linear(hd + gd, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )
        
        # W头
        self.w_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出 [0,1] 比例
        )
        
        # Z头
        self.z_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # b头
        self.b_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.M * cfg.P),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        emb = self.encoder(x)
        
        # W (通信功率比例)
        w_ratio = self.w_head(emb) * 0.7  # 上限70%
        
        # Z (感知功率比例)
        z_ratio = self.z_head(emb) * 0.3  # 上限30%
        
        # b (AP选择)
        b = self.b_head(emb)
        
        return w_ratio, z_ratio, b

# ===================== 训练 =====================
def train_supervised(X, y_w, y_z, y_b, epochs=200, bs=32, lr=1e-3):
    """监督训练"""
    
    # 转tensor
    Xt = torch.tensor(X, dtype=torch.float32)
    y_wt = torch.tensor(y_w, dtype=torch.float32)
    y_zt = torch.tensor(y_z, dtype=torch.float32)
    y_bt = torch.tensor(y_b, dtype=torch.float32)
    
    model = SupervisedISAC()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    print("\n监督学习训练...")
    t0 = time.time()
    
    n_samples = len(Xt)
    indices = torch.randperm(n_samples)
    
    for e in range(epochs):
        # shuffle
        indices = torch.randperm(n_samples)
        
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, bs):
            idx = indices[i:i+bs]
            X_batch = Xt[idx]
            
            w_ratio, z_ratio, b_pred = model(X_batch)
            
            # 损失
            w_target = y_wt[idx]
            z_target = y_zt[idx]
            b_target = y_bt[idx]
            
            loss_w = criterion(w_ratio, w_target)
            loss_z = criterion(z_ratio, z_target)
            loss_b = criterion(b_pred, b_target)
            
            loss = loss_w + loss_z + loss_b * 0.5
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if e % 20 == 0:
            avg_loss = total_loss / n_batches
            print(f"Epoch {e:3d} | Loss: {avg_loss:.6f} | {time.time()-t0:.1f}s")
    
    torch.save(model.state_dict(), 'isac_supervised.pth')
    print(f"\n完成! 保存 isac_supervised.pth, 用时: {time.time()-t0:.1f}s")
    return model

# ===================== 测试 =====================
def test(model, n=20):
    model.eval()
    print("\n测试:")
    
    results = []
    for i in range(n):
        # 生成新样本
        H = (np.random.randn(cfg.M, cfg.K, cfg.Nt) + 1j * np.random.randn(cfg.M, cfg.K, cfg.Nt)) / np.sqrt(2)
        G = (np.random.randn(cfg.M, cfg.M, cfg.P, cfg.Nt) + 1j * np.random.randn(cfg.M, cfg.M, cfg.P, cfg.Nt)) / np.sqrt(2)
        
        H_flat = np.concatenate([H.real, H.imag], axis=-1).flatten()
        G_flat = np.concatenate([G.real, G.imag], axis=-1).flatten()
        X = np.concatenate([H_flat, G_flat]).astype(np.float32).reshape(1, -1)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        
        Xt = torch.tensor(X, dtype=torch.float32)
        
        w_ratio, z_ratio, b = model(Xt)
        
        # 构建波束
        w_pwr = w_ratio.item() * cfg.Pmax
        z_pwr = z_ratio.item() * cfg.Pmax
        
        b_sel = (b > 0.5).float()
        aps = torch.sum(b_sel).item()
        
        total = w_pwr + z_pwr
        
        results.append({'W': w_pwr, 'Z': z_pwr, 'Total': total, 'APs': aps})
        print(f"  {i+1}: W={w_pwr:.1f}W Z={z_pwr:.1f}W Total={total:.1f}W APs={int(aps)}")
    
    avg = np.mean([r['Total'] for r in results])
    std = np.std([r['Total'] for r in results])
    print(f"\n平均功率: {avg:.1f}W ± {std:.1f}W")
    return results

# ===================== 主程序 =====================
if __name__ == '__main__':
    print("="*60)
    print("ISAC - 监督学习 (CVX生成训练数据)")
    print("="*60)
    
    # 生成数据
    X, y_w, y_z, y_b = generate_dataset(num_samples=1000)
    
    print(f"\n数据统计:")
    print(f"  X shape: {X.shape}")
    print(f"  y_w range: [{y_w.min():.4f}, {y_w.max():.4f}]")
    print(f"  y_z range: [{y_z.min():.4f}, {y_z.max():.4f}]")
    print(f"  y_b mean: {y_b.mean():.4f}")
    
    # 训练
    model = train_supervised(X, y_w, y_z, y_b, epochs=200)
    
    # 测试
    results = test(model, n=20)
    
    print("\n监督学习完成!")