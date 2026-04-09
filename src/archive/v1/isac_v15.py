#!/usr/bin/env python3
"""
ISAC - v15 (直接用连续值作为AP选择权重)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

cfg = type('C', (), {
    'M': 16, 'K': 8, 'P': 4, 'Nt': 4,
    'Pmax': 30, 'N_req': 4,
})()

def generate_data(n):
    H = np.random.randn(n, cfg.M, cfg.K, cfg.Nt*2).astype(np.float32)
    H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
    return H

def generate_labels_v15(n):
    """软标签 (0-1连续)"""
    W_labels, Z_labels, B_labels = [], [], []
    
    for i in range(n):
        # 随机选择AP权重
        b = np.random.rand(cfg.M, cfg.P) * 0.5  # 0-0.5随机
        # 确保有一些接近1
        if i % 3 == 0:
            for p in range(cfg.P):
                selected = np.random.choice(cfg.M, cfg.N_req, replace=False)
                b[selected, p] = 0.8 + np.random.rand(cfg.N_req) * 0.2
        
        W_labels.append(0.7)
        Z_labels.append(0.3)
        B_labels.append(b.flatten())
    
    return np.array(W_labels).reshape(-1,1), np.array(Z_labels).reshape(-1,1), np.array(B_labels)

class ISAC_v15(nn.Module):
    def __init__(self):
        super().__init__()
        
        hd = cfg.M * cfg.K * cfg.Nt * 2
        
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
        )
        
        self.W_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.Z_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.B_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, cfg.M * cfg.P), nn.Sigmoid())
    
    def forward(self, x):
        emb = self.encoder(x)
        W_ratio = self.W_head(emb)
        Z_ratio = self.Z_head(emb)
        B = self.B_head(emb)
        return W_ratio, Z_ratio, B

def train_v15(epochs=600, bs=64, lr=1e-3):
    model = ISAC_v15()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2)
    
    print("="*60)
    print("Training ISAC v15 (连续AP选择)")
    print("="*60)
    
    t0 = time.time()
    
    for e in range(epochs):
        H = generate_data(bs)
        y_W, y_Z, y_B = generate_labels_v15(bs)
        
        Xt = torch.tensor(H.reshape(bs, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Xt)
        
        W_target = torch.tensor(y_W, dtype=torch.float32).squeeze()
        Z_target = torch.tensor(y_Z, dtype=torch.float32).squeeze()
        B_target = torch.tensor(y_B, dtype=torch.float32)
        
        loss_W = F.mse_loss(W_ratio.squeeze(), W_target)
        loss_Z = F.mse_loss(Z_ratio.squeeze(), Z_target)
        loss_B = F.mse_loss(B, B_target)  # MSE代替BCE
        
        loss = loss_W + loss_Z + loss_B * 0.3
        
        # 功率约束
        W_pwr = W_ratio.squeeze() * cfg.Pmax
        Z_pwr = Z_ratio.squeeze() * cfg.Pmax
        total_pwr = W_pwr + Z_pwr
        
        overuse = F.relu(total_pwr - cfg.Pmax).mean() * 50
        underuse = F.relu(cfg.Pmax * 0.2 - total_pwr).mean() * 20
        loss = loss + overuse + underuse
        
        # 鼓励选择AP - 使用B的均值作为奖励
        avg_sel = B.mean()
        loss = loss - avg_sel * 0.05  # 小奖励
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if e % 60 == 0:
            w_pwr = W_ratio.mean().item() * cfg.Pmax
            z_pwr = Z_ratio.mean().item() * cfg.Pmax
            total = w_pwr + z_pwr
            b_val = B.mean().item()
            print(f"Epoch {e:3d} | Loss: {loss.item():.4f} | "
                  f"W: {w_pwr:.1f}W | Z: {z_pwr:.1f}W | Total: {total:.1f}W | B_mean: {b_val:.3f}")
    
    torch.save(model.state_dict(), 'isac_v15.pth')
    print(f"\n完成!")
    return model

def test_v15(model, n=20):
    model.eval()
    print("\n测试:")
    
    results = []
    for i in range(n):
        H = generate_data(1)
        Xt = torch.tensor(H.reshape(1, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Xt)
        
        W_pwr = W_ratio.item() * cfg.Pmax
        Z_pwr = Z_ratio.item() * cfg.Pmax
        total = W_pwr + Z_pwr
        
        # 直接用连续值作为权重
        b_val = B.view(cfg.M, cfg.P)
        
        # 选最大的N_req个AP
        aps_per_target = []
        for p in range(cfg.P):
            b_col = b_val[:, p]
            top_k = min(cfg.N_req, cfg.M)
            selected_aps = torch.topk(b_col, top_k).indices
            aps_per_target.append(len(selected_aps))
        
        total_aps = sum(aps_per_target)
        
        results.append({'W': W_pwr, 'Z': Z_pwr, 'Total': total, 'APs': total_aps})
        print(f"  {i+1}: W={W_pwr:.1f}W Z={Z_pwr:.1f}W Total={total:.1f}W APs={total_aps} | 每目标: {aps_per_target}")
    
    avg = np.mean([r['Total'] for r in results])
    aps_avg = np.mean([r['APs'] for r in results])
    print(f"\n平均功率: {avg:.1f}W")
    print(f"平均AP数: {aps_avg:.1f}")
    return results

if __name__ == '__main__':
    print("="*60)
    print("ISAC v15")
    print("="*60)
    
    model = train_v15(epochs=600, bs=64, lr=1e-3)
    results = test_v15(model, n=20)