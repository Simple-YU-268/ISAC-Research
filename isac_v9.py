#!/usr/bin/env python3
"""
ISAC - v9 (修正功率计算)
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

def generate_labels_v9(n):
    W_labels, Z_labels, B_labels = [], [], []
    
    for i in range(n):
        r = np.random.rand()
        if r < 0.33:
            w = cfg.Pmax * 0.7 / cfg.M  # 每AP功率
            z = cfg.Pmax * 0.3 / cfg.M
            b = np.zeros((cfg.M, cfg.P))
            for p in range(cfg.P):
                selected = np.random.choice(cfg.M, cfg.N_req, replace=False)
                b[selected, p] = 1
        elif r < 0.66:
            w = cfg.Pmax * 0.5 / cfg.M
            z = cfg.Pmax * 0.2 / cfg.M
            b = np.zeros((cfg.M, cfg.P))
            for p in range(cfg.P):
                selected = np.random.choice(cfg.M, 2, replace=False)
                b[selected, p] = 1
        else:
            w = cfg.Pmax * 0.85 / cfg.M
            z = cfg.Pmax * 0.4 / cfg.M
            b = (np.random.rand(cfg.M, cfg.P) > 0.3).astype(float)
        
        W_labels.append(w)
        Z_labels.append(z)
        B_labels.append(b.flatten())
    
    return np.array(W_labels).reshape(-1,1), np.array(Z_labels).reshape(-1,1), np.array(B_labels)

class ISAC_v9(nn.Module):
    def __init__(self):
        super().__init__()
        
        hd = cfg.M * cfg.K * cfg.Nt * 2
        
        self.encoder = nn.Sequential(
            nn.Linear(hd, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )
        
        # W头 - 每AP功率 (非负)
        self.W_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.M),
            nn.Softmax(dim=-1)  # 归一化
        )
        
        # Z头
        self.Z_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.M),
            nn.Softmax(dim=-1)
        )
        
        # B头
        self.B_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.M * cfg.P),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        emb = self.encoder(x)
        
        W_ratio = self.W_head(emb)  # (B, M), sum=1
        Z_ratio = self.Z_head(emb)  # (B, M), sum=1
        B = self.B_head(emb)  # (B, M*P)
        
        return W_ratio, Z_ratio, B

def train_v9(epochs=400, bs=64, lr=1e-3):
    model = ISAC_v9()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2)
    bce = nn.BCELoss()
    
    print("="*60)
    print("Training ISAC v9")
    print("="*60)
    
    t0 = time.time()
    
    for e in range(epochs):
        H = generate_data(bs)
        y_W, y_Z, y_B = generate_labels_v9(bs)
        
        Xt = torch.tensor(H.reshape(bs, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Xt)
        
        # 目标: 每AP功率
        W_target = torch.tensor(y_W, dtype=torch.float32)
        Z_target = torch.tensor(y_Z, dtype=torch.float32)
        B_target = torch.tensor(y_B, dtype=torch.float32)
        
        # W, Z 损失 - 直接MSE
        loss_W = F.mse_loss(W_ratio, W_target)
        loss_Z = F.mse_loss(Z_ratio, Z_target)
        
        # B损失
        loss_B = bce(B, B_target)
        
        loss = loss_W + loss_Z + loss_B * 0.5
        
        # 功率约束 (总功率)
        W_pwr = W_ratio.sum(dim=1) * cfg.Pmax  # sum(W_ratio)=1, so sum*cfg.Pmax = total
        Z_pwr = Z_ratio.sum(dim=1) * cfg.Pmax
        
        # 强制: W_pwr + Z_pwr = Pmax
        total_pwr = W_pwr + Z_pwr
        overuse = F.relu(total_pwr - cfg.Pmax).mean() * 100
        underuse = F.relu(cfg.Pmax * 0.3 - total_pwr).mean() * 30
        
        loss = loss + overuse + underuse
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if e % 40 == 0:
            w_pwr = W_ratio.sum(dim=1).mean().item() * cfg.Pmax
            z_pwr = Z_ratio.sum(dim=1).mean().item() * cfg.Pmax
            total = w_pwr + z_pwr
            aps = (B > 0.5).float().sum(dim=1).mean().item()
            print(f"Epoch {e:3d} | Loss: {loss.item():.4f} | "
                  f"W: {w_pwr:.1f}W | Z: {z_pwr:.1f}W | Total: {total:.1f}W | APs: {aps:.1f} | {time.time()-t0:.1f}s")
    
    torch.save(model.state_dict(), 'isac_v9.pth')
    print(f"\n完成!")
    return model

def test_v9(model, n=20):
    model.eval()
    print("\n测试:")
    
    results = []
    for i in range(n):
        H = generate_data(1)
        Xt = torch.tensor(H.reshape(1, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Xt)
        
        W_pwr = W_ratio.sum().item() * cfg.Pmax
        Z_pwr = Z_ratio.sum().item() * cfg.Pmax
        total = W_pwr + Z_pwr
        
        B_sel = (B > 0.5).float()
        aps = B_sel.sum().item()
        
        results.append({'W': W_pwr, 'Z': Z_pwr, 'Total': total, 'APs': aps})
        print(f"  {i+1}: W={W_pwr:.1f}W Z={Z_pwr:.1f}W Total={total:.1f}W APs={int(aps)}")
    
    avg = np.mean([r['Total'] for r in results])
    std = np.std([r['Total'] for r in results])
    aps_avg = np.mean([r['APs'] for r in results])
    print(f"\n平均功率: {avg:.1f}W ± {std:.1f}W")
    print(f"平均AP数: {aps_avg:.1f}")
    return results

if __name__ == '__main__':
    print("="*60)
    print("ISAC v9")
    print("="*60)
    
    model = train_v9(epochs=400, bs=64, lr=1e-3)
    results = test_v9(model, n=20)