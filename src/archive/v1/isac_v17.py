#!/usr/bin/env python3
"""
ISAC - v17 (更精细的功率控制 + AP选择)
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

def generate_labels_v17(n):
    W_labels, Z_labels, B_labels = [], [], []
    
    for i in range(n):
        b = np.random.rand(cfg.M, cfg.P) * 0.25
        for p in range(cfg.P):
            selected = np.random.choice(cfg.M, cfg.N_req, replace=False)
            b[selected, p] = 0.9 + np.random.rand(cfg.N_req) * 0.1
        
        W_labels.append(0.7)
        Z_labels.append(0.3)
        B_labels.append(b.flatten())
    
    return np.array(W_labels).reshape(-1,1), np.array(Z_labels).reshape(-1,1), np.array(B_labels)

class ISAC_v17(nn.Module):
    def __init__(self):
        super().__init__()
        
        hd = cfg.M * cfg.K * cfg.Nt * 2
        
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
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

def train_v17(epochs=1000, bs=64, lr=8e-4):
    model = ISAC_v17()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=200, T_mult=2)
    
    t0 = time.time()
    
    for e in range(epochs):
        H = generate_data(bs)
        y_W, y_Z, y_B = generate_labels_v17(bs)
        
        Xt = torch.tensor(H.reshape(bs, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Xt)
        
        W_target = torch.tensor(y_W, dtype=torch.float32).squeeze()
        Z_target = torch.tensor(y_Z, dtype=torch.float32).squeeze()
        B_target = torch.tensor(y_B, dtype=torch.float32)
        
        loss_W = F.mse_loss(W_ratio.squeeze(), W_target)
        loss_Z = F.mse_loss(Z_ratio.squeeze(), Z_target)
        loss_B = F.mse_loss(B, B_target)
        
        loss = loss_W + loss_Z + loss_B * 0.2
        
        # 精确功率控制
        W_pwr = W_ratio.squeeze() * cfg.Pmax
        Z_pwr = Z_ratio.squeeze() * cfg.Pmax
        total_pwr = W_pwr + Z_pwr
        
        # 只有超过才惩罚
        loss = loss + F.relu(total_pwr - cfg.Pmax).mean() * 50
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    
    torch.save(model.state_dict(), 'isac_v17.pth')
    print(f"v17完成! 用时: {time.time()-t0:.1f}s")
    return model

def test_v17(model, n=20):
    model.eval()
    
    results = []
    for i in range(n):
        H = generate_data(1)
        Xt = torch.tensor(H.reshape(1, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Xt)
        
        W_pwr = W_ratio.item() * cfg.Pmax
        Z_pwr = Z_ratio.item() * cfg.Pmax
        total = W_pwr + Z_pwr
        
        b_val = B.view(cfg.M, cfg.P)
        aps_per_target = []
        for p in range(cfg.P):
            b_col = b_val[:, p]
            selected = torch.topk(b_col, cfg.N_req).indices
            aps_per_target.append(len(selected))
        
        total_aps = sum(aps_per_target)
        results.append({'W': W_pwr, 'Z': Z_pwr, 'Total': total, 'APs': total_aps})
    
    avg_pwr = np.mean([r['Total'] for r in results])
    avg_aps = np.mean([r['APs'] for r in results])
    print(f"v17: 功率={avg_pwr:.1f}W, APs={avg_aps:.0f}")
    return results

if __name__ == '__main__':
    model = train_v17(epochs=1000, bs=64, lr=8e-4)
    test_v17(model, n=20)