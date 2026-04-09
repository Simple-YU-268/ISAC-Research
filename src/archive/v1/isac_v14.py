#!/usr/bin/env python3
"""
ISAC - v14 (平衡AP选择 + 软标签)
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

def generate_labels_v14(n):
    """软标签 + 合理AP数量"""
    W_labels, Z_labels, B_labels = [], [], []
    
    for i in range(n):
        r = np.random.rand()
        if r < 0.4:
            w_ratio, z_ratio = 0.7, 0.3
            b = np.zeros((cfg.M, cfg.P))
            for p in range(cfg.P):
                selected = np.random.choice(cfg.M, cfg.N_req, replace=False)
                b[selected, p] = 1.0
        else:
            w_ratio, z_ratio = 0.6, 0.25
            b = np.random.rand(cfg.M, cfg.P) * 0.5
        
        W_labels.append(w_ratio)
        Z_labels.append(z_ratio)
        B_labels.append(b.flatten())
    
    return np.array(W_labels).reshape(-1,1), np.array(Z_labels).reshape(-1,1), np.array(B_labels)

class ISAC_v14(nn.Module):
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
        self.B_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, cfg.M * cfg.P))
    
    def forward(self, x):
        emb = self.encoder(x)
        W_ratio = torch.sigmoid(self.W_head(emb))
        Z_ratio = torch.sigmoid(self.Z_head(emb))
        B_logits = self.B_head(emb)
        return W_ratio, Z_ratio, B_logits

def train_v14(epochs=600, bs=64, lr=1e-3):
    model = ISAC_v14()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2)
    
    print("="*60)
    print("Training ISAC v14")
    print("="*60)
    
    t0 = time.time()
    
    for e in range(epochs):
        H = generate_data(bs)
        y_W, y_Z, y_B = generate_labels_v14(bs)
        
        Xt = torch.tensor(H.reshape(bs, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B_logits = model(Xt)
        
        W_target = torch.tensor(y_W, dtype=torch.float32).squeeze()
        Z_target = torch.tensor(y_Z, dtype=torch.float32).squeeze()
        B_target = torch.tensor(y_B, dtype=torch.float32)
        
        loss_W = F.mse_loss(W_ratio.squeeze(), W_target)
        loss_Z = F.mse_loss(Z_ratio.squeeze(), Z_target)
        
        # BCE - 让网络学习软标签
        loss_B = F.binary_cross_entropy_with_logits(B_logits, B_target)
        
        loss = loss_W + loss_Z + loss_B * 0.3
        
        # 功率约束
        W_pwr = W_ratio.squeeze() * cfg.Pmax
        Z_pwr = Z_ratio.squeeze() * cfg.Pmax
        total_pwr = W_pwr + Z_pwr
        
        overuse = F.relu(total_pwr - cfg.Pmax).mean() * 50
        underuse = F.relu(cfg.Pmax * 0.2 - total_pwr).mean() * 20
        loss = loss + overuse + underuse
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if e % 60 == 0:
            w_pwr = W_ratio.mean().item() * cfg.Pmax
            z_pwr = Z_ratio.mean().item() * cfg.Pmax
            total = w_pwr + z_pwr
            # 用低阈值看
            aps = (torch.sigmoid(B_logits) > 0.3).float().sum(dim=1).mean().item()
            print(f"Epoch {e:3d} | Loss: {loss.item():.4f} | "
                  f"W: {w_pwr:.1f}W | Z: {z_pwr:.1f}W | Total: {total:.1f}W | APs: {aps:.1f}")
    
    torch.save(model.state_dict(), 'isac_v14.pth')
    print(f"\n完成!")
    return model

def test_v14(model, n=20):
    model.eval()
    print("\n测试 (阈值0.3):")
    
    results = []
    for i in range(n):
        H = generate_data(1)
        Xt = torch.tensor(H.reshape(1, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B_logits = model(Xt)
        
        W_pwr = W_ratio.item() * cfg.Pmax
        Z_pwr = Z_ratio.item() * cfg.Pmax
        total = W_pwr + Z_pwr
        
        # 不同阈值
        for thresh in [0.3, 0.5]:
            B_prob = torch.sigmoid(B_logits)
            B_sel = (B_prob > thresh).float()
            aps = B_sel.sum().item()
            
            print(f"  阈值 {thresh}: APs={int(aps)}")
        
        results.append({'W': W_pwr, 'Z': Z_pwr, 'Total': total})
    
    avg = np.mean([r['Total'] for r in results])
    print(f"\n平均功率: {avg:.1f}W")
    return results

if __name__ == '__main__':
    print("="*60)
    print("ISAC v14")
    print("="*60)
    
    model = train_v14(epochs=600, bs=64, lr=1e-3)
    results = test_v14(model, n=20)