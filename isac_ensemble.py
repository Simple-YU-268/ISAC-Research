#!/usr/bin/env python3
"""
ISAC - 简化版GNN + 集成学习
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

def generate_labels(n, method='mix'):
    """多样化标签"""
    W_labels, Z_labels, B_labels = [], [], []
    
    for i in range(n):
        if method == 'optimal':
            w = cfg.Pmax * 0.7 / (cfg.M * cfg.K)
            z = cfg.Pmax * 0.3 / cfg.M
            b = np.zeros((cfg.M, cfg.P))
            for p in range(cfg.P):
                selected = np.random.choice(cfg.M, cfg.N_req, replace=False)
                b[selected, p] = 1
        elif method == 'sparse':
            w = cfg.Pmax * 0.5 / (cfg.M * cfg.K)
            z = cfg.Pmax * 0.15 / cfg.M
            b = np.zeros((cfg.M, cfg.P))
            for p in range(cfg.P):
                selected = np.random.choice(cfg.M, 2, replace=False)
                b[selected, p] = 1
        elif method == 'dense':
            w = cfg.Pmax * 0.8 / (cfg.M * cfg.K)
            z = cfg.Pmax * 0.35 / cfg.M
            b = (np.random.rand(cfg.M, cfg.P) > 0.5).astype(float)
        else:  # mix
            if i % 3 == 0:
                w = cfg.Pmax * 0.7 / (cfg.M * cfg.K)
                z = cfg.Pmax * 0.3 / cfg.M
                b = np.zeros((cfg.M, cfg.P))
                for p in range(cfg.P):
                    selected = np.random.choice(cfg.M, cfg.N_req, replace=False)
                    b[selected, p] = 1
            elif i % 3 == 1:
                w = cfg.Pmax * 0.4 / (cfg.M * cfg.K)
                z = cfg.Pmax * 0.1 / cfg.M
                b = np.zeros((cfg.M, cfg.P))
            else:
                w = cfg.Pmax * 0.85 / (cfg.M * cfg.K)
                z = cfg.Pmax * 0.4 / cfg.M
                b = (np.random.rand(cfg.M, cfg.P) > 0.3).astype(float)
        
        W_labels.append(w)
        Z_labels.append(z)
        B_labels.append(b.flatten())
    
    return np.array(W_labels).reshape(-1,1), np.array(Z_labels).reshape(-1,1), np.array(B_labels)

class EnsembleISAC(nn.Module):
    """集成学习: 3个不同子网络"""
    def __init__(self):
        super().__init__()
        
        hd = cfg.M * cfg.K * cfg.Nt * 2
        
        # 子网络1: 深层
        self.net1 = nn.Sequential(
            nn.Linear(hd, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )
        
        # 子网络2: 宽层
        self.net2 = nn.Sequential(
            nn.Linear(hd, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
        )
        
        # 子网络3: 残差
        self.net3_pre = nn.Linear(hd, 64)
        self.net3_post = nn.Linear(64, 64)
        
        # 输出融合
        self.W_head = nn.Sequential(nn.Linear(64*3, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.Z_head = nn.Sequential(nn.Linear(64*3, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.B_head = nn.Sequential(nn.Linear(64*3, 32), nn.ReLU(), nn.Linear(32, cfg.M*cfg.P), nn.Sigmoid())
    
    def forward(self, x):
        # x: (B, M*K*Nt*2) 展平
        h1 = self.net1(x)
        h2 = self.net2(x)
        h3 = F.leaky_relu(self.net3_pre(x))
        h3 = F.leaky_relu(self.net3_post(h3) + h3)
        
        # 融合
        h = torch.cat([h1, h2, h3], dim=-1)
        
        W_ratio = self.W_head(h)
        Z_ratio = self.Z_head(h)
        B = self.B_head(h)
        
        return W_ratio, Z_ratio, B

def train(epochs=300, bs=64, lr=1e-3):
    model = EnsembleISAC()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.MSELoss()
    
    print("="*60)
    print("Training Ensemble ISAC")
    print("="*60)
    
    t0 = time.time()
    
    for e in range(epochs):
        H = generate_data(bs)
        methods = ['optimal', 'sparse', 'dense']
        method = methods[e % 3]
        y_W, y_Z, y_B = generate_labels(bs, method)
        
        Xt = torch.tensor(H.reshape(bs, -1), dtype=torch.float32)
        y_Wt = torch.tensor(y_W / cfg.Pmax, dtype=torch.float32)
        y_Zt = torch.tensor(y_Z / cfg.Pmax, dtype=torch.float32)
        y_Bt = torch.tensor(y_B, dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Xt)
        
        loss_W = criterion(W_ratio, y_Wt)
        loss_Z = criterion(Z_ratio, y_Zt)
        loss_B = criterion(B, y_Bt)
        
        loss = loss_W + loss_Z + loss_B * 0.3
        
        # 约束
        W_pwr = W_ratio.mean() * cfg.Pmax
        Z_pwr = Z_ratio.mean() * cfg.Pmax
        pwr_viol = F.relu(W_pwr + Z_pwr - cfg.Pmax)
        loss = loss + pwr_viol * 20
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        
        if e % 30 == 0:
            print(f"Epoch {e:3d} | Loss: {loss.item():.4f} | "
                  f"W: {W_pwr.item():.1f}W | Z: {Z_pwr.item():.1f}W | "
                  f"{time.time()-t0:.1f}s")
    
    torch.save(model.state_dict(), 'isac_ensemble.pth')
    print(f"\n完成! 保存 isac_ensemble.pth")
    return model

def test(model, n=20):
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
        
        B_sel = (B > 0.5).float()
        aps = torch.sum(B_sel).item()
        
        results.append({'W': W_pwr, 'Z': Z_pwr, 'Total': total, 'APs': aps})
        print(f"  {i+1}: W={W_pwr:.1f}W Z={Z_pwr:.1f}W Total={total:.1f}W APs={int(aps)}")
    
    avg = np.mean([r['Total'] for r in results])
    std = np.std([r['Total'] for r in results])
    print(f"\n平均功率: {avg:.1f}W ± {std:.1f}W")
    return results

if __name__ == '__main__':
    print("="*60)
    print("ISAC - 集成学习")
    print("="*60)
    
    model = train(epochs=300, bs=64, lr=1e-3)
    results = test(model, n=20)