"""
ISAC v37 - 更精确的功率控制
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

cfg = type('C', (), {'M': 16, 'K': 8, 'P': 4, 'Nt': 4, 'Pmax': 30, 'N_req': 4})()

def generate_data(n):
    H = np.random.randn(n, cfg.M, cfg.K, cfg.Nt*2).astype(np.float32)
    H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
    return H

def generate_labels_v37(n):
    W_labels, Z_labels, B_labels = [], [], []
    for i in range(n):
        b = np.random.rand(cfg.M, cfg.P) * 0.024
        for p in range(cfg.P):
            selected = np.random.choice(cfg.M, cfg.N_req, replace=False)
            b[selected, p] = 0.993 + np.random.rand(cfg.N_req) * 0.007
        W_labels.append(0.756)
        Z_labels.append(0.356)
        B_labels.append(b.flatten())
    return np.array(W_labels).reshape(-1,1), np.array(Z_labels).reshape(-1,1), np.array(B_labels)

class ISAC_v37(nn.Module):
    def __init__(self):
        super().__init__()
        hd = cfg.M * cfg.K * cfg.Nt * 2
        self.encoder = nn.Sequential(nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256), nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128), nn.Linear(128, 64))
        self.W_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.Z_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.B_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, cfg.M * cfg.P), nn.Sigmoid())
    
    def forward(self, x):
        emb = self.encoder(x)
        return self.W_head(emb), self.Z_head(emb), self.B_head(emb)

def train_v37(epochs=4500, bs=64, lr=1.8e-4):
    model = ISAC_v37()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1.2e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=900, T_mult=2)
    
    for e in range(epochs):
        H = generate_data(bs)
        y_W, y_Z, y_B = generate_labels_v37(bs)
        Xt = torch.tensor(H.reshape(bs, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Xt)
        
        loss_sup = F.mse_loss(W_ratio.squeeze(), torch.tensor(y_W, dtype=torch.float32).squeeze()) + F.mse_loss(Z_ratio.squeeze(), torch.tensor(y_Z, dtype=torch.float32).squeeze()) + F.mse_loss(B, torch.tensor(y_B, dtype=torch.float32)) * 0.012
        
        W_pwr = (W_ratio.squeeze() * cfg.Pmax).mean()
        Z_pwr = (Z_ratio.squeeze() * cfg.Pmax).mean()
        total_pwr = W_pwr + Z_pwr
        
        # 非常精确的功率惩罚
        dev = total_pwr - cfg.Pmax
        loss_pwr = torch.where(dev > 0, dev.pow(2), dev.pow(2)).mean() * 500
        
        loss = loss_sup + loss_pwr
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    
    torch.save(model.state_dict(), 'isac_v37.pth')
    print("v37完成!")
    return model

def test_v37(model, n=200):
    model.eval()
    results = []
    for i in range(n):
        H = generate_data(1)
        Xt = torch.tensor(H.reshape(1, -1), dtype=torch.float32)
        W_ratio, Z_ratio, B = model(Xt)
        total = (W_ratio.item() + Z_ratio.item()) * cfg.Pmax
        B_reshaped = B.view(-1, cfg.M, cfg.P)
        aps = sum([len(torch.topk(B_reshaped[0, :, p], cfg.N_req).indices) for p in range(cfg.P)])
        results.append({'Total': total, 'APs': aps})
    print(f"v37: 功率={np.mean([r['Total'] for r in results]):.3f}W ± {np.std([r['Total'] for r in results]):.3f}W, APs={np.mean([r['APs'] for r in results]):.0f}")
    return results

if __name__ == '__main__':
    model = train_v37()
    test_v37(model)
