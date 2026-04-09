"""
ISAC v31 - 无监督学习版本
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

class ISAC_v31(nn.Module):
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

def unsupervised_loss(W_ratio, Z_ratio, B):
    W_pwr = (W_ratio.squeeze() * cfg.Pmax).mean()
    Z_pwr = (Z_ratio.squeeze() * cfg.Pmax).mean()
    total_pwr = W_pwr + Z_pwr
    
    loss_pwr = F.relu(total_pwr - cfg.Pmax) * 100 + F.relu(cfg.Pmax * 0.95 - total_pwr) * 50
    loss_comm = -W_pwr * 0.1
    loss_sens = -Z_pwr * 0.1
    
    B_reshaped = B.view(-1, cfg.M, cfg.P)
    ap_selections = B_reshaped.sum(dim=1).mean(dim=0)
    target_aps = torch.tensor([cfg.N_req] * cfg.P, dtype=torch.float32).to(B.device)
    loss_ap = F.mse_loss(ap_selections, target_aps) * 10
    
    return loss_pwr + loss_comm + loss_sens + loss_ap

def train_v31(epochs=5000, bs=64, lr=2e-4):
    model = ISAC_v31()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1000, T_mult=2)
    
    for e in range(epochs):
        H = generate_data(bs)
        Xt = torch.tensor(H.reshape(bs, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Xt)
        loss = unsupervised_loss(W_ratio, Z_ratio, B)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    
    torch.save(model.state_dict(), 'isac_v31.pth')
    print("v31完成!")
    return model

def test_v31(model, n=100):
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
    print(f"v31: 功率={np.mean([r['Total'] for r in results]):.3f}W ± {np.std([r['Total'] for r in results]):.3f}W, APs={np.mean([r['APs'] for r in results]):.0f}")
    return results

if __name__ == '__main__':
    model = train_v31()
    test_v31(model)
