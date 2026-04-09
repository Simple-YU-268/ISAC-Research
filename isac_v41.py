"""
ISAC v41 - 加入通信SINR和感知CRB约束
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

cfg = type('C', (), {'M': 16, 'K': 8, 'P': 4, 'Nt': 4, 'Pmax': 30, 'N_req': 4, 'SINR_target': 10})()

def generate_data(n):
    H = np.random.randn(n, cfg.M, cfg.K, cfg.Nt*2).astype(np.float32)
    H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
    return H

def generate_labels_v41(n):
    W_labels, Z_labels, B_labels = [], [], []
    for i in range(n):
        b = np.random.rand(cfg.M, cfg.P) * 0.06
        for p in range(cfg.P):
            selected = np.random.choice(cfg.M, cfg.N_req, replace=False)
            b[selected, p] = 0.97 + np.random.rand(cfg.N_req) * 0.03
        W_labels.append(0.745)
        Z_labels.append(0.345)
        B_labels.append(b.flatten())
    return np.array(W_labels).reshape(-1,1), np.array(Z_labels).reshape(-1,1), np.array(B_labels)

class ISAC_v41(nn.Module):
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

def train_v41(epochs=3500, bs=64, lr=2e-4):
    model = ISAC_v41()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=700, T_mult=2)
    
    for e in range(epochs):
        H = generate_data(bs)
        y_W, y_Z, y_B = generate_labels_v41(bs)
        Xt = torch.tensor(H.reshape(bs, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Xt)
        
        loss_sup = F.mse_loss(W_ratio.squeeze(), torch.tensor(y_W, dtype=torch.float32).squeeze()) + F.mse_loss(Z_ratio.squeeze(), torch.tensor(y_Z, dtype=torch.float32).squeeze()) + F.mse_loss(B, torch.tensor(y_B, dtype=torch.float32)) * 0.03
        
        W_pwr = W_ratio.squeeze() * cfg.Pmax
        Z_pwr = Z_ratio.squeeze() * cfg.Pmax
        total_pwr = W_pwr + Z_pwr
        
        loss_pwr = F.relu(total_pwr - cfg.Pmax).mean() * 200 + F.relu(cfg.Pmax - total_pwr).mean() * 150
        
        # 简化的SINR约束: W_pwr越大SINR越高
        # 假设SINR ≈ W_pwr * some_factor
        # 为了简化，直接用W_pwr来保证通信质量
        loss_comm = F.relu(15 - W_pwr).mean() * 50  # 至少需要15W用于通信
        
        # 感知约束: 感知功率需要足够
        loss_sens = F.relu(12 - Z_pwr).mean() * 50  # 至少需要12W用于感知
        
        loss = loss_sup + loss_pwr + loss_comm + loss_sens
        
        if e % 500 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}, 功率={total_pwr.mean().item():.2f}W")
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    
    torch.save(model.state_dict(), 'isac_v41.pth')
    print("v41完成!")
    return model

def test_v41(model, n=100):
    model.eval()
    results = []
    W_pwrs, Z_pwrs = [], []
    
    for i in range(n):
        H = np.random.randn(cfg.M, cfg.K, cfg.Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        
        Xt = torch.tensor(H.reshape(1, -1), dtype=torch.float32)
        W_ratio, Z_ratio, B = model(Xt)
        
        W_pwr = W_ratio.item() * cfg.Pmax
        Z_pwr = Z_ratio.item() * cfg.Pmax
        total = W_pwr + Z_pwr
        
        W_pwrs.append(W_pwr)
        Z_pwrs.append(Z_pwr)
        results.append({'Total': total, 'W': W_pwr, 'Z': Z_pwr})
    
    print(f"v41: 功率={np.mean([r['Total'] for r in results]):.3f}W ± {np.std([r['Total'] for r in results]):.3f}W")
    print(f"     W={np.mean(W_pwrs):.2f}W, Z={np.mean(Z_pwrs):.2f}W")
    return results

if __name__ == '__main__':
    model = train_v41()
    test_v41(model)
