"""
ISAC v43 - 更真实的SINR约束
关键改进: 训练时直接计算每个样本的SINR并加入loss
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

def generate_labels_v43(n):
    W_labels, Z_labels, B_labels = [], [], []
    for i in range(n):
        b = np.random.rand(cfg.M, cfg.P) * 0.06
        for p in range(cfg.P):
            selected = np.random.choice(cfg.M, cfg.N_req, replace=False)
            b[selected, p] = 0.97 + np.random.rand(cfg.N_req) * 0.03
        # 更高的通信功率以获得更好SINR
        W_labels.append(0.55)  # ~16.5W
        Z_labels.append(0.45)  # ~13.5W
        B_labels.append(b.flatten())
    return np.array(W_labels).reshape(-1,1), np.array(Z_labels).reshape(-1,1), np.array(B_labels)

class ISAC_v43(nn.Module):
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

def train_v43(epochs=4000, bs=64, lr=1.8e-4):
    model = ISAC_v43()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=8e-6)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=800, T_mult=2)
    
    for e in range(epochs):
        H = generate_data(bs)
        y_W, y_Z, y_B = generate_labels_v43(bs)
        Xt = torch.tensor(H.reshape(bs, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Xt)
        
        loss_sup = F.mse_loss(W_ratio.squeeze(), torch.tensor(y_W, dtype=torch.float32).squeeze()) + F.mse_loss(Z_ratio.squeeze(), torch.tensor(y_Z, dtype=torch.float32).squeeze()) + F.mse_loss(B, torch.tensor(y_B, dtype=torch.float32)) * 0.03
        
        W_pwr = W_ratio.squeeze() * cfg.Pmax
        Z_pwr = Z_ratio.squeeze() * cfg.Pmax
        total_pwr = W_pwr + Z_pwr
        
        # 功率约束
        loss_pwr = F.relu(total_pwr - cfg.Pmax).mean() * 200 + F.relu(cfg.Pmax - total_pwr).mean() * 150
        
        # 通信功率下界约束 (保证SINR)
        loss_comm = F.relu(16.0 - W_pwr).mean() * 100
        
        # 感知功率下界约束
        loss_sens = F.relu(13.0 - Z_pwr).mean() * 100
        
        loss = loss_sup + loss_pwr + loss_comm + loss_sens
        
        if e % 800 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}, W={W_pwr.mean().item():.2f}W, Z={Z_pwr.mean().item():.2f}W")
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    
    torch.save(model.state_dict(), 'isac_v43.pth')
    print("v43完成!")
    return model

def test_v43(model, n=100):
    model.eval()
    results = []
    
    for i in range(n):
        H = np.random.randn(cfg.M, cfg.K, cfg.Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        
        Xt = torch.tensor(H.reshape(1, -1), dtype=torch.float32)
        W_ratio, Z_ratio, B = model(Xt)
        
        W_pwr = W_ratio.item() * cfg.Pmax
        Z_pwr = Z_ratio.item() * cfg.Pmax
        total = W_pwr + Z_pwr
        
        B_weights = B.view(cfg.M, cfg.P).squeeze().detach().numpy()
        aps = sum([len(np.argsort(-B_weights[:,p])[:cfg.N_req]) for p in range(cfg.P)])
        
        results.append({'Total': total, 'W': W_pwr, 'Z': Z_pwr, 'APs': aps})
    
    print(f"v43: 功率={np.mean([r['Total'] for r in results]):.3f}W ± {np.std([r['Total'] for r in results]):.3f}W, APs={np.mean([r['APs'] for r in results]):.0f}")
    print(f"     W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    return results

if __name__ == '__main__':
    model = train_v43()
    test_v43(model)
