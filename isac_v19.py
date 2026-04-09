import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

cfg = type('C', (), {'M': 16, 'K': 8, 'P': 4, 'Nt': 4, 'Pmax': 30, 'N_req': 4})()

def generate_data(n):
    H = np.random.randn(n, cfg.M, cfg.K, cfg.Nt*2).astype(np.float32)
    H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
    return H

def generate_labels_v19(n):
    W_labels, Z_labels, B_labels = [], [], []
    for i in range(n):
        b = np.random.rand(cfg.M, cfg.P) * 0.15
        for p in range(cfg.P):
            selected = np.random.choice(cfg.M, cfg.N_req, replace=False)
            b[selected, p] = 0.92 + np.random.rand(cfg.N_req) * 0.08
        W_labels.append(0.72)
        Z_labels.append(0.32)
        B_labels.append(b.flatten())
    return np.array(W_labels).reshape(-1,1), np.array(Z_labels).reshape(-1,1), np.array(B_labels)

class ISAC_v19(nn.Module):
    def __init__(self):
        super().__init__()
        hd = cfg.M * cfg.K * cfg.Nt * 2
        self.encoder = nn.Sequential(nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256), nn.Dropout(0.08), nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128), nn.Linear(128, 64))
        self.W_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.Z_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.B_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, cfg.M * cfg.P), nn.Sigmoid())
    
    def forward(self, x):
        emb = self.encoder(x)
        return self.W_head(emb), self.Z_head(emb), self.B_head(emb)

def train_v19(epochs=1200, bs=64, lr=6e-4):
    model = ISAC_v19()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=8e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=250, T_mult=2)
    t0 = time.time()
    for e in range(epochs):
        H = generate_data(bs)
        y_W, y_Z, y_B = generate_labels_v19(bs)
        Xt = torch.tensor(H.reshape(bs, -1), dtype=torch.float32)
        W_ratio, Z_ratio, B = model(Xt)
        W_target = torch.tensor(y_W, dtype=torch.float32).squeeze()
        Z_target = torch.tensor(y_Z, dtype=torch.float32).squeeze()
        B_target = torch.tensor(y_B, dtype=torch.float32)
        loss = F.mse_loss(W_ratio.squeeze(), W_target) + F.mse_loss(Z_ratio.squeeze(), Z_target) + F.mse_loss(B, B_target) * 0.15
        total_pwr = (W_ratio.squeeze() + Z_ratio.squeeze()) * cfg.Pmax
        loss = loss + F.relu(total_pwr - cfg.Pmax).mean() * 60 + F.relu(cfg.Pmax * 0.95 - total_pwr).mean() * 30
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    torch.save(model.state_dict(), 'isac_v19.pth')
    print(f"v19完成! 用时: {time.time()-t0:.1f}s")
    return model

def test_v19(model, n=30):
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
        aps = sum([len(torch.topk(b_val[:, p], cfg.N_req).indices) for p in range(cfg.P)])
        results.append({'W': W_pwr, 'Z': Z_pwr, 'Total': total, 'APs': aps})
    print(f"v19: 功率={np.mean([r['Total'] for r in results]):.1f}W ± {np.std([r['Total'] for r in results]):.1f}W, APs={np.mean([r['APs'] for r in results]):.0f}")
    return results

if __name__ == '__main__':
    model = train_v19()
    test_v19(model)
