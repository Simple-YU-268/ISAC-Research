"""
ISAC v33 - 加入GNN图注意力机制
参考论文: Heterogeneous Graph Neural Network for Cooperative ISAC Beamforming
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

def generate_labels(n):
    W_labels, Z_labels, B_labels = [], [], []
    for i in range(n):
        b = np.random.rand(cfg.M, cfg.P) * 0.025
        for p in range(cfg.P):
            selected = np.random.choice(cfg.M, cfg.N_req, replace=False)
            b[selected, p] = 0.993 + np.random.rand(cfg.N_req) * 0.007
        W_labels.append(0.756)
        Z_labels.append(0.356)
        B_labels.append(b.flatten())
    return np.array(W_labels).reshape(-1,1), np.array(Z_labels).reshape(-1,1), np.array(B_labels)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.a = nn.Linear(out_dim * 2, 1)
    
    def forward(self, x):
        # x: (B, M, in_dim)
        h = self.W(x)  # (B, M, out_dim)
        h1 = h.unsqueeze(2).expand(-1, -1, cfg.M, -1)
        h2 = h.unsqueeze(1).expand(-1, cfg.M, -1, -1)
        att = self.a(torch.cat([h1, h2], dim=-1)).squeeze(-1)  # (B, M, M)
        att = F.softmax(att, dim=-1)
        return torch.bmm(att, h)  # (B, M, out_dim)

class ISAC_v33(nn.Module):
    def __init__(self):
        super().__init__()
        hd = cfg.M * cfg.K * cfg.Nt * 2
        
        # 初始投影
        self.proj = nn.Linear(hd, 64)
        
        # 图注意力层
        self.gat1 = GraphAttentionLayer(64, 64)
        self.gat2 = GraphAttentionLayer(64, 32)
        
        # 全局池化
        self.pool = nn.Linear(32, 1)
        
        # 输出头
        self.W_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        self.Z_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        self.B_head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, cfg.M * cfg.P), nn.Sigmoid())
    
    def forward(self, x):
        # x: (B, M*K*Nt*2)
        h = x.view(-1, cfg.M, cfg.K * cfg.Nt * 2)
        h = self.proj(h)  # (B, M, 64)
        
        # 图注意力
        h = self.gat1(h)
        h = F.leaky_relu(h, 0.2)
        h = self.gat2(h)
        
        # 全局表示
        w = self.pool(h).squeeze(-1)  # (B, M)
        h_agg = (h * w.softmax(-1).unsqueeze(-1)).sum(dim=1)  # (B, 32)
        
        return self.W_head(h_agg), self.Z_head(h_agg), self.B_head(h_agg)

def train_v33(epochs=5000, bs=64, lr=2e-4):
    model = ISAC_v33()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1000, T_mult=2)
    
    for e in range(epochs):
        H = generate_data(bs)
        y_W, y_Z, y_B = generate_labels(bs)
        Xt = torch.tensor(H.reshape(bs, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Xt)
        
        loss_sup = F.mse_loss(W_ratio.squeeze(), torch.tensor(y_W, dtype=torch.float32).squeeze()) + F.mse_loss(Z_ratio.squeeze(), torch.tensor(y_Z, dtype=torch.float32).squeeze()) + F.mse_loss(B, torch.tensor(y_B, dtype=torch.float32)) * 0.01
        
        total_pwr = (W_ratio.squeeze() + Z_ratio.squeeze()) * cfg.Pmax
        loss_pwr = F.relu(total_pwr - cfg.Pmax).mean() * 400 + F.relu(cfg.Pmax - total_pwr).mean() * 300
        
        loss = loss_sup + loss_pwr
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    
    torch.save(model.state_dict(), 'isac_v33.pth')
    print("v33完成!")
    return model

def test_v33(model, n=100):
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
    print(f"v33: 功率={np.mean([r['Total'] for r in results]):.3f}W ± {np.std([r['Total'] for r in results]):.3f}W, APs={np.mean([r['APs'] for r in results]):.0f}")
    return results

if __name__ == '__main__':
    model = train_v33()
    test_v33(model)
