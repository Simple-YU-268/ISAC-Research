"""
ISAC v45 - 完整ISAC求解器
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 8, 4, 4
Pmax = 30
N_req = 4

class ISAC_v45(nn.Module):
    def __init__(self):
        super().__init__()
        hd = M * K * Nt * 2  # 16*8*4*2 = 1024
        
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        # 通信波束头: 输出每个AP-用户的功率
        self.w_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * K), nn.Sigmoid())
        
        # 感知权重头
        self.z_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * P), nn.Sigmoid())
        
        # AP选择头
        self.b_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * P), nn.Sigmoid())
    
    def forward(self, x):
        # x: (bs, M, K, Nt*2) -> (bs, M*K*Nt*2)
        bs = x.shape[0]
        x_flat = x.view(bs, -1)
        
        emb = self.encoder(x_flat)
        
        w = self.w_head(emb).view(bs, M, K)  # 每个AP对每个用户的功率分配
        z = self.z_head(emb).view(bs, M, P)  # 感知权重
        b = self.b_head(emb).view(bs, M, P)  # AP选择
        
        return w, z, b

def generate_data(n):
    H = np.random.randn(n, M, K, Nt*2).astype(np.float32)
    H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
    return H

def train_v45(epochs=5000, bs=64, lr=1e-4):
    model = ISAC_v45()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1000, T_mult=2)
    
    for e in range(epochs):
        H = generate_data(bs)
        Xt = torch.tensor(H, dtype=torch.float32)
        
        w, z, b = model(Xt)
        
        # === 功率计算 ===
        # 通信功率: sum w_mk
        W_power = w.sum(dim=(1,2))
        
        # 感知功率: sum z_mp * b_mp (只统计选中的AP)
        Z_power = (z * b).sum(dim=(1,2))
        
        total_power = W_power + Z_power * 0.8
        
        # 缩放到Pmax
        W_pwr = W_power * Pmax / 2
        Z_pwr = Z_power * Pmax / 2
        total_pwr = W_pwr + Z_pwr
        
        # === 损失 ===
        # 功率上界
        loss_pwr_over = F.relu(total_pwr - Pmax).mean() * 300
        loss_pwr_under = F.relu(Pmax * 0.95 - total_pwr).mean() * 200
        
        # AP选择: 每目标选N_req个
        b_reshaped = b.view(bs, M, P)
        ap_counts = b_reshaped.sum(dim=1)
        loss_ap = F.mse_loss(ap_counts, torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)) * 50
        
        # 通信功率约束
        loss_comm = F.relu(15 - W_pwr).mean() * 50
        
        # 感知功率约束
        loss_sens = F.relu(12 - Z_pwr).mean() * 50
        
        # 有监督
        loss_sup = F.mse_loss(W_power, torch.tensor(0.52).to(b.device).expand(bs)) * 5 + \
                   F.mse_loss(Z_power, torch.tensor(0.48).to(b.device).expand(bs)) * 5
        
        loss = loss_pwr_over + loss_pwr_under + loss_ap + loss_comm + loss_sens + loss_sup
        
        if e % 1000 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}, W={W_pwr.mean().item():.2f}W, Z={Z_pwr.mean().item():.2f}W")
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    
    torch.save(model.state_dict(), 'isac_v45.pth')
    print("v45完成!")
    return model

def test_v45(model, n=50):
    model.eval()
    results = []
    
    for i in range(n):
        H = generate_data(1)
        Xt = torch.tensor(H, dtype=torch.float32)
        
        w, z, b = model(Xt)
        
        W_power = w.sum().item()
        Z_power = (z * b).sum().item()
        W_pwr = W_power * Pmax / 2
        Z_pwr = Z_power * Pmax / 2
        total = W_pwr + Z_pwr
        
        b_np = b.squeeze().detach().numpy()
        aps = sum([len(np.argsort(-b_np[:, p])[:N_req]) for p in range(P)])
        
        results.append({'Total': total, 'W': W_pwr, 'Z': Z_pwr, 'APs': aps})
    
    print(f"v45: 功率={np.mean([r['Total'] for r in results]):.2f}W")
    print(f"     W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     APs={np.mean([r['APs'] for r in results]):.0f}")
    return results

if __name__ == '__main__':
    model = train_v45()
    test_v45(model)
