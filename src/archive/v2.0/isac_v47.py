"""
ISAC v47 - 使用更强的通信功率约束，目标是达到有效SINR
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 8, 4, 4
Pmax = 30
N_req = 4

class ISAC_v47(nn.Module):
    def __init__(self):
        super().__init__()
        hd = M * K * Nt * 2
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        # 通信功率分配 (M, K)
        self.w_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * K), nn.Sigmoid())
        
        # 感知权重 (M, P)
        self.z_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * P), nn.Sigmoid())
        
        # AP选择 (M, P)
        self.b_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * P), nn.Sigmoid())
    
    def forward(self, x):
        bs = x.shape[0]
        x_flat = x.view(bs, -1)
        emb = self.encoder(x_flat)
        w = self.w_head(emb).view(bs, M, K)
        z = self.z_head(emb).view(bs, M, P)
        b = self.b_head(emb).view(bs, M, P)
        return w, z, b

def train_v47(epochs=7000, bs=64, lr=6e-5):
    model = ISAC_v47()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1400, T_mult=2)
    
    # 目标: 通信功率至少20W以达到有效SINR
    target_W = 20.0
    
    for e in range(epochs):
        H = np.random.randn(bs, M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        Ht = torch.tensor(H, dtype=torch.float32)
        
        w, z, b = model(Ht)
        
        W_power = w.sum(dim=(1,2))
        Z_power = (z * b).sum(dim=(1,2))
        
        W_pwr = W_power * Pmax * 0.7
        Z_pwr = Z_power * Pmax * 0.4
        total_pwr = W_pwr + Z_pwr
        
        # 功率约束
        loss_pwr_over = F.relu(total_pwr - Pmax).mean() * 400
        loss_pwr_under = F.relu(Pmax * 0.98 - total_pwr).mean() * 350
        
        # AP选择
        b_reshaped = b.view(bs, M, P)
        ap_counts = b_reshaped.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 80
        
        # === 通信功率核心约束 ===
        # 强制W >= target_W
        loss_comm = F.relu(target_W - W_pwr).mean() * 200
        
        # 限制Z不能太大，否则无法保证通信
        loss_z_limit = F.relu(Z_pwr - 12).mean() * 30
        
        loss = loss_pwr_over + loss_pwr_under + loss_ap + loss_comm + loss_z_limit
        
        if e % 1400 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}, W={W_pwr.mean().item():.2f}W, Z={Z_pwr.mean().item():.2f}W, total={total_pwr.mean().item():.2f}W")
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    
    torch.save(model.state_dict(), 'isac_v47.pth')
    print("v47完成!")
    return model

def test_v47(model, n=100):
    model.eval()
    results = []
    
    for i in range(n):
        H = np.random.randn(M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        Xt = torch.tensor(H.reshape(1, M, K, Nt*2), dtype=torch.float32)
        
        w, z, b = model(Xt)
        
        W_power = w.squeeze().sum().item()
        Z_power = (z * b).squeeze().sum().item()
        W_pwr = W_power * Pmax * 0.7
        Z_pwr = Z_power * Pmax * 0.4
        total = W_pwr + Z_pwr
        
        # SINR计算
        H_real = H[:, :, :Nt]
        signal = np.sum(np.sum(H_real**2, axis=2), axis=0) * (W_pwr / M)
        total_sig = np.sum(signal)
        interference = total_sig - signal + 0.01
        sinr = signal / (interference + 1e-8)
        sinr_db = 10 * np.log10(sinr + 1e-8)
        
        results.append({'Total': total, 'W': W_pwr, 'Z': Z_pwr, 'SINR': sinr_db.min()})
    
    print(f"v47: 功率={np.mean([r['Total'] for r in results]):.2f}W")
    print(f"     W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR最小值: {np.min([r['SINR'] for r in results]):.2f}dB")
    print(f"     SINR平均值: {np.mean([r['SINR'] for r in results]):.2f}dB")
    return results

if __name__ == '__main__':
    model = train_v47()
    test_v47(model)
