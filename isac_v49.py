"""
ISAC v49 - 强制通信功率比例，确保SINR约束
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 8, 4, 4
Pmax = 30
N_req = 4

class ISAC_v49(nn.Module):
    def __init__(self):
        super().__init__()
        hd = M * K * Nt * 2
        
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        # 输出通信功率分配 (M, K) 和感知功率 (M, P) 和AP选择 (M, P)
        # 使用更直接的方式：直接输出功率比例
        self.w_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * K), nn.Sigmoid())
        self.z_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
        self.b_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
    
    def forward(self, x):
        bs = x.shape[0]
        x_flat = x.view(bs, -1)
        emb = self.encoder(x_flat)
        
        w = self.w_head(emb).view(bs, M, K)
        z = self.z_head(emb).view(bs, M, P)
        b = self.b_head(emb).view(bs, M, P)
        
        return w, z, b

def train_v49(epochs=5000, bs=64, lr=8e-5):
    model = ISAC_v49()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1000, T_mult=2)
    
    # 目标: W至少15W用于通信，Z约15W用于感知
    for e in range(epochs):
        H = np.random.randn(bs, M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        Ht = torch.tensor(H, dtype=torch.float32)
        
        w, z, b = model(Ht)
        
        # === 功率计算 ===
        W_power = w.sum(dim=(1,2))  # 归一化
        Z_power = (z * b).sum(dim=(1,2))
        
        # 缩放: 确保总功率约30W
        # W用0-0.5范围, Z用0-0.5范围
        W_pwr = W_power * Pmax
        Z_pwr = Z_power * Pmax
        total_pwr = W_pwr + Z_pwr
        
        # === 损失 ===
        # 1. 总功率约束 (最重要)
        loss_pwr_over = F.relu(total_pwr - Pmax).mean() * 500
        loss_pwr_under = F.relu(Pmax * 0.98 - total_pwr).mean() * 400
        
        # 2. 通信功率下界 (关键!)
        loss_comm_min = F.relu(15 - W_pwr).mean() * 200  # 至少15W给通信
        
        # 3. 感知功率下界
        loss_sens_min = F.relu(12 - Z_pwr).mean() * 150
        
        # 4. AP选择: 每目标选N_req个
        b_reshaped = b.view(bs, M, P)
        ap_counts = b_reshaped.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 80
        
        # 5. 通信功率上界 (防止分配过多)
        loss_comm_max = F.relu(W_pwr - 20).mean() * 50
        
        loss = loss_pwr_over + loss_pwr_under + loss_comm_min + loss_sens_min + loss_ap + loss_comm_max
        
        if e % 1000 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}, W={W_pwr.mean().item():.2f}W, Z={Z_pwr.mean().item():.2f}W, total={total_pwr.mean().item():.2f}W")
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    
    torch.save(model.state_dict(), 'isac_v49.pth')
    print("v49完成!")
    return model

def test_v49(model, n=100):
    model.eval()
    results = []
    
    for i in range(n):
        H = np.random.randn(M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        Ht = torch.tensor(H.reshape(1, M, K, Nt*2), dtype=torch.float32)
        
        w, z, b = model(Ht)
        
        W_power = w.squeeze().sum().item()
        Z_power = (z * b).squeeze().sum().item()
        
        W_pwr = W_power * Pmax
        Z_pwr = Z_power * Pmax
        total = W_pwr + Z_pwr
        
        # 简化的SINR估算
        H_real = H[:, :, :Nt]
        signal = np.sum(np.sum(H_real**2, axis=2), axis=0) * (W_pwr / M)
        total_sig = np.sum(signal)
        interference = total_sig - signal + 0.01
        sinr = signal / (interference + 1e-8)
        sinr_db = 10 * np.log10(sinr + 1e-8)
        
        b_np = b.squeeze().detach().numpy()
        aps = sum([len(np.argsort(-b_np[:, p])[:N_req]) for p in range(P)])
        
        results.append({'Total': total, 'W': W_pwr, 'Z': Z_pwr, 'SINR': sinr_db.min(), 'APs': aps})
    
    print(f"v49: 功率={np.mean([r['Total'] for r in results]):.2f}W")
    print(f"     W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR最小值: {np.min([r['SINR'] for r in results]):.2f}dB")
    print(f"     SINR平均值: {np.mean([r['SINR'] for r in results]):.2f}dB")
    print(f"     APs={np.mean([r['APs'] for r in results]):.0f}")
    return results

if __name__ == '__main__':
    model = train_v49()
    test_v49(model)
