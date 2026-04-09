"""
ISAC v51 - 简化场景: 减少用户数+强制信道匹配
核心: 让部分用户使用正交资源减少干扰
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 简化场景
M, K, P, Nt = 16, 4, 4, 4  # 减少用户数到4
Pmax = 30
N_req = 4

class ISAC_v51(nn.Module):
    def __init__(self):
        super().__init__()
        hd = M * K * Nt * 2
        
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        # 输出通信权重 (M, K) 和感知权重
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

def compute_sinr_simple(H, w):
    """简化的SINR计算"""
    H_real = H[:, :, :Nt]
    sinrs = []
    for k in range(K):
        # 信号: 用户k从所有AP收到的信号
        signal = 0
        for m in range(M):
            w_mk = w[m, k] * 0.3  # 缩放
            h_mk = H_real[m, k, :]
            signal += (w_mk ** 2) * np.sum(h_mk ** 2)
        
        # 干扰: 其他用户的干扰
        interference = 0
        for j in range(K):
            if j != k:
                for m in range(M):
                    w_mj = w[m, j] * 0.3
                    h_mk = H_real[m, k, :]
                    interference += (w_mj ** 2) * np.sum(h_mk ** 2)
        
        sinr = signal / (interference + 0.1)
        sinrs.append(10 * np.log10(sinr + 1e-8))
    return np.array(sinrs)

def train_v51(epochs=6000, bs=64, lr=8e-5):
    model = ISAC_v51()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    for e in range(epochs):
        model.zero_grad()
        
        H = np.random.randn(bs, M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        Ht = torch.tensor(H, dtype=torch.float32)
        
        w, z, b = model(Ht)
        
        # 功率计算
        W_power = w.sum(dim=(1,2))
        Z_power = (z * b).sum(dim=(1,2))
        
        W_pwr = W_power * 10
        Z_pwr = Z_power * 10
        total_pwr = W_pwr + Z_pwr
        
        # 损失
        loss_pwr = F.relu(total_pwr - Pmax).mean() * 400 + F.relu(Pmax * 0.97 - total_pwr).mean() * 300
        
        b_reshaped = b.view(bs, M, P)
        ap_counts = b_reshaped.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 80
        
        # 通信功率约束
        loss_comm = F.relu(18 - W_pwr).mean() * 200 + F.relu(W_pwr - 25).mean() * 50
        
        # SINR约束
        sinr_np = compute_sinr_simple(H, w.detach().numpy())
        sinr_tensor = torch.tensor(sinr_np, dtype=torch.float32)
        loss_sinr = F.relu(8 - sinr_tensor).mean() * 150
        
        loss = loss_pwr + loss_ap + loss_comm + loss_sinr
        
        loss.backward()
        opt.step()
        
        if e % 1200 == 0:
            print(f"Epoch {e}: W={W_pwr.mean().item():.2f}W, Z={Z_pwr.mean().item():.2f}W, SINR={sinr_tensor.mean().item():.2f}dB")
    
    torch.save(model.state_dict(), 'isac_v51.pth')
    print("v51完成!")
    return model

def test_v51(model, n=100):
    model.eval()
    results = []
    
    for i in range(n):
        H = np.random.randn(M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        
        w, z, b = model(torch.tensor(H.reshape(1, M, K, Nt*2), dtype=torch.float32))
        
        W_pwr = w.sum().item() * 10
        Z_pwr = (z * b).sum().item() * 10
        total = W_pwr + Z_pwr
        
        sinr_np = compute_sinr_simple(H, w.squeeze().detach().numpy())
        
        results.append({'Total': total, 'W': W_pwr, 'Z': Z_pwr, 'SINR_min': sinr_np.min()})
    
    print(f"v51: 功率={np.mean([r['Total'] for r in results]):.2f}W, W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR最小值: {np.min([r['SINR_min'] for r in results]):.2f}dB, 平均: {np.mean([r['SINR_min'] for r in results]):.2f}dB")
    return results

if __name__ == '__main__':
    model = train_v51()
    test_v51(model)
