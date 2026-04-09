"""
ISAC v53 - 预编码设计: 学习预编码矩阵来消除用户间干扰
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30
N_req = 4

class ISAC_v53(nn.Module):
    def __init__(self):
        super().__init__()
        hd = M * K * Nt * 2
        
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        # 输出预编码矩阵 (M, K, Nt)
        self.precoder_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * K * Nt), nn.Tanh())
        self.power_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * K), nn.Sigmoid())
        
        self.z_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
        self.b_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
    
    def forward(self, x):
        bs = x.shape[0]
        x_flat = x.view(bs, -1)
        emb = self.encoder(x_flat)
        
        # 预编码矩阵
        precoder = self.precoder_head(emb).view(bs, M, K, Nt)
        power = self.power_head(emb).view(bs, M, K)
        
        # 组合成最终波束
        w = precoder * power.unsqueeze(-1)
        
        z = self.z_head(emb).view(bs, M, P)
        b = self.b_head(emb).view(bs, M, P)
        
        return w, z, b

def compute_sinr_precoded(H, w):
    """预编码波束的SINR"""
    H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
    sinrs = []
    
    for k in range(K):
        # 信号: 使用预编码后的波束
        signal = np.abs(np.sum(np.conj(w[:, k, :]) * H_complex[:, k, :])) ** 2
        
        # 干扰
        interference = 0
        for j in range(K):
            if j != k:
                interference += np.abs(np.sum(np.conj(w[:, j, :]) * H_complex[:, k, :])) ** 2
        
        sinr = signal / (interference + 0.01)
        sinrs.append(10 * np.log10(sinr + 1e-8))
    
    return np.array(sinrs)

def train_v53(epochs=6000, bs=64, lr=5e-5):
    model = ISAC_v53()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    for e in range(epochs):
        model.zero_grad()
        
        H = np.random.randn(bs, M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        Ht = torch.tensor(H, dtype=torch.float32)
        
        w, z, b = model(Ht)
        
        # 功率约束
        W_power = (w ** 2).sum(dim=(1,2,3))
        Z_power = (z * b).sum(dim=(1,2))
        
        W_pwr = W_power * 2
        Z_pwr = Z_power * 2
        total_pwr = W_pwr + Z_pwr
        
        # 损失
        loss_pwr = F.relu(total_pwr - Pmax).mean() * 500 + F.relu(Pmax * 0.95 - total_pwr).mean() * 400
        
        # AP选择
        b_reshaped = b.view(bs, M, P)
        ap_counts = b_reshaped.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 100
        
        # 通信功率
        loss_comm = F.relu(20 - W_pwr).mean() * 200
        
        # 计算SINR
        w_np = w.detach().numpy()
        sinr_list = []
        for i in range(bs):
            sinr = compute_sinr_precoded(H[i], w_np[i])
            sinr_list.append(sinr.min())
        sinr_tensor = torch.tensor(sinr_list, dtype=torch.float32)
        
        loss_sinr = F.relu(0 - sinr_tensor).mean() * 100
        
        loss = loss_pwr + loss_ap + loss_comm + loss_sinr
        
        loss.backward()
        opt.step()
        
        if e % 1200 == 0:
            print(f"Epoch {e}: W={W_pwr.mean().item():.2f}W, Z={Z_pwr.mean().item():.2f}W, total={total_pwr.mean().item():.2f}W, SINR={sinr_tensor.mean().item():.2f}dB")
    
    torch.save(model.state_dict(), 'isac_v53.pth')
    print("v53完成!")
    return model

def test_v53(model, n=50):
    model.eval()
    results = []
    
    for i in range(n):
        H = np.random.randn(M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        
        w, z, b = model(torch.tensor(H.reshape(1, M, K, Nt*2), dtype=torch.float32))
        
        w_np = w.squeeze().detach().numpy()
        
        W_pwr = np.sum(w_np ** 2)
        Z_pwr = (z * b).sum().item() * 2
        total = W_pwr + Z_pwr
        
        sinr_np = compute_sinr_precoded(H, w_np)
        
        results.append({'Total': total, 'W': W_pwr, 'Z': Z_pwr, 'SINR_min': sinr_np.min()})
    
    print(f"v53: 功率={np.mean([r['Total'] for r in results]):.2f}W")
    print(f"     W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR最小值: {np.min([r['SINR_min'] for r in results]):.2f}dB")
    return results

if __name__ == '__main__':
    model = train_v53()
    test_v53(model)
