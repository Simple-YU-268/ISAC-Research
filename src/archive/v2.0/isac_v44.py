"""
ISAC v44 - 学习实际波束成形向量 (更真实的模型)
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

class ISAC_v44(nn.Module):
    def __init__(self):
        super().__init__()
        hd = cfg.M * cfg.K * cfg.Nt * 2
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        # 通信波束成形输出 (M, K, Nt) - 每个AP-用户对的波束
        self.W_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, cfg.M * cfg.K * cfg.Nt), nn.Sigmoid())
        
        # 感知波束成形输出 (M, P) - AP选择权重
        self.Z_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, cfg.M * cfg.P), nn.Sigmoid())
    
    def forward(self, x):
        emb = self.encoder(x)
        
        # 通信波束 (M, K, Nt)
        W = self.W_head(emb).view(-1, cfg.M, cfg.K, cfg.Nt)
        
        # 感知权重 (M, P)
        Z = self.Z_head(emb).view(-1, cfg.M, cfg.P)
        
        return W, Z

def compute_real_sinr(H, W):
    """计算实际通信SINR (简化版)"""
    # H: (M, K, Nt), W: (M, K, Nt)
    # 信号: |w_k^H h_k|^2
    # 干扰: sum_{j!=k} |w_j^H h_k|^2
    
    M, K, Nt = H.shape
    
    sinrs = []
    for k in range(K):
        h_k = H[:, k, :]  # (M, Nt)
        w_k = W[:, k, :]  # (M, Nt)
        
        # 信号功率
        signal = np.abs(np.sum(w_k * h_k, axis=1))**2  # (M,)
        signal_total = np.sum(signal)
        
        # 干扰功率
        interference = 0
        for j in range(K):
            if j != k:
                w_j = W[:, j, :]
                interference += np.sum(np.abs(np.sum(w_j * h_k, axis=1))**2)
        
        if interference > 0:
            sinr = signal_total / (interference + 0.001)
        else:
            sinr = signal_total / 0.001
        sinrs.append(sinr)
    
    return np.array(sinrs)

def train_v44(epochs=4000, bs=64, lr=1.5e-4):
    model = ISAC_v44()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=800, T_mult=2)
    
    for e in range(epochs):
        H = generate_data(bs)
        Xt = torch.tensor(H.reshape(bs, -1), dtype=torch.float32)
        
        W, Z = model(Xt)
        
        # 通信功率约束
        W_power = W.sum(dim=(1,2,3)) * 1.0  # 归一化后乘Pmax
        
        # 感知功率约束
        Z_power = Z.sum(dim=(1,2)) * 1.0
        
        total_power = (W_power + Z_power * 0.9) * cfg.Pmax / 2
        loss_pwr = F.relu(total_power - cfg.Pmax).mean() * 200 + F.relu(cfg.Pmax - total_power).mean() * 150
        
        # AP选择: 每个目标选N_req个AP
        Z_reshaped = Z.view(bs, cfg.M, cfg.P)
        loss_ap = 0
        for p in range(cfg.P):
            top_N = torch.topk(Z_reshaped[:, :, p], cfg.N_req, dim=1).values
            loss_ap += (top_N.mean() - 0.8).abs()  # 鼓励选择高权重AP
        
        loss = loss_pwr + loss_ap * 10
        
        if e % 800 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}, W_pwr={W_power.mean().item():.2f}, Z_pwr={Z_power.mean().item():.2f}")
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    
    torch.save(model.state_dict(), 'isac_v44.pth')
    print("v44完成!")
    return model

def test_v44(model, n=50):
    model.eval()
    results = []
    
    for i in range(n):
        H = np.random.randn(cfg.M, cfg.K, cfg.Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        
        Xt = torch.tensor(H.reshape(1, -1), dtype=torch.float32)
        W, Z = model(Xt)
        
        W_np = W.squeeze().detach().numpy()
        Z_np = Z.squeeze().detach().numpy()
        
        W_pwr = np.sum(W_np) * 1.0
        Z_pwr = np.sum(Z_np) * 0.9
        total = (W_pwr + Z_pwr) * cfg.Pmax / 2
        
        # 计算SINR
        H_real = H[:, :, :cfg.Nt]  # 取实部作为信道
        sinrs = compute_real_sinr(H_real, W_np * 0.1)  # 缩放波束
        sinr_db = 10 * np.log10(sinrs + 1e-8)
        
        # AP选择
        aps = sum([len(np.argsort(-Z_np[:,p])[:cfg.N_req]) for p in range(cfg.P)])
        
        results.append({'Total': total, 'W': W_pwr*cfg.Pmax/2, 'Z': Z_pwr*cfg.Pmax/2, 'sinr_min': sinr_db.min(), 'APs': aps})
    
    print(f"v44: 功率={np.mean([r['Total'] for r in results]):.2f}W, W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR最小值: {np.min([r['sinr_min'] for r in results]):.2f}dB, APs={np.mean([r['APs'] for r in results]):.0f}")
    return results

if __name__ == '__main__':
    model = train_v44()
    test_v44(model)
