"""
ISAC v55 - 优化器监督学习
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30
N_req = 4

def generate_data(n_samples):
    X_list, p_list = [], []
    
    for _ in range(n_samples):
        H = np.random.randn(M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
        
        def objective(p_vec):
            p = p_vec.reshape(M, K)
            power = np.sum(p)
            if power > Pmax * 0.7:
                return 1e6
            
            w = np.sqrt(p[:, :, None]) * (np.random.randn(M, K, Nt) + 1j * np.random.randn(M, K, Nt))
            w = w / np.sqrt(M * Nt)
            
            sinrs = []
            for k in range(K):
                signal = np.abs(np.sum(np.conj(w[:, k, :]) * H_complex[:, k, :])) ** 2
                interference = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H_complex[:, k, :])) ** 2 for j in range(K) if j != k)
                sinrs.append(signal / (interference + 0.01))
            
            min_sinr = min(sinrs)
            if min_sinr < 3:
                return power + 30 * (3 - min_sinr)
            return power
        
        p0 = np.random.rand(M, K) * 2
        try:
            result = minimize(objective, p0.flatten(), method='SLSQP', options={'maxiter': 200, 'ftol': 1e-2})
            p_opt = result.x.reshape(M, K)
        except:
            p_opt = p0
        
        if np.sum(p_opt) > 0:
            p_opt = p_opt * Pmax * 0.6 / np.sum(p_opt)
        
        X_list.append(H.flatten())
        p_list.append(p_opt.flatten())
    
    return np.array(X_list), np.array(p_list)

print("生成数据...")
X, P_w = generate_data(150)
print(f"完成: {len(X)} 样本")

class ISAC_v55(nn.Module):
    def __init__(self):
        super().__init__()
        hd = M * K * Nt * 2
        
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        self.p_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * K), nn.Sigmoid())
        self.z_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
        self.b_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
    
    def forward(self, x):
        bs = x.shape[0]
        x_flat = x.view(bs, -1)
        emb = self.encoder(x_flat)
        
        p = self.p_head(emb)
        z = self.z_head(emb).view(bs, M, P)
        b = self.b_head(emb).view(bs, M, P)
        
        return p, z, b

def train_v55(epochs=3000, bs=32, lr=1e-4):
    model = ISAC_v55()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    Xt = torch.tensor(X, dtype=torch.float32)
    P_t = torch.tensor(P_w, dtype=torch.float32)
    
    for e in range(epochs):
        opt.zero_grad()
        
        idx = torch.randperm(len(X))[:bs]
        X_batch = Xt[idx]
        P_batch = P_t[idx]
        
        p, z, b = model(X_batch)
        
        loss_sup = F.mse_loss(p, P_batch) * 50
        
        # 功率计算 (正确维度)
        W_pwr = p.sum(dim=1) * 10
        Z_pwr = (z * b).sum(dim=(1,2)) * 10
        total_pwr = W_pwr + Z_pwr
        
        loss_pwr = F.relu(total_pwr - Pmax).mean() * 300 + F.relu(Pmax * 0.95 - total_pwr).mean() * 200
        
        ap_counts = b.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 50
        
        loss = loss_sup + loss_pwr + loss_ap
        
        loss.backward()
        opt.step()
        
        if e % 600 == 0:
            print(f"Epoch {e}: W={W_pwr.mean().item():.2f}W, Z={Z_pwr.mean().item():.2f}W")
    
    torch.save(model.state_dict(), 'isac_v55.pth')
    print("v55完成!")
    return model

def test_v55(model, n=50):
    model.eval()
    results = []
    
    for i in range(n):
        H = np.random.randn(M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        
        p, z, b = model(torch.tensor(H.reshape(1, -1), dtype=torch.float32))
        
        p_np = p.squeeze().detach().numpy()
        
        w = np.sqrt(p_np[:, None, None]) * (np.random.randn(M, K, Nt) + 1j * np.random.randn(M, K, Nt))
        w = w / np.sqrt(M * Nt)
        
        W_pwr = np.sum(p_np) * 10
        Z_pwr = (z * b).sum().item() * 10
        total = W_pwr + Z_pwr
        
        H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
        sinrs = []
        for k in range(K):
            signal = np.abs(np.sum(np.conj(w[:, k, :]) * H_complex[:, k, :])) ** 2
            interference = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H_complex[:, k, :])) ** 2 for j in range(K) if j != k)
            sinrs.append(10 * np.log10(signal / (interference + 0.01) + 1e-8))
        
        results.append({'Total': total, 'W': W_pwr, 'Z': Z_pwr, 'SINR_min': min(sinrs)})
    
    print(f"v55: 功率={np.mean([r['Total'] for r in results]):.2f}W, W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR最小值: {np.min([r['SINR_min'] for r in results]):.2f}dB, 平均: {np.mean([r['SINR_min'] for r in results]):.2f}dB")
    print(f"     ≥ 0dB: {sum(1 for r in results if r['SINR_min'] >= 0)}/{n}")
    return results

if __name__ == '__main__':
    model = train_v55()
    test_v55(model)
