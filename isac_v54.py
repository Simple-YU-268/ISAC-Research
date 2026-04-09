"""
ISAC v54 - 使用优化器生成的数据作为监督
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

def generate_optimal_data(n_samples):
    """用优化器生成监督数据"""
    X_data = []
    w_data = []
    z_data = []
    b_data = []
    
    for _ in range(n_samples):
        H = np.random.randn(M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
        
        # 优化通信波束
        def objective(w_vec):
            w = w_vec[:M*K*Nt].reshape(M, K, Nt)
            power = np.sum(np.abs(w) ** 2)
            if power > Pmax * 0.8:  # 留一些给感知
                return 1e6
            
            sinrs = []
            for k in range(K):
                signal = np.abs(np.sum(np.conj(w[:, k, :]) * H_complex[:, k, :])) ** 2
                interference = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H_complex[:, k, :])) ** 2 for j in range(K) if j != k)
                sinrs.append(signal / (interference + 0.01))
            
            min_sinr = min(sinrs)
            
            if min_sinr < 3:  # 目标SINR > 3dB
                return power + 50 * (3 - min_sinr)
            
            return power
        
        w0 = np.random.randn(M, K, Nt) * 0.05
        try:
            result = minimize(objective, w0.flatten(), method='SLSQP', 
                           options={'maxiter': 200, 'ftol': 1e-2})
            w_opt = result.x[:M*K*Nt].reshape(M, K, Nt)
        except:
            w_opt = w0
        
        # 功率归一化
        W_power = np.sum(np.abs(w_opt) ** 2)
        if W_power > 0:
            w_normalized = w_opt * np.sqrt(Pmax * 0.6 / W_power)
        else:
            w_normalized = w_opt
        
        # 感知和AP选择 (简化)
        z = np.random.rand(M, P) * 0.5
        b = np.random.rand(M, P)
        for p in range(P):
            selected = np.random.choice(M, N_req, replace=False)
            b[selected, p] = 0.9 + np.random.rand(N_req) * 0.1
        
        X_data.append(H.flatten())
        w_data.append(w_normalized.flatten())
        z_data.append(z.flatten())
        b_data.append(b.flatten())
    
    return np.array(X_data), np.array(w_data), np.array(z_data), np.array(b_data)

print("生成优化数据...")
X, w_labels, z_labels, b_labels = generate_optimal_data(200)
print(f"生成完成: {len(X)} 样本")

class ISAC_v54(nn.Module):
    def __init__(self):
        super().__init__()
        hd = M * K * Nt * 2
        
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        # 输出通信波束
        self.w_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * K * Nt), nn.Tanh())
        
        self.z_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
        self.b_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
    
    def forward(self, x):
        bs = x.shape[0]
        x_flat = x.view(bs, -1)
        emb = self.encoder(x_flat)
        
        w = self.w_head(emb).view(bs, M, K, Nt)
        z = self.z_head(emb).view(bs, M, P)
        b = self.b_head(emb).view(bs, M, P)
        
        return w, z, b

def train_v54(epochs=3000, bs=32, lr=1e-4):
    model = ISAC_v54()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    Xt = torch.tensor(X, dtype=torch.float32)
    w_labels_t = torch.tensor(w_labels, dtype=torch.float32)
    z_labels_t = torch.tensor(z_labels, dtype=torch.float32)
    b_labels_t = torch.tensor(b_labels, dtype=torch.float32)
    
    indices = torch.arange(len(X))
    
    for e in range(epochs):
        opt.zero_grad()
        
        # 随机采样
        idx = torch.randperm(len(X))[:bs]
        X_batch = Xt[idx]
        w_labels_batch = w_labels_t[idx]
        
        w, z, b = model(X_batch)
        
        # 有监督损失
        loss_w = F.mse_loss(w, w_labels_batch) * 100
        
        # 功率约束
        W_power = (w ** 2).sum(dim=(1,2,3))
        Z_power = (z * b).sum(dim=(1,2))
        
        W_pwr = W_power * 1.5
        Z_pwr = Z_power * 1.5
        total_pwr = W_pwr + Z_pwr
        
        loss_pwr = F.relu(total_pwr - Pmax).mean() * 200
        
        # AP选择
        ap_counts = b.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 50
        
        loss = loss_w + loss_pwr + loss_ap
        
        loss.backward()
        opt.step()
        
        if e % 600 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}, W_pwr={W_pwr.mean().item():.2f}W")
    
    torch.save(model.state_dict(), 'isac_v54.pth')
    print("v54完成!")
    return model

def test_v54(model, n=50):
    model.eval()
    results = []
    
    for i in range(n):
        H = np.random.randn(M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        
        w, z, b = model(torch.tensor(H.reshape(1, -1), dtype=torch.float32))
        
        w_np = w.squeeze().detach().numpy()
        
        W_pwr = np.sum(w_np ** 2)
        Z_pwr = (z * b).sum().item() * 1.5
        total = W_pwr + Z_pwr
        
        # 计算SINR
        H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
        sinrs = []
        for k in range(K):
            signal = np.abs(np.sum(np.conj(w_np[:, k, :]) * H_complex[:, k, :])) ** 2
            interference = sum(np.abs(np.sum(np.conj(w_np[:, j, :]) * H_complex[:, k, :])) ** 2 for j in range(K) if j != k)
            sinrs.append(10 * np.log10(signal / (interference + 0.01) + 1e-8))
        
        results.append({'Total': total, 'W': W_pwr, 'Z': Z_pwr, 'SINR_min': min(sinrs), 'SINR_mean': np.mean(sinrs)})
    
    print(f"v54: 功率={np.mean([r['Total'] for r in results]):.2f}W")
    print(f"     W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR最小值: {np.min([r['SINR_min'] for r in results]):.2f}dB")
    print(f"     SINR平均值: {np.mean([r['SINR_min'] for r in results]):.2f}dB")
    return results

if __name__ == '__main__':
    model = train_v54()
    test_v54(model)
