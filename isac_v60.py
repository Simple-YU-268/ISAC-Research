"""
ISAC v60 - 使用MMSE预编码作为监督
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30
N_req = 4

def generate_mmse_data(n_samples):
    X_list, power_list, z_list, b_list = [], [], [], []
    
    for _ in range(n_samples):
        ap_pos = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        user_pos = np.random.uniform(-50, 50, (K, 2))
        
        H = np.zeros((M, K, Nt*2), dtype=np.float32)
        H_complex = np.zeros((M, K, Nt), dtype=complex)
        
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap_pos[m] - user_pos[k])**2)), 5)
                pl = (d / 10) ** -2
                h = np.sqrt(pl/2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
                H[m, k, :Nt] = np.real(h)
                H[m, k, Nt:] = np.imag(h)
                H_complex[m, k, :] = h
        
        # MMSE预编码
        H_stack = np.vstack([H_complex[m, :, :] for m in range(M)])
        sigma2 = 0.1  # 适当噪声
        
        HH = H_stack @ H_stack.conj().T
        HH_reg = HH + sigma2 * np.eye(M * Nt)
        
        try:
            HH_inv = np.linalg.inv(HH_reg)
            W_mmse = HH_inv @ H_stack
            
            power = np.sum(np.abs(W_mmse) ** 2)
            W_mmse = W_mmse * np.sqrt(Pmax * 0.6 / power)
            
            p_mmse = np.zeros(M * K)
            for k in range(K):
                for m in range(M):
                    p_mmse[m + k*M] = np.sum(np.abs(W_mmse[m*Nt:(m+1)*Nt, k]) ** 2)
        except:
            p_mmse = np.ones(M * K) / (M * K) * Pmax * 0.6
        
        # 感知和AP选择
        target_pos = np.random.uniform(-30, 30, (P, 2))
        z = np.zeros(M * P)
        b = np.zeros(M * P)
        
        for p in range(P):
            for m in range(M):
                d = np.sqrt(np.sum((ap_pos[m] - target_pos[p])**2))
                z[m + p*M] = 1.0 / (d + 1)
            
            selected = np.random.choice(M, N_req, replace=False)
            for m in selected:
                b[m + p*M] = 1.0
        
        X_list.append(H.flatten())
        power_list.append(p_mmse)
    
    return np.array(X_list), np.array(power_list)

print("生成MMSE数据...")
X, P_mmse = generate_mmse_data(200)

class ISAC_v60(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        self.p_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * K), nn.Sigmoid())
        self.z_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
        self.b_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
    
    def forward(self, x):
        emb = self.net(x)
        p = self.p_head(emb)
        z = self.z_head(emb).view(-1, M, P)
        b = self.b_head(emb).view(-1, M, P)
        return p, z, b

def train_v60(epochs=3000, bs=32, lr=1e-4):
    model = ISAC_v60()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    Xt = torch.tensor(X, dtype=torch.float32)
    Pt = torch.tensor(P_mmse, dtype=torch.float32)
    
    for e in range(epochs):
        opt.zero_grad()
        idx = torch.randperm(len(X))[:bs]
        
        p, z, b = model(Xt[idx])
        
        loss_sup = F.mse_loss(p, Pt[idx]) * 50
        
        W_pwr = p.sum(dim=1) * 3
        Z_pwr = (z * b).sum(dim=(1,2)) * 3
        total_pwr = W_pwr + Z_pwr
        
        loss_pwr = F.relu(total_pwr - Pmax).mean() * 400 + F.relu(Pmax * 0.98 - total_pwr).mean() * 300
        
        ap_counts = b.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 50
        
        loss = loss_sup + loss_pwr + loss_ap
        loss.backward()
        opt.step()
        
        if e % 600 == 0:
            print(f"Epoch {e}: W={W_pwr.mean().item():.2f}W, Z={Z_pwr.mean().item():.2f}W")
    
    torch.save(model.state_dict(), 'isac_v60.pth')
    print("v60完成!")
    return model

def test_v60(model, n=50):
    model.eval()
    results = []
    
    for i in range(n):
        ap_pos = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        user_pos = np.random.uniform(-50, 50, (K, 2))
        
        H = np.zeros((M, K, Nt*2), dtype=np.float32)
        H_complex = np.zeros((M, K, Nt), dtype=complex)
        
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap_pos[m] - user_pos[k])**2)), 5)
                pl = (d / 10) ** -2
                h = np.sqrt(pl/2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
                H[m, k, :Nt] = np.real(h)
                H[m, k, Nt:] = np.imag(h)
                H_complex[m, k, :] = h
        
        p, z, b = model(torch.tensor(H.flatten().reshape(1, -1), dtype=torch.float32))
        
        p_np = p.squeeze().detach().numpy()
        
        # 使用MMSE相位
        H_stack = np.vstack([H_complex[m, :, :] for m in range(M)])
        sigma2 = 0.1
        
        HH = H_stack @ H_stack.conj().T
        HH_reg = HH + sigma2 * np.eye(M * Nt)
        
        try:
            HH_inv = np.linalg.inv(HH_reg)
            W_mmse = HH_inv @ H_stack
            power = np.sum(np.abs(W_mmse) ** 2)
            W_mmse = W_mmse * np.sqrt(Pmax * 0.6 / power)
            
            w = np.zeros((M, K, Nt), dtype=complex)
            for k in range(K):
                for m in range(M):
                    w[m, k, :] = W_mmse[m*Nt:(m+1)*Nt, k]
        except:
            w = np.zeros((M, K, Nt), dtype=complex)
        
        W_pwr = np.sum(p_np) * 3
        Z_pwr = (z * b).sum().item() * 3
        
        sinrs = []
        for k in range(K):
            signal = np.abs(np.sum(np.conj(w[:, k, :]) * H_complex[:, k, :])) ** 2
            interference = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H_complex[:, k, :])) ** 2 for j in range(K) if j != k)
            sinrs.append(10 * np.log10(signal / (interference + 0.01) + 1e-8))
        
        results.append({'Total': W_pwr + Z_pwr, 'W': W_pwr, 'Z': Z_pwr, 'SINR_min': min(sinrs)})
    
    print(f"v60: 功率={np.mean([r['Total'] for r in results]):.2f}W")
    print(f"     W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR_min: {np.min([r['SINR_min'] for r in results]):.2f}dB, 平均: {np.mean([r['SINR_min'] for r in results]):.2f}dB")
    print(f"     ≥ 0dB: {sum(1 for r in results if r['SINR_min'] >= 0)}/{n}")

if __name__ == '__main__':
    model = train_v60()
    test_v60(model)
