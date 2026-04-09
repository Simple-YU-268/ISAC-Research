"""ISAC v63 - 测试不同正则化参数"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30
N_req = 4

def generate_data(n_samples, sigma2):
    X_list, power_list = [], []
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
        
        H_stack = np.vstack([H_complex[m, :, :] for m in range(M)])
        HH = H_stack @ H_stack.conj().T
        HH_reg = HH + sigma2 * np.eye(M * Nt)
        
        try:
            HH_inv = np.linalg.inv(HH_reg)
            W_mmse = HH_inv @ H_stack
            power = np.sum(np.abs(W_mmse) ** 2)
            W_mmse = W_mmse * np.sqrt(Pmax * 0.65 / power)
            
            p_mmse = np.zeros(M * K)
            for k in range(K):
                for m in range(M):
                    p_mmse[m + k*M] = np.sum(np.abs(W_mmse[m*Nt:(m+1)*Nt, k]) ** 2)
        except:
            p_mmse = np.ones(M * K) / (M * K) * Pmax * 0.65
        
        X_list.append(H.flatten())
        power_list.append(p_mmse)
    
    return np.array(X_list), np.array(power_list)

class ISAC_v63(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32)
        )
        self.p_head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, M * K), nn.Sigmoid())
        self.z_head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
        self.b_head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
    
    def forward(self, x):
        emb = self.net(x)
        p = self.p_head(emb)
        z = self.z_head(emb).view(-1, M, P)
        b = self.b_head(emb).view(-1, M, P)
        return p, z, b

best_sigma2, best_positive = None, 0

for sigma2 in [0.3, 0.5, 0.7, 1.0]:
    print(f"\n=== sigma2={sigma2} ===")
    X, P_mmse = generate_data(300, sigma2)
    
    model = ISAC_v63()
    opt = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=2e-5)
    
    Xt = torch.tensor(X, dtype=torch.float32)
    Pt = torch.tensor(P_mmse, dtype=torch.float32)
    
    for e in range(3000):
        opt.zero_grad()
        idx = torch.randperm(len(X))[:64]
        p, z, b = model(Xt[idx])
        
        loss = F.mse_loss(p, Pt[idx]) * 30
        W_pwr = p.sum(dim=1) * 2.5
        Z_pwr = (z * b).sum(dim=(1,2)) * 2.5
        total_pwr = W_pwr + Z_pwr
        
        loss += F.relu(total_pwr - Pmax).mean() * 600 + F.relu(Pmax * 0.98 - total_pwr).mean() * 500
        
        ap_counts = b.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(64, -1)
        loss += F.mse_loss(ap_counts, target_counts) * 40
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    
    # 测试 (eval模式)
    model.eval()
    results = []
    for _ in range(50):
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
        
        H_stack = np.vstack([H_complex[m, :, :] for m in range(M)])
        HH = H_stack @ H_stack.conj().T
        HH_reg = HH + sigma2 * np.eye(M * Nt)
        
        try:
            HH_inv = np.linalg.inv(HH_reg)
            W_mmse = HH_inv @ H_stack
            power = np.sum(np.abs(W_mmse) ** 2)
            W_mmse = W_mmse * np.sqrt(Pmax * 0.65 / power)
            
            w = np.zeros((M, K, Nt), dtype=complex)
            for k in range(K):
                for m in range(M):
                    w[m, k, :] = W_mmse[m*Nt:(m+1)*Nt, k]
        except:
            w = np.zeros((M, K, Nt), dtype=complex)
        
        sinrs = []
        for k in range(K):
            signal = np.abs(np.sum(np.conj(w[:, k, :]) * H_complex[:, k, :])) ** 2
            interference = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H_complex[:, k, :])) ** 2 for j in range(K) if j != k)
            sinrs.append(10 * np.log10(signal / (interference + 0.01) + 1e-8))
        
        results.append(min(sinrs))
    
    positive = sum(1 for r in results if r >= 0)
    print(f"sigma2={sigma2}: ≥0dB: {positive}/50, 平均: {np.mean(results):.2f}dB")
    
    if positive > best_positive:
        best_positive = positive
        best_sigma2 = sigma2
        torch.save(model.state_dict(), 'isac_v63.pth')

print(f"\n最佳sigma2={best_sigma2}, 正SINR率={best_positive}/50")
