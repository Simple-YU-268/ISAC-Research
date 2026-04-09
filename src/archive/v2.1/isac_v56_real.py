"""ISAC v56 - 真实信道模型"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30
N_req = 4

def generate_real_channel(n_samples):
    X_list, p_list = [], []
    
    for _ in range(n_samples):
        ap_pos = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        user_pos = np.random.uniform(-50, 50, (K, 2))
        
        H = np.zeros((M, K, Nt*2), dtype=np.float32)
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap_pos[m] - user_pos[k])**2)), 5)
                pl = (d / 10) ** -2
                h = np.sqrt(pl/2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
                H[m, k, :Nt] = np.real(h)
                H[m, k, Nt:] = np.imag(h)
        
        # ZF
        H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
        H_stack = np.vstack([H_complex[m, :, :] for m in range(M)])
        try:
            H_pinv = np.linalg.pinv(H_stack)
            p_zf = np.sum(np.abs(H_pinv) ** 2, axis=1)
        except:
            p_zf = np.ones(K) / K * Pmax * 0.7
        
        if np.sum(p_zf) > 0:
            p_zf = p_zf * Pmax * 0.7 / np.sum(p_zf)
        
        p_assign = np.zeros((M, K))
        for k in range(K):
            p_assign[:, k] = p_zf[k] / M
        
        X_list.append(H.flatten())
        p_list.append(p_assign.flatten())
    
    return np.array(X_list), np.array(p_list)

print("生成数据...")
X, P_zf = generate_real_channel(200)

class ISAC_v56(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        self.p_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * K), nn.Sigmoid())
        self.z_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * P), nn.Sigmoid())
        self.b_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * P), nn.Sigmoid())
    
    def forward(self, x):
        emb = self.net(x)
        p = self.p_head(emb)
        z = self.z_head(emb).view(-1, M, P)
        b = self.b_head(emb).view(-1, M, P)
        return p, z, b

def train_v56(epochs=3000, bs=32, lr=1e-4):
    model = ISAC_v56()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    Xt = torch.tensor(X, dtype=torch.float32)
    Pt = torch.tensor(P_zf, dtype=torch.float32)
    
    for e in range(epochs):
        opt.zero_grad()
        idx = torch.randperm(len(X))[:bs]
        
        p, z, b = model(Xt[idx])
        
        loss_sup = F.mse_loss(p, Pt[idx]) * 30
        
        W_pwr = p.sum(dim=1) * 10
        Z_pwr = (z * b).sum(dim=(1,2)) * 10
        total_pwr = W_pwr + Z_pwr
        
        loss_pwr = F.relu(total_pwr - Pmax).mean() * 400 + F.relu(Pmax * 0.95 - total_pwr).mean() * 300
        
        ap_counts = b.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 50
        
        loss = loss_sup + loss_pwr + loss_ap
        loss.backward()
        opt.step()
        
        if e % 600 == 0:
            print(f"Epoch {e}: W={W_pwr.mean().item():.2f}W, Z={Z_pwr.mean().item():.2f}W")
    
    torch.save(model.state_dict(), 'isac_v56.pth')
    print("v56完成!")
    return model

def test_v56(model, n=50):
    model.eval()
    results = []
    
    for i in range(n):
        ap_pos = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        user_pos = np.random.uniform(-50, 50, (K, 2))
        
        H = np.zeros((M, K, Nt*2), dtype=np.float32)
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap_pos[m] - user_pos[k])**2)), 5)
                pl = (d / 10) ** -2
                h = np.sqrt(pl/2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
                H[m, k, :Nt] = np.real(h)
                H[m, k, Nt:] = np.imag(h)
        
        p, z, b = model(torch.tensor(H.flatten().reshape(1, -1), dtype=torch.float32))
        
        p_np = p.squeeze().detach().numpy()
        z_np = z.squeeze().detach().numpy()
        b_np = b.squeeze().detach().numpy()
        
        w = np.sqrt(p_np[:, None, None]) * (np.random.randn(M, K, Nt) + 1j * np.random.randn(M, K, Nt))
        w = w / np.sqrt(M * Nt)
        
        W_pwr = np.sum(p_np) * 10
        Z_pwr = np.sum(z_np * b_np) * 10
        
        H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
        sinrs = []
        for k in range(K):
            signal = np.abs(np.sum(np.conj(w[:, k, :]) * H_complex[:, k, :])) ** 2
            interference = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H_complex[:, k, :])) ** 2 for j in range(K) if j != k)
            sinrs.append(10 * np.log10(signal / (interference + 0.01) + 1e-8))
        
        results.append({'Total': W_pwr + Z_pwr, 'W': W_pwr, 'Z': Z_pwr, 'SINR_min': min(sinrs)})
    
    print(f"v56: 功率={np.mean([r['Total'] for r in results]):.2f}W, W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR_min: {np.min([r['SINR_min'] for r in results]):.2f}dB, 平均: {np.mean([r['SINR_min'] for r in results]):.2f}dB")
    print(f"     ≥ 0dB: {sum(1 for r in results if r['SINR_min'] >= 0)}/{n}")

if __name__ == '__main__':
    model = train_v56()
    test_v56(model)
