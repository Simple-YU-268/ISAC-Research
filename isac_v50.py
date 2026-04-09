"""
ISAC v50 - 完整波束学习
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 8, 4, 4
Pmax = 30
N_req = 4

class ISAC_v50(nn.Module):
    def __init__(self):
        super().__init__()
        hd = M * K * Nt * 2
        
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        self.w_mag_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * K * Nt), nn.Sigmoid())
        self.w_phase_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * K * Nt), nn.Tanh())
        self.z_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
        self.b_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
    
    def forward(self, x):
        bs = x.shape[0]
        x_flat = x.view(bs, -1)
        emb = self.encoder(x_flat)
        
        w_mag = self.w_mag_head(emb).view(bs, M, K, Nt) * 0.4
        w_phase = self.w_phase_head(emb).view(bs, M, K, Nt)
        z = self.z_head(emb).view(bs, M, P)
        b = self.b_head(emb).view(bs, M, P)
        
        return w_mag, w_phase, z, b

def compute_sinr_np(H, w_mag, w_phase):
    """numpy版本SINR计算"""
    bs = w_mag.shape[0]
    all_sinrs = []
    
    for i in range(bs):
        sinrs = []
        for k in range(K):
            w = w_mag[i, :, k, :] * np.exp(1j * w_phase[i, :, k, :] * np.pi)
            h = H[i, :, k, :Nt] + 1j * H[i, :, k, Nt:]
            
            signal = np.abs(np.sum(np.conj(w) * h)) ** 2
            interference = 0
            for j in range(K):
                if j != k:
                    w_j = w_mag[i, :, j, :] * np.exp(1j * w_phase[i, :, j, :] * np.pi)
                    interference += np.abs(np.sum(np.conj(w_j) * h)) ** 2
            
            sinrs.append(10 * np.log10(signal / (interference + 0.001) + 1e-8))
        all_sinrs.append(sinrs)
    
    return np.array(all_sinrs)

def train_v50(epochs=5000, bs=64, lr=3e-5):
    model = ISAC_v50()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    for e in range(epochs):
        model.zero_grad()
        
        H = np.random.randn(bs, M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        Ht = torch.tensor(H, dtype=torch.float32)
        
        w_mag, w_phase, z, b = model(Ht)
        
        # 功率计算
        W_power = (w_mag ** 2).sum(dim=(1,2,3))
        Z_power = (z * b).sum(dim=(1,2))
        
        W_pwr = W_power * 20
        Z_pwr = Z_power * 15
        total_pwr = W_pwr + Z_pwr
        
        # 损失
        loss_pwr = F.relu(total_pwr - Pmax).mean() * 300 + F.relu(Pmax * 0.95 - total_pwr).mean() * 200
        
        b_reshaped = b.view(bs, M, P)
        ap_counts = b_reshaped.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 80
        
        loss_comm = F.relu(15 - W_pwr).mean() * 150
        loss_sens = F.relu(10 - Z_pwr).mean() * 100
        
        # SINR损失 (detach以避免numpy转换问题)
        sinr_np = compute_sinr_np(H, w_mag.detach().numpy(), w_phase.detach().numpy())
        sinr_tensor = torch.tensor(sinr_np, dtype=torch.float32)
        loss_sinr = F.relu(5 - sinr_tensor).mean() * 80
        
        loss = loss_pwr + loss_ap + loss_comm + loss_sens + loss_sinr
        
        loss.backward()
        opt.step()
        
        if e % 1000 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}, W={W_pwr.mean().item():.2f}W, Z={Z_pwr.mean().item():.2f}W, SINR={sinr_tensor.mean().item():.2f}dB")
    
    torch.save(model.state_dict(), 'isac_v50.pth')
    print("v50完成!")
    return model

def test_v50(model, n=50):
    model.eval()
    results = []
    
    for i in range(n):
        H = np.random.randn(M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        Ht = torch.tensor(H.reshape(1, M, K, Nt*2), dtype=torch.float32)
        
        w_mag, w_phase, z, b = model(Ht)
        
        w_mag_np = w_mag.squeeze().detach().numpy()
        w_phase_np = w_phase.squeeze().detach().numpy()
        
        W_power = np.sum(w_mag_np ** 2)
        Z_power = (z * b).sum().item()
        
        W_pwr = W_power * 20
        Z_pwr = Z_power * 15
        total = W_pwr + Z_pwr
        
        sinr_np = compute_sinr_np(H.reshape(1, M, K, Nt*2), w_mag_np.reshape(1, M, K, Nt), w_phase_np.reshape(1, M, K, Nt))
        
        b_np = b.squeeze().detach().numpy()
        aps = sum([len(np.argsort(-b_np[:, p])[:N_req]) for p in range(P)])
        
        results.append({'Total': total, 'W': W_pwr, 'Z': Z_pwr, 'SINR_min': sinr_np.min(), 'APs': aps})
    
    print(f"v50: 功率={np.mean([r['Total'] for r in results]):.2f}W")
    print(f"     W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR最小值: {np.min([r['SINR_min'] for r in results]):.2f}dB")
    print(f"     SINR平均值: {np.mean([r['SINR_min'] for r in results]):.2f}dB")
    return results

if __name__ == '__main__':
    model = train_v50()
    test_v50(model)
