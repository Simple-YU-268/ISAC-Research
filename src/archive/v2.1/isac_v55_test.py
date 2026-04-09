"""Test v55"""
import torch
import numpy as np
import torch.nn as nn

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30

class ISAC_v55(nn.Module):
    def __init__(self):
        super().__init__()
        hd = M * K * Nt * 2
        self.encoder = nn.Sequential(nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128), nn.Linear(128, 64))
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

model = ISAC_v55()
model.load_state_dict(torch.load('isac_v55.pth'))
model.eval()

results = []
for i in range(50):
    H = np.random.randn(M, K, Nt*2).astype(np.float32)
    H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
    
    # Fix: keep batch dimension
    Xt = torch.tensor(H.reshape(1, -1), dtype=torch.float32)
    p, z, b = model(Xt)
    
    # Fix: don't squeeze, just reshape
    p_np = p.squeeze().detach().numpy()  # (64,)
    p_reshaped = p_np.reshape(M, K)  # (16, 4)
    
    # Build beam
    w = np.sqrt(p_reshaped[:, :, None]) * (np.random.randn(M, K, Nt) + 1j * np.random.randn(M, K, Nt))
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

print(f"v55: 功率={np.mean([r['Total'] for r in results]):.2f}W")
print(f"     W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
print(f"     SINR最小值: {np.min([r['SINR_min'] for r in results]):.2f}dB")
print(f"     SINR平均值: {np.mean([r['SINR_min'] for r in results]):.2f}dB")
print(f"     ≥ 0dB: {sum(1 for r in results if r['SINR_min'] >= 0)}/50")
