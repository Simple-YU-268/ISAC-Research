"""
ISAC v48 - 完整波束学习版本
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 8, 4, 4
Pmax = 30
N_req = 4

class ISAC_v48(nn.Module):
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
        
        w_mag = self.w_mag_head(emb).view(bs, M, K, Nt)
        w_phase = self.w_phase_head(emb).view(bs, M, K, Nt)
        z = self.z_head(emb).view(bs, M, P)
        b = self.b_head(emb).view(bs, M, P)
        
        return w_mag, w_phase, z, b

def compute_beam_sinr_batch(H, w_mag, w_phase):
    """批量计算SINR"""
    bs = w_mag.shape[0]
    sinr_db_all = []
    
    for b in range(bs):
        sinr_db_list = []
        for k in range(K):
            # 简化的信号和干扰计算
            # 信号: sum_m |w_mk^H h_mk|^2
            signal = 0
            for m in range(M):
                w_vec = w_mag[b, m, k, :] * torch.exp(1j * w_phase[b, m, k, :] * np.pi)
                h_vec = H[b, m, k, :Nt].float() + 1j * H[b, m, k, Nt:].float()
                signal += torch.abs(torch.sum(torch.conj(w_vec) * h_vec)) ** 2
            
            # 干扰
            interference = 0
            for j in range(K):
                if j != k:
                    for m in range(M):
                        w_vec = w_mag[b, m, j, :] * torch.exp(1j * w_phase[b, m, j, :] * np.pi)
                        h_vec = H[b, m, k, :Nt].float() + 1j * H[b, m, k, Nt:].float()
                        interference += torch.abs(torch.sum(torch.conj(w_vec) * h_vec)) ** 2
            
            sinr = signal / (interference + 0.001)
            sinr_db_list.append(10 * torch.log10(sinr + 1e-8))
        
        sinr_db_all.append(torch.stack(sinr_db_list))
    
    return torch.stack(sinr_db_all)

def train_v48(epochs=4000, bs=32, lr=5e-5):
    model = ISAC_v48()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-5)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=800, T_mult=2)
    
    for e in range(epochs):
        H = np.random.randn(bs, M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        Ht = torch.tensor(H, dtype=torch.float32)
        
        w_mag, w_phase, z, b = model(Ht)
        
        # 功率
        W_power = (w_mag ** 2).sum(dim=(1,2,3))
        Z_power = (z * b).sum(dim=(1,2))
        
        W_pwr = W_power * Pmax / 2
        Z_pwr = Z_power * Pmax / 2
        total_pwr = W_pwr + Z_pwr
        
        # 损失
        loss_pwr_over = F.relu(total_pwr - Pmax).mean() * 300
        loss_pwr_under = F.relu(Pmax * 0.95 - total_pwr).mean() * 200
        
        # AP选择
        b_reshaped = b.view(bs, M, P)
        ap_counts = b_reshaped.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 50
        
        # SINR约束 (采样计算以加速)
        if e % 200 == 0:
            sinr_db = compute_beam_sinr_batch(Ht, w_mag[:8], w_phase[:8])
            loss_sinr = F.relu(10 - sinr_db).mean() * 30
        else:
            loss_sinr = 0
        
        loss = loss_pwr_over + loss_pwr_under + loss_ap + loss_sinr
        
        if e % 800 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}, W={W_pwr.mean().item():.2f}W, Z={Z_pwr.mean().item():.2f}W")
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    
    torch.save(model.state_dict(), 'isac_v48.pth')
    print("v48完成!")
    return model

def test_v48(model, n=30):
    model.eval()
    results = []
    
    for i in range(n):
        H = np.random.randn(M, K, Nt*2).astype(np.float32)
        H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
        Ht = torch.tensor(H.reshape(1, M, K, Nt*2), dtype=torch.float32)
        
        w_mag, w_phase, z, b = model(Ht)
        
        W_power = (w_mag ** 2).sum().item()
        Z_power = (z * b).sum().item()
        W_pwr = W_power * Pmax / 2
        Z_pwr = Z_power * Pmax / 2
        total = W_pwr + Z_pwr
        
        # SINR
        sinr_db = compute_beam_sinr_batch(Ht, w_mag, w_phase)[0]
        
        b_np = b.squeeze().detach().numpy()
        aps = sum([len(np.argsort(-b_np[:, p])[:N_req]) for p in range(P)])
        
        results.append({'Total': total, 'W': W_pwr, 'Z': Z_pwr, 'SINR_min': sinr_db.min().item(), 'APs': aps})
    
    print(f"v48: 功率={np.mean([r['Total'] for r in results]):.2f}W")
    print(f"     W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR最小值: {np.min([r['SINR_min'] for r in results]):.2f}dB")
    return results

if __name__ == '__main__':
    model = train_v48()
    test_v48(model)
