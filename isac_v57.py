"""
ISAC v57 - 学习迫零预编码的幅度和相位
使用ZF的相位作为监督，让网络学习与信道对齐的波束
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30
N_req = 4

def generate_zf_data(n_samples):
    """生成迫零预编码数据 (幅度+相位)"""
    X_list, w_real_list, w_imag_list = [], [], []
    
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
        
        # 计算ZF预编码
        H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
        H_stack = np.vstack([H_complex[m, :, :] for m in range(M)])
        
        try:
            H_pinv = np.linalg.pinv(H_stack)  # (K, M*Nt)
        except:
            H_pinv = np.zeros((K, M*Nt))
        
        # 提取每个AP-用户的波束
        w_zf = np.zeros((M, K, Nt), dtype=complex)
        for k in range(K):
            for m in range(M):
                w_zf[m, k, :] = H_pinv[k, m*Nt:(m+1)*Nt]
        
        # 归一化功率
        power = np.sum(np.abs(w_zf) ** 2)
        if power > 0:
            w_zf = w_zf * np.sqrt(Pmax * 0.7 / power)
        
        # 幅度 + 相位 (作为监督目标)
        w_mag = np.abs(w_zf)
        w_phase = np.angle(w_zf) / np.pi  # 归一化到 [-1, 1]
        
        X_list.append(H.flatten())
        w_real_list.append(w_mag.flatten())
        w_imag_list.append(w_phase.flatten())
    
    return np.array(X_list), np.array(w_real_list), np.array(w_imag_list)

print("生成ZF监督数据...")
X, W_mag_target, W_phase_target = generate_zf_data(200)
print(f"X: {X.shape}, W_mag: {W_mag_target.shape}, W_phase: {W_phase_target.shape}")

class ISAC_v57(nn.Module):
    def __init__(self):
        super().__init__()
        hd = M * K * Nt * 2
        
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        # 输出波束幅度和相位
        self.w_mag_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * K * Nt), nn.Sigmoid())
        self.w_phase_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * K * Nt), nn.Tanh())
        
        # 感知
        self.z_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * P), nn.Sigmoid())
        self.b_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * P), nn.Sigmoid())
    
    def forward(self, x):
        emb = self.encoder(x)
        
        w_mag = self.w_mag_head(emb).view(-1, M, K, Nt)
        w_phase = self.w_phase_head(emb).view(-1, M, K, Nt)
        
        z = self.z_head(emb).view(-1, M, P)
        b = self.b_head(emb).view(-1, M, P)
        
        return w_mag, w_phase, z, b

def train_v57(epochs=3000, bs=32, lr=1e-4):
    model = ISAC_v57()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    Xt = torch.tensor(X, dtype=torch.float32)
    Mt = torch.tensor(W_mag_target, dtype=torch.float32)
    Pt = torch.tensor(W_phase_target, dtype=torch.float32)
    
    for e in range(epochs):
        opt.zero_grad()
        idx = torch.randperm(len(X))[:bs]
        
        w_mag, w_phase, z, b = model(Xt[idx])
        
        # 监督损失: 幅度和相位
        loss_mag = F.mse_loss(w_mag.view(bs, -1), Mt[idx]) * 50
        loss_phase = F.mse_loss(w_phase.view(bs, -1), Pt[idx]) * 30
        
        # 功率约束
        W_pwr = (w_mag ** 2).sum(dim=(1,2,3)) * 20
        Z_pwr = (z * b).sum(dim=(1,2)) * 20
        total_pwr = W_pwr + Z_pwr
        
        loss_pwr = F.relu(total_pwr - Pmax).mean() * 400 + F.relu(Pmax * 0.95 - total_pwr).mean() * 300
        
        # AP选择
        ap_counts = b.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 50
        
        loss = loss_mag + loss_phase + loss_pwr + loss_ap
        
        loss.backward()
        opt.step()
        
        if e % 600 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}, W_pwr={W_pwr.mean().item():.2f}W")
    
    torch.save(model.state_dict(), 'isac_v57.pth')
    print("v57完成!")
    return model

def test_v57(model, n=50):
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
        
        w_mag, w_phase, z, b = model(torch.tensor(H.flatten().reshape(1, -1), dtype=torch.float32))
        
        # 使用学习到的幅度和相位构建波束
        w_mag_np = w_mag.squeeze().detach().numpy()
        w_phase_np = w_phase.squeeze().detach().numpy()
        
        # 复数波束
        w = w_mag_np * np.exp(1j * w_phase_np * np.pi)
        
        W_pwr = np.sum(w_mag_np ** 2) * 20
        Z_pwr = (z * b).sum().item() * 20
        total = W_pwr + Z_pwr
        
        # SINR
        H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
        sinrs = []
        for k in range(K):
            signal = np.abs(np.sum(np.conj(w[:, k, :]) * H_complex[:, k, :])) ** 2
            interference = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H_complex[:, k, :])) ** 2 for j in range(K) if j != k)
            sinrs.append(10 * np.log10(signal / (interference + 0.01) + 1e-8))
        
        results.append({'Total': total, 'W': W_pwr, 'Z': Z_pwr, 'SINR_min': min(sinrs), 'SINR_mean': np.mean(sinrs)})
    
    print(f"v57: 功率={np.mean([r['Total'] for r in results]):.2f}W")
    print(f"     W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR_min: {np.min([r['SINR_min'] for r in results]):.2f}dB, 平均: {np.mean([r['SINR_min'] for r in results]):.2f}dB")
    print(f"     ≥ 0dB: {sum(1 for r in results if r['SINR_min'] >= 0)}/{n}")
    return results

if __name__ == '__main__':
    model = train_v57()
    test_v57(model)
