"""
ISAC v58 - 直接学习ZF功率分配 + 微调
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30
N_req = 4

def generate_zf_power_data(n_samples):
    """生成ZF功率分配数据"""
    X_list, p_list, z_list, b_list = [], [], [], []
    
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
        
        # 计算ZF伪逆
        H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
        H_stack = np.vstack([H_complex[m, :, :] for m in range(M)])
        
        try:
            H_pinv = np.linalg.pinv(H_stack)
            p_zf = np.sum(np.abs(H_pinv) ** 2, axis=1)
        except:
            p_zf = np.ones(K) / K * Pmax * 0.6
        
        # 归一化
        if np.sum(p_zf) > 0:
            p_zf = p_zf * Pmax * 0.6 / np.sum(p_zf)
        
        # 分配到AP-用户对
        p_assign = np.zeros(M * K)
        for k in range(K):
            p_assign[k*M:(k+1)*M] = p_zf[k] / M
        
        # 感知权重 (基于位置)
        target_pos = np.random.uniform(-30, 30, (P, 2))
        z = np.zeros(M * P)
        for p in range(P):
            for m in range(M):
                d = np.sqrt(np.sum((ap_pos[m] - target_pos[p])**2))
                z[m + p*M] = 1.0 / (d + 1)
        
        # AP选择
        b = np.zeros(M * P)
        for p in range(P):
            selected = np.random.choice(M, N_req, replace=False)
            for m in selected:
                b[m + p*M] = 1.0
        
        X_list.append(H.flatten())
        p_list.append(p_assign)
    
    return np.array(X_list), np.array(p_list)

print("生成数据...")
X, P_zf = generate_zf_power_data(200)
print(f"X: {X.shape}, P: {P_zf.shape}")

class ISAC_v58(nn.Module):
    def __init__(self):
        super().__init__()
        hd = M * K * Nt * 2
        
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        # 直接输出功率分配 (M*K)
        self.p_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * K), nn.Sigmoid())
        
        self.z_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
        self.b_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, M * P), nn.Sigmoid())
    
    def forward(self, x):
        emb = self.encoder(x)
        p = self.p_head(emb)
        z = self.z_head(emb).view(-1, M, P)
        b = self.b_head(emb).view(-1, M, P)
        return p, z, b

def train_v58(epochs=3000, bs=32, lr=1e-4):
    model = ISAC_v58()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    Xt = torch.tensor(X, dtype=torch.float32)
    Pt = torch.tensor(P_zf, dtype=torch.float32)
    
    for e in range(epochs):
        opt.zero_grad()
        idx = torch.randperm(len(X))[:bs]
        
        p, z, b = model(Xt[idx])
        
        # 监督
        loss_sup = F.mse_loss(p, Pt[idx]) * 50
        
        # 功率
        W_pwr = p.sum(dim=1) * 3
        Z_pwr = (z * b).sum(dim=(1,2)) * 3
        total_pwr = W_pwr + Z_pwr
        
        loss_pwr = F.relu(total_pwr - Pmax).mean() * 500 + F.relu(Pmax * 0.98 - total_pwr).mean() * 400
        
        # AP选择
        ap_counts = b.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 50
        
        loss = loss_sup + loss_pwr + loss_ap
        loss.backward()
        opt.step()
        
        if e % 600 == 0:
            print(f"Epoch {e}: W={W_pwr.mean().item():.2f}W, Z={Z_pwr.mean().item():.2f}W, total={total_pwr.mean().item():.2f}W")
    
    torch.save(model.state_dict(), 'isac_v58.pth')
    print("v58完成!")
    return model

def test_v58(model, n=50):
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
        
        p_np = p.squeeze().detach().numpy()  # (64,)
        z_np = z.squeeze().detach().numpy()
        b_np = b.squeeze().detach().numpy()
        
        # 构建波束: 使用ZF相位
        H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
        
        # 计算ZF相位
        H_stack = np.vstack([H_complex[m, :, :] for m in range(M)])
        try:
            H_pinv = np.linalg.pinv(H_stack)
        except:
            H_pinv = np.zeros((K, M*Nt))
        
        # 为每个用户构建波束，使用网络预测的功率
        w = np.zeros((M, K, Nt), dtype=complex)
        for k in range(K):
            p_k = p_np[k*M:(k+1)*M].sum()  # 用户k的总功率
            if p_k > 0:
                # 使用ZF相位
                w_zf = H_pinv[k, :]
                w_zf = w_zf / (np.linalg.norm(w_zf) + 1e-8)
                # 应用网络预测的功率
                for m in range(M):
                    p_mk = p_np[m + k*M]
                    w[m, k, :] = np.sqrt(p_mk) * w_zf[m*Nt:(m+1)*Nt]
        
        W_pwr = np.sum(p_np) * 3
        Z_pwr = np.sum(z_np * b_np) * 3
        
        sinrs = []
        for k in range(K):
            signal = np.abs(np.sum(np.conj(w[:, k, :]) * H_complex[:, k, :])) ** 2
            interference = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H_complex[:, k, :])) ** 2 for j in range(K) if j != k)
            sinrs.append(10 * np.log10(signal / (interference + 0.01) + 1e-8))
        
        results.append({'Total': W_pwr + Z_pwr, 'W': W_pwr, 'Z': Z_pwr, 'SINR_min': min(sinrs), 'SINR_mean': np.mean(sinrs)})
    
    print(f"v58: 功率={np.mean([r['Total'] for r in results]):.2f}W")
    print(f"     W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR_min: {np.min([r['SINR_min'] for r in results]):.2f}dB, 平均: {np.mean([r['SINR_min'] for r in results]):.2f}dB")
    print(f"     ≥ 0dB: {sum(1 for r in results if r['SINR_min'] >= 0)}/{n}")
    return results

if __name__ == '__main__':
    model = train_v58()
    test_v58(model)
