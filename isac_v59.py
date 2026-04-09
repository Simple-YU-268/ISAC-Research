"""
ISAC v59 - 直接应用ZF预编码
网络只负责选择哪些AP参与通信
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30
N_req = 4

def generate_data(n_samples):
    X_list, ap_select_list = [], []
    
    for _ in range(n_samples):
        ap_pos = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        user_pos = np.random.uniform(-50, 50, (K, 2))
        target_pos = np.random.uniform(-30, 30, (P, 2))
        
        H = np.zeros((M, K, Nt*2), dtype=np.float32)
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap_pos[m] - user_pos[k])**2)), 5)
                pl = (d / 10) ** -2
                h = np.sqrt(pl/2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
                H[m, k, :Nt] = np.real(h)
                H[m, k, Nt:] = np.imag(h)
        
        # 计算ZF预编码的最优AP选择
        H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
        H_stack = np.vstack([H_complex[m, :, :] for m in range(M)])
        
        try:
            H_pinv = np.linalg.pinv(H_stack)
            p_zf = np.sum(np.abs(H_pinv) ** 2, axis=1)  # 每个用户的功率
        except:
            p_zf = np.ones(K) / K * Pmax * 0.6
        
        # 选择功率最小的AP子集
        ap_contribution = np.zeros(M)
        for k in range(K):
            for m in range(M):
                ap_contribution[m] += p_zf[k] / M
        
        # 选择信号最强的AP
        signal_strength = np.zeros(M)
        for k in range(K):
            signal_strength += np.sum(np.abs(H_complex[:, k, :]) ** 2, axis=1)
        
        # 选择信号最强的N_req个AP用于通信
        ap_select = np.zeros(M * P)
        for p in range(P):
            selected = np.argsort(-signal_strength)[:N_req]
            for m in selected:
                ap_select[m + p*M] = 1.0
        
        X_list.append(H.flatten())
        ap_select_list.append(ap_select)
    
    return np.array(X_list), np.array(ap_select_list)

print("生成数据...")
X, B_target = generate_data(200)

class ISAC_v59(nn.Module):
    def __init__(self):
        super().__init__()
        hd = M * K * Nt * 2
        
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        self.b_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * P), nn.Sigmoid())
        self.z_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * P), nn.Sigmoid())
    
    def forward(self, x):
        emb = self.encoder(x)
        b = self.b_head(emb).view(-1, M, P)
        z = self.z_head(emb).view(-1, M, P)
        return b, z

def train_v59(epochs=3000, bs=32, lr=1e-4):
    model = ISAC_v59()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    Xt = torch.tensor(X, dtype=torch.float32)
    Bt = torch.tensor(B_target, dtype=torch.float32)
    
    for e in range(epochs):
        opt.zero_grad()
        idx = torch.randperm(len(X))[:bs]
        
        b, z = model(Xt[idx])
        
        # 监督AP选择
        loss_b = F.binary_cross_entropy_with_logits(b.view(bs, -1), Bt[idx]) * 50
        
        # 功率约束
        P_comm = Pmax * 0.6
        P_sens = Pmax * 0.4
        
        Z_pwr = (z * b).sum(dim=(1,2)) * 20
        
        loss_pwr = F.relu(Z_pwr - P_sens).mean() * 200
        
        # AP选择约束
        ap_counts = b.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 50
        
        loss = loss_b + loss_pwr + loss_ap
        loss.backward()
        opt.step()
        
        if e % 600 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}")
    
    torch.save(model.state_dict(), 'isac_v59.pth')
    print("v59完成!")
    return model

def test_v59(model, n=50):
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
        
        b, z = model(torch.tensor(H.flatten().reshape(1, -1), dtype=torch.float32))
        
        b_np = (b.squeeze().detach().numpy() > 0.5).astype(float)
        z_np = z.squeeze().detach().numpy()
        
        # 使用选中的AP进行通信
        H_complex = H[:, :, :Nt] + 1j * H[:, :, Nt:]
        
        # 等功率分配给选中的AP
        W_pwr = Pmax * 0.6
        Z_pwr = np.sum(z_np * b_np) * 20
        
        # ZF预编码 (只使用选中的AP)
        selected_aps = np.where(b_np[:, 0] > 0.5)[0]  # 简化为目标0
        
        if len(selected_aps) >= K:
            H_sel = H_complex[selected_aps, :, :]
            H_stack = np.vstack([H_sel[m, :, :] for m in range(len(selected_aps))])
            
            try:
                H_pinv = np.linalg.pinv(H_stack)
                
                w = np.zeros((M, K, Nt), dtype=complex)
                for k in range(K):
                    w_zf = H_pinv[k, :]
                    w_zf = w_zf / (np.linalg.norm(w_zf) + 1e-8) * np.sqrt(W_pwr / K / len(selected_aps))
                    
                    for idx, m in enumerate(selected_aps):
                        w[m, k, :] = w_zf[idx*Nt:(idx+1)*Nt]
                
                sinrs = []
                for k in range(K):
                    signal = np.abs(np.sum(np.conj(w[:, k, :]) * H_complex[:, k, :])) ** 2
                    interference = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H_complex[:, k, :])) ** 2 for j in range(K) if j != k)
                    sinrs.append(10 * np.log10(signal / (interference + 0.01) + 1e-8))
            except:
                sinrs = [-100] * K
        else:
            sinrs = [-100] * K
        
        results.append({'Total': W_pwr + Z_pwr, 'W': W_pwr, 'Z': Z_pwr, 'SINR_min': min(sinrs)})
    
    print(f"v59: 功率={np.mean([r['Total'] for r in results]):.2f}W")
    print(f"     W={np.mean([r['W'] for r in results]):.2f}W, Z={np.mean([r['Z'] for r in results]):.2f}W")
    print(f"     SINR_min: {np.min([r['SINR_min'] for r in results]):.2f}dB, 平均: {np.mean([r['SINR_min'] for r in results]):.2f}dB")
    print(f"     ≥ 0dB: {sum(1 for r in results if r['SINR_min'] >= 0)}/{n}")

if __name__ == '__main__':
    model = train_v59()
    test_v59(model)
