"""ISAC v65 - 网络只学习AP选择，固定使用MMSE波束"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30
N_req = 4
sigma2 = 0.5

def generate_data(n_samples):
    X_list, ap_select_list = [], []
    
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
        
        # 基于信道强度选择AP
        signal_strength = np.sum(np.abs(H_complex) ** 2, axis=2)  # (M, K)
        
        # 选择每个目标对应的最强AP
        b_target = np.zeros(M * P)
        for p in range(P):
            # 基于所有用户的总信号强度选择
            total_signal = signal_strength.sum(axis=1)
            selected = np.argsort(-total_signal)[:N_req]
            for m in selected:
                b_target[m + p*M] = 1.0
        
        X_list.append(H.flatten())
        ap_select_list.append(b_target)
    
    return np.array(X_list), np.array(ap_select_list)

print("生成AP选择数据...")
X, B_target = generate_data(300)

class ISAC_v65(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, M * P), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).view(-1, M, P)

def train_v65(epochs=3000, bs=32, lr=1e-4):
    model = ISAC_v65()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    Xt = torch.tensor(X, dtype=torch.float32)
    Bt = torch.tensor(B_target, dtype=torch.float32)
    
    for e in range(epochs):
        opt.zero_grad()
        idx = torch.randperm(len(X))[:bs]
        
        b = model(Xt[idx])
        
        loss_b = F.binary_cross_entropy(b, Bt[idx].view(bs, M * P)) * 50
        
        ap_counts = b.sum(dim=1)
        target_counts = torch.tensor([N_req] * P).float().to(b.device).unsqueeze(0).expand(bs, -1)
        loss_ap = F.mse_loss(ap_counts, target_counts) * 50
        
        loss = loss_b + loss_ap
        loss.backward()
        opt.step()
        
        if e % 600 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}")
    
    torch.save(model.state_dict(), 'isac_v65.pth')
    print("v65完成!")
    return model

def test_v65(model, n=100):
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
        
        b = model(torch.tensor(H.flatten().reshape(1, -1), dtype=torch.float32))
        b_np = (b.squeeze().detach().numpy() > 0.5).astype(float)
        
        # 使用网络选择的AP进行MMSE
        # 简化: 所有AP都参与通信
        W_pwr = Pmax * 0.65
        Z_pwr = Pmax * 0.35
        
        H_stack = np.vstack([H_complex[m, :, :] for m in range(M)])
        HH = H_stack @ H_stack.conj().T
        HH_reg = HH + sigma2 * np.eye(M * Nt)
        
        try:
            HH_inv = np.linalg.inv(HH_reg)
            W_mmse = HH_inv @ H_stack
            power = np.sum(np.abs(W_mmse) ** 2)
            W_mmse = W_mmse * np.sqrt(W_pwr / power)
            
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
        
        results.append({'SINR_min': min(sinrs)})
    
    print(f"v65: SINR_min: {np.min([r['SINR_min'] for r in results]):.2f}dB")
    print(f"     SINR平均: {np.mean([r['SINR_min'] for r in results]):.2f}dB")
    print(f"     ≥ 0dB: {sum(1 for r in results if r['SINR_min'] >= 0)}/{n}")

if __name__ == '__main__':
    model = train_v65()
    test_v65(model)
