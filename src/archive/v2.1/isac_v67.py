"""
ISAC v67 - 结合MRT和MMSE，网络预测混合权重
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30
N_req = 4

def mrt_precoding(H, Pmax):
    H_complex = H
    w = np.zeros((M, K, Nt), dtype=complex)
    for k in range(K):
        h_k_conj = np.conj(H_complex[:, k, :])
        norm = np.sqrt(np.sum(np.abs(h_k_conj) ** 2))
        if norm > 0:
            w[:, k, :] = h_k_conj / norm * np.sqrt(Pmax * 0.7 / K)
    return w

def mmse_precoding(H, Pmax, sigma2=0.3):
    H_complex = H
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
        return w
    except:
        return mrt_precoding(H_complex, Pmax)

def compute_sinr(H, w):
    sinrs = []
    for k in range(K):
        signal = np.abs(np.sum(np.conj(w[:, k, :]) * H[:, k, :])) ** 2
        interference = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H[:, k, :])) ** 2 for j in range(K) if j != k)
        sinrs.append(10 * np.log10(signal / (interference + 0.01) + 1e-8))
    return np.array(sinrs)

def generate_data(n_samples):
    X_list, w_mrt_list, w_mmse_list = [], [], []
    
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
        
        w_mrt = mrt_precoding(H_complex, Pmax)
        w_mmse = mmse_precoding(H_complex, Pmax, sigma2=0.3)
        
        X_list.append(H.flatten())
        w_mrt_list.append(w_mrt.flatten())
        w_mmse_list.append(w_mmse.flatten())
    
    return np.array(X_list), np.array(w_mrt_list), np.array(w_mmse_list)

print("生成MRT+MMSE数据...")
X, W_mrt, W_mmse = generate_data(500)

class ISAC_v67(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1), nn.Sigmoid()  # 混合权重
        )
    
    def forward(self, x):
        return self.net(x)

def train_v67(epochs=4000, bs=32, lr=8e-5):
    model = ISAC_v67()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-5)
    
    Xt = torch.tensor(X, dtype=torch.float32)
    W_mrt_t = torch.tensor(W_mrt, dtype=torch.float32)
    W_mmse_t = torch.tensor(W_mmse, dtype=torch.float32)
    
    for e in range(epochs):
        opt.zero_grad()
        idx = torch.randperm(len(X))[:bs]
        
        alpha = model(Xt[idx])  # (bs, 1)
        
        # 混合波束
        w_mixed = alpha * W_mrt_t[idx] + (1 - alpha) * W_mmse_t[idx]
        
        # 计算功率约束
        W_pwr = (w_mixed ** 2).sum(dim=(1,2)) * 1.5
        
        loss_pwr = F.relu(W_pwr - Pmax).mean() * 300 + F.relu(Pmax * 0.95 - W_pwr).mean() * 200
        
        loss = loss_pwr
        loss.backward()
        opt.step()
        
        if e % 800 == 0:
            print(f"Epoch {e}: W_pwr={W_pwr.mean().item():.2f}W, alpha={alpha.mean().item():.2f}")
    
    torch.save(model.state_dict(), 'isac_v67.pth')
    print("v67完成!")
    return model

def test_v67(model, n=100):
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
        
        # 预测混合权重
        alpha = model(torch.tensor(H.flatten().reshape(1, -1), dtype=torch.float32)).item()
        
        # 构建混合波束
        w_mrt = mrt_precoding(H_complex, Pmax)
        w_mmse = mmse_precoding(H_complex, Pmax, sigma2=0.3)
        
        w = alpha * w_mrt + (1 - alpha) * w_mmse
        
        W_pwr = np.sum(np.abs(w) ** 2) * 1.5
        
        sinrs = compute_sinr(H_complex, w)
        results.append({'W': W_pwr, 'SINR_min': min(sinrs)})
    
    print(f"v67: 功率={np.mean([r['W'] for r in results]):.2f}W")
    print(f"     SINR_min: {np.min([r['SINR_min'] for r in results]):.2f}dB, 平均: {np.mean([r['SINR_min'] for r in results]):.2f}dB")
    print(f"     ≥ 0dB: {sum(1 for r in results if r['SINR_min'] >= 0)}/{n}")

if __name__ == '__main__':
    model = train_v67()
    test_v67(model)
