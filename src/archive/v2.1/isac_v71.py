"""ISAC v71 - 简化的监督学习: 直接用MMSE功率分配作为标签"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30

def sinr(H, w):
    s = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(w[:, k, :]) * H[:, k, :]))**2
        inter = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H[:, k, :]))**2 for j in range(K) if j != k)
        s.append(10 * np.log10(sig / (inter + 0.01) + 1e-8))
    return np.array(s)

def generate_mmse_labels(n_samples):
    X_list, P_list = [], []
    
    for _ in range(n_samples):
        ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        u = np.random.uniform(-50, 50, (K, 2))
        
        H = np.zeros((M, K, Nt), dtype=complex)
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap[m] - u[k])**2)), 5)
                H[m, k, :] = np.sqrt((d / 10)**-2 / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        
        # MMSE预编码
        Hs = np.vstack([H[m, :, :] for m in range(M)])
        HH = Hs @ Hs.T.conj() + 0.3 * np.eye(M * Nt)
        
        try:
            Hi = np.linalg.inv(HH)
            W = Hi @ Hs
            p = np.sum(np.abs(W)**2)
            W = W * np.sqrt(Pmax * 0.65 / p)
            
            # 每个AP-用户的功率
            p_mmse = np.zeros(M * K)
            for k in range(K):
                for m in range(M):
                    p_mmse[m + k * M] = np.sum(np.abs(W[m * Nt:(m+1)*Nt, k])**2)
        except:
            p_mmse = np.ones(M * K) * Pmax * 0.65 / (M * K)
        
        h_input = np.zeros((M, K, Nt * 2), dtype=np.float32)
        h_input[:, :, :Nt] = np.real(H)
        h_input[:, :, Nt:] = np.imag(H)
        
        X_list.append(h_input.flatten())
        P_list.append(p_mmse)
    
    return np.array(X_list), np.array(P_list)

print("生成MMSE监督数据...")
X, P_mmse = generate_mmse_labels(500)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, M * K), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

def train(ep=5000):
    m = Net()
    o = optim.AdamW(m.parameters(), lr=5e-5, weight_decay=1e-5)
    Xt = torch.tensor(X, dtype=torch.float32)
    Pt = torch.tensor(P_mmse, dtype=torch.float32)
    
    for e in range(ep):
        o.zero_grad()
        i = torch.randperm(len(X))[:32]
        
        p_pred = m(Xt[i])
        
        # 监督损失
        loss_sup = F.mse_loss(p_pred, Pt[i]) * 40
        
        # 功率约束
        W_pwr = p_pred.sum(dim=1) * 1.0
        loss_pwr = F.relu(W_pwr - Pmax).mean() * 300 + F.relu(Pmax * 0.95 - W_pwr).mean() * 200
        
        loss = loss_sup + loss_pwr
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        o.step()
        
        if e % 1000 == 0:
            print(f"Epoch {e}: W_pwr={W_pwr.mean().item():.2f}W")
    
    torch.save(m.state_dict(), 'isac_v71.pth')
    return m

def test(m, n=100):
    m.eval()
    r = []
    
    for _ in range(n):
        ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        u = np.random.uniform(-50, 50, (K, 2))
        
        H = np.zeros((M, K, Nt), dtype=complex)
        for m_i in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap[m_i] - u[k])**2)), 5)
                H[m_i, k, :] = np.sqrt((d / 10)**-2 / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        
        h_input = np.zeros((M, K, Nt * 2), dtype=np.float32)
        h_input[:, :, :Nt] = np.real(H)
        h_input[:, :, Nt:] = np.imag(H)
        
        p_pred = m(torch.tensor(h_input.flatten().reshape(1, -1), dtype=torch.float32)).squeeze()
        p_np = p_pred.detach().numpy()
        
        # 使用预测功率的MMSE
        Hs = np.vstack([H[m, :, :] for m in range(M)])
        HH = Hs @ Hs.T.conj() + 0.3 * np.eye(M * Nt)
        
        try:
            Hi = np.linalg.inv(HH)
            W = Hi @ Hs
            p = np.sum(np.abs(W)**2)
            # 按预测功率比例缩放
            scale = np.sum(p_np * 1.0) / p * 0.65
            scale = max(0.1, min(scale, 2.0))
            W = W * np.sqrt(Pmax * scale / p)
            
            w = np.zeros((M, K, Nt), dtype=complex)
            for k in range(K):
                for m in range(M):
                    w[m, k, :] = W[m * Nt:(m+1)*Nt, k]
        except:
            w = np.zeros((M, K, Nt), dtype=complex)
        
        W_pwr = np.sum(p_np) * 1.0
        
        r.append({'pwr': W_pwr, 'sinr': sinr(H, w).min()})
    
    print(f"v71: 功率={np.mean([x['pwr'] for x in r]):.2f}W")
    print(f"     SINR_min={np.min([x['sinr'] for x in r]):.2f}dB, 平均={np.mean([x['sinr'] for x in r]):.2f}dB")
    print(f"     ≥0dB: {sum(1 for x in r if x['sinr']>=0)}/{n}")

m = train()
test(m)
