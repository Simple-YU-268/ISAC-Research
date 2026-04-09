"""ISAC v72 - 知识蒸馏: 让网络学习完整的MMSE输出"""
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

def generate_teacher_data(n_samples):
    """生成教师(MMSE)的完整输出作为标签"""
    X_list, W_mag_list, W_phase_list = [], [], []
    
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
            
            # 提取幅度和相位
            w_mag = np.abs(W)  # (M*Nt, K)
            w_phase = np.angle(W)  # (M*Nt, K)
        except:
            w_mag = np.ones((M * Nt, K)) * 0.1
            w_phase = np.zeros((M * Nt, K))
        
        h_input = np.zeros((M, K, Nt * 2), dtype=np.float32)
        h_input[:, :, :Nt] = np.real(H)
        h_input[:, :, Nt:] = np.imag(H)
        
        X_list.append(h_input.flatten())
        W_mag_list.append(w_mag.flatten())
        W_phase_list.append(w_phase.flatten())
    
    return np.array(X_list), np.array(W_mag_list), np.array(W_phase_list)

print("生成教师数据...")
X, W_mag, W_phase = generate_teacher_data(600)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
        )
        
        self.mag_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * Nt * K), nn.Sigmoid())
        self.phase_head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, M * Nt * K), nn.Tanh())
    
    def forward(self, x):
        z = self.encoder(x)
        mag = self.mag_head(z) * 2.0  # 放大
        phase = self.phase_head(z) * np.pi
        return mag, phase

def train(ep=5000):
    m = Net()
    o = optim.AdamW(m.parameters(), lr=5e-5, weight_decay=1e-5)
    Xt = torch.tensor(X, dtype=torch.float32)
    Mt = torch.tensor(W_mag, dtype=torch.float32)
    Pt = torch.tensor(W_phase, dtype=torch.float32)
    
    for e in range(ep):
        o.zero_grad()
        i = torch.randperm(len(X))[:32]
        
        mag_pred, phase_pred = m(Xt[i])
        
        # 知识蒸馏损失
        loss_mag = F.mse_loss(mag_pred, Mt[i]) * 20
        loss_phase = F.mse_loss(phase_pred, Pt[i]) * 5
        
        # 功率约束
        w_complex = mag_pred * torch.exp(1j * phase_pred)
        W_pwr = (torch.abs(w_complex) ** 2).sum() * 1.0
        loss_pwr = F.relu(W_pwr - Pmax).mean() * 200
        
        loss = loss_mag + loss_phase + loss_pwr
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        o.step()
        
        if e % 1000 == 0:
            print(f"Epoch {e}: mag_loss={loss_mag.item():.4f}, pwr={W_pwr.item():.2f}W")
    
    torch.save(m.state_dict(), 'isac_v72.pth')
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
        
        mag, phase = m(torch.tensor(h_input.flatten().reshape(1, -1), dtype=torch.float32))
        mag_np = mag.squeeze().detach().numpy()
        phase_np = phase.squeeze().detach().numpy()
        
        # 重构波束
        w = mag_np.reshape(M * Nt, K) * np.exp(1j * phase_np.reshape(M * Nt, K))
        w = w.reshape(M, Nt, K).transpose(0, 2, 1)  # (M, K, Nt)
        
        W_pwr = np.sum(np.abs(w)**2)
        
        r.append({'pwr': W_pwr, 'sinr': sinr(H, w).min()})
    
    print(f"v72: 功率={np.mean([x['pwr'] for x in r]):.2f}W")
    print(f"     SINR_min={np.min([x['sinr'] for x in r]):.2f}dB, 平均={np.mean([x['sinr'] for x in r]):.2f}dB")
    print(f"     ≥0dB: {sum(1 for x in r if x['sinr']>=0)}/{n}")

m = train()
test(m)
