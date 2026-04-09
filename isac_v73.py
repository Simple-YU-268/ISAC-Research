"""ISAC v73 - 结合监督学习 + MRT/MMSE混合选择"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30

def mrt_beam(H, Pmax):
    w = np.zeros((M, K, Nt), dtype=complex)
    for k in range(K):
        h = np.conj(H[:, k, :])
        norm = np.sqrt(np.sum(np.abs(h)**2))
        if norm > 0:
            w[:, k, :] = h / norm * np.sqrt(Pmax * 0.7 / K)
    return w

def mmse_beam(H, Pmax, s2=0.3):
    Hs = np.vstack([H[m, :, :] for m in range(M)])
    HH = Hs @ Hs.T.conj() + s2 * np.eye(M * Nt)
    try:
        Hi = np.linalg.inv(HH)
        W = Hi @ Hs
        p = np.sum(np.abs(W)**2)
        W = W * np.sqrt(Pmax * 0.65 / p)
        w = np.zeros((M, K, Nt), dtype=complex)
        for k in range(K):
            for m in range(M):
                w[m, k, :] = W[m * Nt:(m+1)*Nt, k]
        return w
    except:
        return mrt_beam(H, Pmax)

def sinr(H, w):
    s = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(w[:, k, :]) * H[:, k, :]))**2
        inter = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H[:, k, :]))**2 for j in range(K) if j != k)
        s.append(10 * np.log10(sig / (inter + 0.01) + 1e-8))
    return np.array(s)

def generate_data(n):
    X, A = [], []
    for _ in range(n):
        ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        u = np.random.uniform(-50, 50, (K, 2))
        
        H = np.zeros((M, K, Nt), dtype=complex)
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap[m] - u[k])**2)), 5)
                H[m, k, :] = np.sqrt((d / 10)**-2 / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        
        w_m = mrt_beam(H, Pmax)
        w_x = mmse_beam(H, Pmax, 0.3)
        
        # 找最优alpha
        best_a, best_s = 0, -100
        for a in np.linspace(0, 1, 21):
            w = a * w_m + (1 - a) * w_x
            s = sinr(H, w).min()
            if s > best_s:
                best_s, best_a = s, a
        
        h = np.zeros((M, K, Nt * 2), dtype=np.float32)
        h[:, :, :Nt] = np.real(H)
        h[:, :, Nt:] = np.imag(H)
        X.append(h.flatten())
        A.append(best_a)
    return np.array(X), np.array(A)

print("生成更多训练数据...")
X, A = generate_data(1000)  # 增加训练样本

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

def train(ep=6000):
    m = Net()
    o = optim.AdamW(m.parameters(), lr=5e-5, weight_decay=2e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(o, T_max=ep)
    Xt = torch.tensor(X, dtype=torch.float32)
    At = torch.tensor(A, dtype=torch.float32)
    
    for e in range(ep):
        o.zero_grad()
        i = torch.randperm(len(X))[:32]
        loss = F.mse_loss(m(Xt[i]).squeeze(), At[i]) * 30
        loss.backward()
        o.step()
        sched.step()
        
        if e % 1200 == 0:
            print(f"Epoch {e}: pred={m(Xt[i]).mean().item():.2f}, target={At[i].mean():.2f}")
    
    torch.save(m.state_dict(), 'isac_v73.pth')
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
        
        a = m(torch.tensor(H.flatten().reshape(1,-1), dtype=torch.float32)).item()
        w = a * mrt_beam(H, Pmax) + (1 - a) * mmse_beam(H, Pmax, 0.3)
        r.append(sinr(H, w).min())
    
    print(f"v73: SINR_min={min(r):.2f}dB, 平均={np.mean(r):.2f}dB, ≥0dB={sum(1 for x in r if x>=0)}/{n}")

m = train()
test(m)
