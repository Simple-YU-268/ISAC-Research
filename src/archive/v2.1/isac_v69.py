"""ISAC v69 - 尝试更细致的混合策略"""
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

def mmse_beam(H, Pmax, s2):
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

def zf_beam(H, Pmax):
    Hs = np.vstack([H[m, :, :] for m in range(M)])
    try:
        Hp = np.linalg.pinv(Hs)
        W = Hp * np.sqrt(Pmax * 0.6 / K)
        w = np.zeros((M, K, Nt), dtype=complex)
        for k in range(K):
            for m in range(M):
                w[m, k, :] = W[k, m*Nt:(m+1)*Nt]
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

def gen_data(n):
    X, A = [], []
    for _ in range(n):
        ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        u = np.random.uniform(-50, 50, (K, 2))
        H = np.zeros((M, K, Nt), dtype=complex)
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap[m] - u[k])**2)), 5)
                H[m, k, :] = np.sqrt((d / 10)**-2 / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        
        # 测试多种组合
        methods = [('MRT', lambda: mrt_beam(H, Pmax)),
                   ('MMSE0.3', lambda: mmse_beam(H, Pmax, 0.3)),
                   ('MMSE0.5', lambda: mmse_beam(H, Pmax, 0.5)),
                   ('ZF', lambda: zf_beam(H, Pmax))]
        
        best_s, best_name, best_w = -100, None, None
        for name, fn in methods:
            w = fn()
            s = sinr(H, w).min()
            if s > best_s:
                best_s, best_name, best_w = s, name, w
        
        h = np.zeros((M, K, Nt * 2), dtype=np.float32)
        h[:, :, :Nt] = np.real(H)
        h[:, :, Nt:] = np.imag(H)
        X.append(h.flatten())
        
        # 编码为one-hot
        if best_name == 'MRT': A.append([1,0,0,0])
        elif best_name == 'MMSE0.3': A.append([0,1,0,0])
        elif best_name == 'MMSE0.5': A.append([0,0,1,0])
        else: A.append([0,0,0,1])
    
    return np.array(X), np.array(A)

print("生成3种预编码选择数据...")
X, A = gen_data(600)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 4), nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.fc(x)

def train(ep=4000):
    m = Net()
    o = optim.AdamW(m.parameters(), lr=8e-5)
    Xt = torch.tensor(X, dtype=torch.float32)
    At = torch.tensor(A, dtype=torch.float32)
    
    for e in range(ep):
        o.zero_grad()
        i = torch.randperm(len(X))[:32]
        loss = F.cross_entropy(m(Xt[i]), At[i].argmax(dim=1)) * 50
        loss.backward()
        o.step()
        if e % 800 == 0:
            print(f"Epoch {e}")
    
    torch.save(m.state_dict(), 'isac_v69.pth')
    return m

def test(m, n=100):
    m.eval()
    r = []
    for _ in range(n):
        ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        u = np.random.uniform(-50, 50, (K, 2))
        H = np.zeros((M, K, Nt), dtype=complex)
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap[m] - u[k])**2)), 5)
                H[m, k, :] = np.sqrt((d / 10)**-2 / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        
        p = m(torch.tensor(H.flatten().reshape(1,-1), dtype=torch.float32)).squeeze()
        method_id = p.argmax().item()
        
        if method_id == 0: w = mrt_beam(H, Pmax)
        elif method_id == 1: w = mmse_beam(H, Pmax, 0.3)
        elif method_id == 2: w = mmse_beam(H, Pmax, 0.5)
        else: w = zf_beam(H, Pmax)
        
        r.append(sinr(H, w).min())
    
    print(f"v69: SINR_min={min(r):.2f}dB, 平均={np.mean(r):.2f}dB, ≥0dB={sum(1 for x in r if x>=0)}/{n}")

m = train()
test(m)
