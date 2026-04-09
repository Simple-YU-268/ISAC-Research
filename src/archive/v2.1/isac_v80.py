"""ISAC v80 - 10用户场景 (修正输入维度)"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 10用户场景
M, K, Nt = 16, 10, 4
Pmax = 30
sigma2 = 0.5

def mmse_beam(H, Pmax):
    Hs = H.reshape(M * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M * Nt)
    try:
        W = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(W)**2)
        W = W * np.sqrt(Pmax / p)
        return W.reshape(M, Nt, K)
    except:
        return None

def sinr(H, w):
    Hs = H.reshape(M * Nt, K)
    if w.ndim == 3:
        W = w.reshape(M * Nt, K)
    else:
        W = w
    
    sinrs = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W[:, k]) @ Hs[:, k]))**2
        inter = sum(np.abs(np.sum(np.conj(W[:, j]) @ Hs[:, k]))**2 for j in range(K) if j != k)
        sinrs.append(10 * np.log10(sig / (inter + sigma2) + 1e-10))
    return np.array(sinrs)

def gen_data(n):
    X, A = [], []
    for _ in range(n):
        ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        u = np.random.uniform(-50, 50, (K, 2))
        
        H = np.zeros((M, K, Nt), dtype=complex)
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap[m] - u[k])**2)), 5)
                H[m, k, :] = np.sqrt((d / 10)**-2.5 / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        
        # 直接用MMSE (无需混合，因为MMSE本身就是最优的)
        w = mmse_beam(H, Pmax)
        
        sinrs = sinr(H, w)
        best_sinr = sinrs.min()
        
        h = np.zeros((M, K, Nt * 2), dtype=np.float32)
        h[:, :, :Nt] = np.real(H)
        h[:, :, Nt:] = np.imag(H)
        X.append(h.flatten())
        A.append(best_sinr)  # 直接回归最优SINR值
    return np.array(X), np.array(A)

print("=== ISAC v80 - 10用户场景 (直接预测最优SINR) ===")
print(f"生成1000个训练样本...")
X, A = gen_data(1000)

# 输入维度是 M*K*Nt*2 = 16*10*4*2 = 1280
input_dim = M * K * Nt * 2

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1)  # 输出SINR预测
        )
    
    def forward(self, x):
        return self.fc(x)

m = Net()
o = optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
Xt = torch.tensor(X, dtype=torch.float32)
At = torch.tensor(A, dtype=torch.float32)

print("训练...")
for e in range(3000):
    o.zero_grad()
    i = torch.randperm(len(X))[:32]
    loss = ((m(Xt[i]).squeeze() - At[i]) ** 2).mean() * 10
    loss.backward()
    o.step()
    if e % 600 == 0:
        print(f"Epoch {e}: pred={m(Xt[i]).mean().item():.2f}, target={At[i].mean():.2f}")

# 测试 - 使用网络选择最佳功率
m.eval()
r = []
for _ in range(100):
    ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
    u = np.random.uniform(-50, 50, (K, 2))
    H = np.zeros((M, K, Nt), dtype=complex)
    for m_i in range(M):
        for k in range(K):
            d = max(np.sqrt(np.sum((ap[m_i] - u[k])**2)), 5)
            H[m_i, k, :] = np.sqrt((d / 10)**-2.5 / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
    
    h_input = np.zeros((M, K, Nt * 2), dtype=np.float32)
    h_input[:, :, :Nt] = np.real(H)
    h_input[:, :, Nt:] = np.imag(H)
    
    # 网络预测最佳功率
    pred_sinr = m(torch.tensor(h_input.flatten().reshape(1, -1), dtype=torch.float32)).item()
    
    # 搜索最佳功率
    best_sinr = -100
    for pmax_test in [10, 20, 30, 50, 100]:
        w = mmse_beam(H, pmax_test)
        if w is not None:
            s = sinr(H, w).min()
            if s > best_sinr:
                best_sinr = s
    
    # 直接用30W MMSE
    w = mmse_beam(H, 30)
    sinrs = sinr(H, w)
    
    r.append({'min': sinrs.min(), 'mean': sinrs.mean(), 'ok': sum(sinrs >= 0)})

print(f"\n直接MMSE (Pmax=30W):")
print(f"  SINR_min: {np.min([x['min'] for x in r]):.2f}dB, 平均{np.mean([x['min'] for x in r]):.2f}dB")
print(f"  全部10用户≥0dB: {sum(1 for x in r if x['ok']==10)}/100")
print(f"  平均满足: {np.mean([x['ok'] for x in r]):.1f}/10")

torch.save(m.state_dict(), 'isac_v80.pth')
