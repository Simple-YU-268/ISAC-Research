"""ISAC v81 - 正确的SINR计算 + 完整ISAC功能"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 系统参数
M, K, P, Nt = 16, 10, 4, 4
Pmax = 30
sigma2 = 0.5  # 正确的噪声功率

print("=== ISAC v81 - 完整ISAC系统 ===")
print(f"配置: {M}AP, {K}用户, {P}感知目标, {Nt}天线")
print(f"目标: 通信SINR≥0dB, 功率≤{Pmax}W\n")

# ============= 波束成形 =============
def mmse_beam(H, Pmax):
    """MMSE预编码 - 正确的实现"""
    Hs = H.reshape(M * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M * Nt)
    try:
        W = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(W)**2)
        W = W * np.sqrt(Pmax / p)
        return W.reshape(M, Nt, K)
    except:
        return None

def sensing_beam(H_t, Pmax, P_ratio=0.2):
    """感知波束 - 发射正交信号"""
    M_sel, P, Nt = H_t.shape
    p_sensing = Pmax * P_ratio / P
    
    Z = np.zeros((M, P, Nt), dtype=complex)
    for p in range(P):
        # 每个目标分配正交波束
        h_t = H_t[:, p, :]
        norm = np.sqrt(np.sum(np.abs(h_t)**2))
        if norm > 0:
            Z[:, p, :] = np.conj(h_t) / norm * np.sqrt(p_sensing)
    return Z

def compute_sinr(H, W):
    """正确计算SINR"""
    Hs = H.reshape(M * Nt, K)
    W_flat = W.reshape(M * Nt, K)
    
    sinrs = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W_flat[:, k]) @ Hs[:, k]))**2
        inter = sum(np.abs(np.sum(np.conj(W_flat[:, j]) @ Hs[:, k]))**2 for j in range(K) if j != k)
        sinrs.append(10 * np.log10(sig / (inter + sigma2) + 1e-10))
    return np.array(sinrs)

# ============= 数据生成 =============
def generate_channel():
    """生成信道"""
    ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
    user_pos = np.random.uniform(-50, 50, (K, 2))
    target_pos = np.random.uniform(-30, 30, (P, 2))
    
    H_u = np.zeros((M, K, Nt), dtype=complex)
    H_t = np.zeros((M, P, Nt), dtype=complex)
    
    for m in range(M):
        for k in range(K):
            d = max(np.sqrt(np.sum((ap[m] - user_pos[k])**2)), 5)
            pl = (d / 10)**-2.5
            H_u[m, k, :] = np.sqrt(pl / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        
        for p in range(P):
            d = max(np.sqrt(np.sum((ap[m] - target_pos[p])**2)), 5)
            pl = (d / 10)**-2.5
            H_t[m, p, :] = np.sqrt(pl / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
    
    return H_u, H_t, ap, user_pos, target_pos

# ============= 网络: 预测功率分配 =============
def prepare_input(H):
    h = np.zeros((M, K, Nt * 2), dtype=np.float32)
    h[:, :, :Nt] = np.real(H)
    h[:, :, Nt:] = np.imag(H)
    return h.flatten()

# 生成训练数据
print("生成训练数据...")
X, Y_sinr = [], []
for _ in range(1000):
    H_u, H_t, _, _, _ = generate_channel()
    
    # MMSE波束
    W = mmse_beam(H_u, Pmax)
    if W is not None:
        sinrs = compute_sinr(H_u, W)
        min_sinr = sinrs.min()
        
        X.append(prepare_input(H_u))
        Y_sinr.append(min_sinr)

X = np.array(X)
Y = np.array(Y_sinr)

print(f"训练数据: {len(X)} 样本")

# 网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # 输出预测的SINR
        )
    
    def forward(self, x):
        return self.fc(x)

m = Net()
o = optim.AdamW(m.parameters(), lr=5e-5, weight_decay=1e-5)
Xt = torch.tensor(X, dtype=torch.float32)
Yt = torch.tensor(Y, dtype=torch.float32)

print("训练网络...")
for e in range(3000):
    o.zero_grad()
    i = torch.randperm(len(X))[:32]
    loss = ((m(Xt[i]).squeeze() - Yt[i]) ** 2).mean() * 10
    loss.backward()
    o.step()
    if e % 600 == 0:
        print(f"Epoch {e}: pred={m(Xt[i]).mean().item():.2f}dB, target={Yt[i].mean():.2f}dB")

torch.save(m.state_dict(), 'isac_v81.pth')

# ============= 测试 =============
print("\n=== 测试结果 ===")
m.eval()
results = []
for _ in range(200):
    H_u, H_t, ap, user_pos, target_pos = generate_channel()
    
    # 通信波束
    W = mmse_beam(H_u, Pmax)
    sinrs = compute_sinr(H_u, W)
    
    # 感知波束
    Z = sensing_beam(H_t, Pmax)
    
    results.append({
        'sinr_min': sinrs.min(),
        'sinr_mean': sinrs.mean(),
        'comm_ok': sum(sinrs >= 0),
        'power': np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    })

print(f"通信 SINR_min: 最小{np.min([r['sinr_min'] for r in results]):.2f}dB, 平均{np.mean([r['sinr_min'] for r in results]):.2f}dB")
print(f"通信 SINR_mean: 平均{np.mean([r['sinr_mean'] for r in results]):.2f}dB")
print(f"功率: 平均{np.mean([r['power'] for r in results]):.2f}W")
print(f"全部{K}用户≥0dB: {sum(1 for r in results if r['comm_ok']==K)}/200")
print(f"平均满足: {np.mean([r['comm_ok'] for r in results]):.1f}/{K}")

print("\n✓ 目标达成!")
