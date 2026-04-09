"""
ISAC最终方案 - 完整功能
- 16 AP, 10用户, 4感知目标
- 通信 SINR≥0dB (全部用户)
- 功率≤30W
- AP选择 (基于信道强度)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

M, K, P, Nt = 16, 10, 4, 4
Pmax = 30
sigma2 = 0.5

print("╔══════════════════════════════════════════════════════════════╗")
print("║           ISAC 最终方案 - 完整实现                            ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║ 系统: {M}AP, {K}用户, {P}感知目标, {Nt}天线                        ║")
print(f"║ 目标: SINR≥0dB, 功率≤{Pmax}W                                  ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

# ============= 波束成形 =============
def mmse_beam(H, Pmax):
    M_sel, K, Nt = H.shape
    Hs = H.reshape(M_sel * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    try:
        W = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(W)**2)
        W = W * np.sqrt(Pmax / p)
        return W.reshape(M_sel, Nt, K)
    except:
        return None

def sensing_beam(H_t, Pmax):
    M_sel, P, Nt = H_t.shape
    p_sensing = Pmax / P
    Z = np.zeros((M_sel, P, Nt), dtype=complex)
    for p in range(P):
        h_t = H_t[:, p, :]
        norm = np.sqrt(np.sum(np.abs(h_t)**2))
        if norm > 0:
            Z[:, p, :] = np.conj(h_t) / norm * np.sqrt(p_sensing)
    return Z

def compute_sinr(H, W):
    M_sel, K, Nt = H.shape
    Hs = H.reshape(M_sel * Nt, K)
    W_flat = W.reshape(M_sel * Nt, K)
    sinrs = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W_flat[:, k]) @ Hs[:, k]))**2
        inter = sum(np.abs(np.sum(np.conj(W_flat[:, j]) @ Hs[:, k]))**2 for j in range(K) if j != k)
        sinrs.append(10 * np.log10(sig / (inter + sigma2) + 1e-10))
    return np.array(sinrs)

def compute_crb(H_t, Z):
    M_sel, P, Nt = H_t.shape
    crbs = []
    for p in range(P):
        h_eq = Z[:, p, :] * np.conj(H_t[:, p, :])
        power = np.sum(np.abs(h_eq)**2)
        crb = 1 / (power + 0.1)
        crbs.append(crb)
    return np.mean(crbs) if crbs else 1000

def generate_channel():
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
    
    return H_u, H_t

def select_ap(H_u, N_req=8):
    """基于信道强度选择AP"""
    signal_power = np.sum(np.abs(H_u) ** 2, axis=2)
    total_signal = signal_power.sum(axis=1)
    selected = np.argsort(-total_signal)[:N_req]
    ap_mask = np.zeros(M, dtype=bool)
    ap_mask[selected] = True
    return ap_mask

# ============= 神经网络 =============
def prepare_input(H):
    h = np.zeros((M, K, Nt * 2), dtype=np.float32)
    h[:, :, :Nt] = np.real(H)
    h[:, :, Nt:] = np.imag(H)
    return h.flatten()

print("生成训练数据...")
X, Y = [], []
for _ in range(1000):
    H_u, H_t = generate_channel()
    ap_mask = select_ap(H_u, N_req=8)
    H_u_sel = H_u[ap_mask, :, :]
    
    W = mmse_beam(H_u_sel, Pmax * 0.8)
    if W is not None:
        sinrs = compute_sinr(H_u_sel, W)
        X.append(prepare_input(H_u))
        Y.append(sinrs.min())

X = np.array(X)
Y = np.array(Y)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

print("训练网络...")
m = Net()
o = optim.AdamW(m.parameters(), lr=5e-5, weight_decay=1e-5)
Xt = torch.tensor(X, dtype=torch.float32)
Yt = torch.tensor(Y, dtype=torch.float32)

for e in range(3000):
    o.zero_grad()
    i = torch.randperm(len(X))[:32]
    loss = ((m(Xt[i]).squeeze() - Yt[i]) ** 2).mean() * 10
    loss.backward()
    o.step()

torch.save(m.state_dict(), 'isac_final.pth')

# ============= 测试 =============
print("\n=== 最终测试 (500次) ===")
m.eval()
results = []
for _ in range(500):
    H_u, H_t = generate_channel()
    
    # AP选择
    ap_mask = select_ap(H_u, N_req=8)
    H_u_sel = H_u[ap_mask, :, :]
    H_t_sel = H_t[ap_mask, :, :]
    
    # 波束成形
    W = mmse_beam(H_u_sel, Pmax * 0.8)
    Z = sensing_beam(H_t_sel, Pmax * 0.2)
    
    # 性能计算
    sinrs = compute_sinr(H_u_sel, W)
    crb = compute_crb(H_t_sel, Z)
    
    total_pwr = np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)
    
    results.append({
        'sinr_min': sinrs.min(),
        'sinr_mean': sinrs.mean(),
        'comm_ok': sum(sinrs >= 0),
        'crb': crb,
        'power': total_pwr
    })

# 输出结果
print(f"\n【通信性能】")
print(f"  SINR_min: 最小{np.min([r['sinr_min'] for r in results]):.2f}dB")
print(f"  SINR_min: 平均{np.mean([r['sinr_min'] for r in results]):.2f}dB")
print(f"  全部{K}用户≥0dB: {sum(1 for r in results if r['comm_ok']==K)}/500")

print(f"\n【感知性能】")
print(f"  CRB平均: {np.mean([r['crb'] for r in results]):.4f}")

print(f"\n【功率】")
print(f"  平均功率: {np.mean([r['power'] for r in results]):.2f}W")
print(f"  功率≤30W: {sum(1 for r in results if r['power'] <= 30)}/500")

# 判断目标达成
all_ok = sum(1 for r in results if r['comm_ok'] == K and r['power'] <= 30)
print(f"\n【目标达成】")
print(f"  完全满足: {all_ok}/500 ({all_ok/500*100:.1f}%)")

if all_ok >= 450:
    print("\n🎉 目标达成!")
else:
    print("\n⚠️ 需要调整参数")
