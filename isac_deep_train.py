"""深度训练 - 大规模训练 + 多次测试"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 10, 4, 4
sigma2 = 0.5
Pmax = 80

print("="*60)
print("ISAC 深度训练系统")
print("="*60)

# ============= 波束函数 =============
def mmse_comm(H, P_comm):
    M_sel = H.shape[0]
    Hs = H.reshape(M_sel * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    try:
        W = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(W)**2)
        W = W * np.sqrt(P_comm / p)
        return W.reshape(M_sel, Nt, K)
    except:
        return None

def sensing_beam(H_t, P_sens):
    M_sel, P, Nt = H_t.shape
    p_per = P_sens / P
    Z = np.zeros((M_sel, P, Nt), dtype=complex)
    for p in range(P):
        for m in range(M_sel):
            h = H_t[m, p, :]
            norm = np.sqrt(np.sum(np.abs(h)**2))
            if norm > 0:
                Z[m, p, :] = np.conj(h) / norm * np.sqrt(p_per / M_sel)
    return Z

def compute_sinr(H, W):
    M_sel = H.shape[0]
    Hs = H.reshape(M_sel * Nt, K)
    W_flat = W.reshape(M_sel * Nt, K)
    sinrs = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W_flat[:, k]) @ Hs[:, k]))**2
        inter = sum(np.abs(np.sum(np.conj(W_flat[:, j]) @ Hs[:, k]))**2 for j in range(K) if j != k)
        sinrs.append(10 * np.log10(sig / (inter + sigma2) + 1e-10))
    return np.array(sinrs)

def compute_sensing_snr(H_t, Z):
    M_sel = H_t.shape[0]
    snrs = []
    for p in range(P):
        signal = sum(np.abs(np.sum(Z[m, p, :] * np.conj(H_t[m, p, :])))**2 for m in range(M_sel))
        noise = sigma2 * np.sum(np.abs(Z)**2)
        snrs.append(10 * np.log10(signal / (noise + 1e-10) + 1e-10))
    return np.array(snrs)

# ============= 数据生成 =============
def generate_channel():
    ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
    user_pos = np.random.uniform(-15, 15, (K, 2))
    target_pos = np.random.uniform(-12, 12, (P, 2))
    
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

def find_optimal_ap(H_u, n_ap=5):
    """使用CVX找最优AP组合"""
    from itertools import combinations
    
    best_sinr = -100
    best_combo = None
    
    all_combos = list(combinations(range(M), n_ap))
    
    for combo in all_combos:
        H_sel = H_u[list(combo), :, :]
        W = mmse_comm(H_sel, Pmax * 0.75)
        
        if W is not None:
            sinrs = compute_sinr(H_sel, W)
            min_sinr = min(sinrs)
            
            if min_sinr > best_sinr:
                best_sinr = min_sinr
                best_combo = combo
    
    return best_combo, best_sinr

# ============= 生成大规模训练数据 =============
print("\n[1/4] 生成训练数据...")

X, Y = [], []
target_n_ap = 5

for i in range(3000):
    H_u, H_t = generate_channel()
    combo, _ = find_optimal_ap(H_u, target_n_ap)
    
    # 标签
    label = np.zeros(M, dtype=np.float32)
    if combo:
        for idx in combo:
            label[idx] = 1.0
    
    # 输入
    H_input = np.zeros((M, K, Nt * 2), dtype=np.float32)
    H_input[:, :, :Nt] = np.real(H_u)
    H_input[:, :, Nt:] = np.imag(H_u)
    
    X.append(H_input.flatten())
    Y.append(label)

X = np.array(X)
Y = np.array(Y)
print(f"训练数据: {len(X)} 样本")

# ============= 神经网络 =============
class APNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 512), nn.LeakyReLU(0.2), nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, M), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

print("\n[2/4] 训练网络...")

model = APNet()
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
Xt = torch.tensor(X, dtype=torch.float32)
Yt = torch.tensor(Y, dtype=torch.float32)

batch_size = 64
epochs = 10000

for e in range(epochs):
    opt.zero_grad()
    idx = torch.randperm(len(X))[:batch_size]
    
    pred = model(Xt[idx])
    
    # 交叉熵损失
    loss = F.binary_cross_entropy(pred, Yt[idx]) * 50
    
    # AP数量约束
    select_count = pred.sum(dim=1)
    loss += F.relu(select_count - 7).abs().mean() * 10
    loss += F.relu(3 - select_count).abs().mean() * 10
    
    loss.backward()
    opt.step()
    
    if e % 1000 == 0:
        print(f"Epoch {e}/{epochs}")

torch.save(model.state_dict(), 'isac_deep.pth')
print("模型保存!")

# ============= 测试 =============
print("\n[3/4] 深度测试...")

model.eval()

def test_model(n_tests=500):
    success = 0
    comm_ok = 0
    sensing_ok = 0
    power_ok = 0
    
    for _ in range(n_tests):
        H_u, H_t = generate_channel()
        
        # 网络选择
        pred = model(torch.tensor(H_u.flatten().reshape(1, -1), dtype=torch.float32)).squeeze()
        pred_np = pred.detach().numpy()
        
        # 选择前5个
        top5 = np.argsort(-pred_np)[:5]
        action = np.zeros(M)
        action[top5] = 1
        
        ap_mask = action > 0
        H_u_sel = H_u[ap_mask, :, :]
        H_t_sel = H_t[ap_mask, :, :]
        
        W = mmse_comm(H_u_sel, Pmax * 0.75)
        Z = sensing_beam(H_t_sel, Pmax * 0.25)
        
        # 检查约束
        comm_sinrs = compute_sinr(H_u_sel, W)
        sensing_snrs = compute_sensing_snr(H_t_sel, Z)
        
        c_ok = all(s >= 0 for s in comm_sinrs)
        s_ok = all(s >= -22 for s in sensing_snrs)
        p_ok = (np.sum(np.abs(W)**2) + np.sum(np.abs(Z)**2)) <= Pmax
        
        if c_ok:
            comm_ok += 1
        if s_ok:
            sensing_ok += 1
        if p_ok:
            power_ok += 1
        if c_ok and s_ok and p_ok:
            success += 1
    
    return success, comm_ok, sensing_ok, power_ok

# 多次测试取平均
results = []
for run in range(5):
    s, c, sen, p = test_model(500)
    results.append((s, c, sen, p))
    print(f"测试{run+1}: 完全满足={s}/500, 通信={c}, 感知={sen}, 功率={p}")

avg_success = np.mean([r[0] for r in results])
print(f"\n平均结果: 完全满足 {avg_success:.1f}/500 ({avg_success*100/500:.1f}%)")

print("\n[4/4] 完成!")
print("="*60)
