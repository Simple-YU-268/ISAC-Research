"""ISAC AP选择神经网络 - 完整版"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations

M, K, P, Nt = 16, 10, 4, 4
Pmax = 30
sigma2 = 0.5

def mmse_beam(H, Pmax):
    M_sel = H.shape[0]
    Hs = H.reshape(M_sel * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    try:
        W = np.linalg.inv(HH) @ Hs
        p = np.sum(np.abs(W)**2)
        W = W * np.sqrt(Pmax * 0.8 / p)
        return W.reshape(M_sel, Nt, K)
    except:
        return None

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

def find_optimal_ap(H, target_ap=3):
    all_combos = list(combinations(range(M), target_ap))
    best_sinr = -100
    best_combo = None
    for combo in all_combos:
        H_sel = H[list(combo), :, :]
        W = mmse_beam(H_sel, Pmax * 0.8)
        if W is not None:
            sinrs = compute_sinr(H_sel, W)
            min_sinr = sinrs.min()
            if min_sinr > best_sinr:
                best_sinr = min_sinr
                best_combo = combo
    return best_combo, best_sinr

def generate_channel():
    ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
    user_pos = np.random.uniform(-30, 30, (K, 2))
    H = np.zeros((M, K, Nt), dtype=complex)
    for m in range(M):
        for k in range(K):
            d = max(np.sqrt(np.sum((ap[m] - user_pos[k])**2)), 5)
            pl = (d / 10)**-2.5
            H[m, k, :] = np.sqrt(pl / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
    return H

def prepare_input(H):
    H_real = np.zeros((M, K, Nt * 2), dtype=np.float32)
    H_real[:, :, :Nt] = np.real(H)
    H_real[:, :, Nt:] = np.imag(H)
    return H_real.flatten()

# 生成更多训练数据
print("生成训练数据...")
X, Y_ap, Y_sinr = [], [], []
for _ in range(800):
    H = generate_channel()
    combo, sinr = find_optimal_ap(H, target_ap=3)
    label = np.zeros(M, dtype=np.float32)
    if combo:
        for idx in combo:
            label[idx] = 1.0
    X.append(prepare_input(H))
    Y_ap.append(label)
    Y_sinr.append(sinr)

X = np.array(X)
Y_ap = np.array(Y_ap)
Y_sinr = np.array(Y_sinr)
print(f"训练数据: {len(X)} 样本")

class APNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 512), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, M), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

model = APNet()
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
Xt = torch.tensor(X, dtype=torch.float32)
Yt_ap = torch.tensor(Y_ap, dtype=torch.float32)
Yt_sinr = torch.tensor(Y_sinr, dtype=torch.float32)

print("训练...")
for e in range(3000):
    opt.zero_grad()
    idx = torch.randperm(len(X))[:32]
    
    pred = model(Xt[idx])
    
    # AP选择损失
    loss_ap = F.binary_cross_entropy(pred, Yt_ap[idx]) * 30
    
    # 可选: 预测SINR
    # loss_sinr = ((pred.mean() - Yt_sinr[idx].mean()) ** 2) * 0.1
    
    # AP数量约束
    select_count = pred.sum(dim=1)
    loss_count = F.relu(select_count - 4).abs().mean() * 5 + F.relu(2 - select_count).abs().mean() * 5
    
    loss = loss_ap + loss_count
    loss.backward()
    opt.step()
    
    if e % 600 == 0:
        print(f"Epoch {e}")

torch.save(model.state_dict(), 'isac_ap_nn.pth')

# 测试
print("\n=== 测试 ===")
model.eval()
for N_TEST in [3, 4, 5]:
    results = []
    for _ in range(200):
        H = generate_channel()
        pred = model(torch.tensor(prepare_input(H).reshape(1, -1), dtype=torch.float32)).squeeze()
        pred_np = pred.detach().numpy()
        
        topN = np.argsort(-pred_np)[:N_TEST]
        action = np.zeros(M)
        action[topN] = 1
        
        ap_mask = action > 0
        H_sel = H[ap_mask, :, :]
        W = mmse_beam(H_sel, Pmax * 0.8)
        sinrs = compute_sinr(H_sel, W)
        
        results.append({'sinr_min': sinrs.min(), 'ok': sum(sinrs >= 0)})
    
    print(f"N={N_TEST}: SINR_min平均={np.mean([r['sinr_min'] for r in results]):.2f}dB, 全部≥0dB: {sum(1 for r in results if r['ok']==K)}/200")
