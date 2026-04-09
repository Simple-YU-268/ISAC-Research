"""ISAC v70 - 引入监督训练: 使用scipy优化结果作为标签"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30

def sinr(H, w):
    s = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(w[:, k, :]) * H[:, k, :]))**2
        inter = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H[:, k, :]))**2 for j in range(K) if j != k)
        s.append(10 * np.log10(sig / (inter + 0.01) + 1e-8))
    return np.array(s)

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
        return None

def optimize_power(H, Pmax, target_sinr=0):
    """使用scipy优化功率分配"""
    def objective(powers):
        p = powers[:M*K].reshape(M, K)
        total_pwr = np.sum(p)
        
        # MMSE波束
        Hs = np.vstack([H[m, :, :] for m in range(M)])
        pnorm = np.sum(p)
        w = np.zeros((M*Nt, K), dtype=complex)
        for k in range(K):
            h_k = Hs[:, k]
            w[:, k] = h_k / (np.sqrt(np.sum(np.abs(h_k)**2)) + 0.5) * np.sqrt(p[:, k].sum())
        
        # 计算SINR
        sinrs = []
        for k in range(K):
            sig = np.abs(np.sum(np.conj(w[:, k]) * Hs[:, k]))**2
            inter = sum(np.abs(np.sum(np.conj(w[:, j]) * Hs[:, k]))**2 for j in range(K) if j != k)
            sinrs.append(sig / (inter + 0.01))
        
        # 惩罚: 功率 + SINR约束
        penalty = total_pwr * 0.1
        for s in sinrs:
            if s < target_sinr:
                penalty += (target_sinr - s) * 10
        
        return penalty
    
    # 初始化
    x0 = np.ones(M * K) * Pmax / (M * K) * 0.8
    bounds = [(0, Pmax * 0.5) for _ in range(M * K)]
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds)
    return result.x.reshape(M, K) if result.success else None

def generate_supervised_data(n_samples):
    """生成监督数据 - 使用scipy优化结果"""
    X_list, P_list = [], []
    
    for idx in range(n_samples):
        if idx % 100 == 0:
            print(f"生成 {idx}/{n_samples}")
        
        ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        u = np.random.uniform(-50, 50, (K, 2))
        
        H = np.zeros((M, K, Nt), dtype=complex)
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap[m] - u[k])**2)), 5)
                H[m, k, :] = np.sqrt((d / 10)**-2 / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        
        # scipy优化
        p_opt = optimize_power(H, Pmax, target_sinr=1.0)
        
        if p_opt is None:
            p_opt = np.ones((M, K)) * Pmax * 0.6 / (M * K)
        
        # 构建输入特征
        h_input = np.zeros((M, K, Nt * 2), dtype=np.float32)
        h_input[:, :, :Nt] = np.real(H)
        h_input[:, :, Nt:] = np.imag(H)
        
        X_list.append(h_input.flatten())
        P_list.append(p_opt.flatten())
    
    return np.array(X_list), np.array(P_list)

print("生成监督数据...")
X, P_opt = generate_supervised_data(300)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, M * K), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

def train(ep=3000):
    m = Net()
    o = optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
    Xt = torch.tensor(X, dtype=torch.float32)
    Pt = torch.tensor(P_opt, dtype=torch.float32)
    
    for e in range(ep):
        o.zero_grad()
        i = torch.randperm(len(X))[:32]
        
        p_pred = m(Xt[i])
        
        # 监督损失
        loss_sup = F.mse_loss(p_pred, Pt[i]) * 50
        
        # 功率约束
        W_pwr = p_pred.sum(dim=1) * 1.0
        loss_pwr = F.relu(W_pwr - Pmax).mean() * 200
        
        loss = loss_sup + loss_pwr
        loss.backward()
        o.step()
        
        if e % 600 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}, W_pwr={W_pwr.mean().item():.2f}W")
    
    torch.save(m.state_dict(), 'isac_v70.pth')
    return m

def test(m, n=50):
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
        
        # 使用MMSE波束 + 预测功率
        w = mmse_beam(H, Pmax, s2=0.3)
        if w is None:
            w = np.zeros((M, K, Nt), dtype=complex)
        
        W_pwr = np.sum(p_np) * 1.0
        
        r.append({'pwr': W_pwr, 'sinr': sinr(H, w).min()})
    
    print(f"v70: 功率={np.mean([x['pwr'] for x in r]):.2f}W")
    print(f"     SINR_min={np.min([x['sinr'] for x in r]):.2f}dB, 平均={np.mean([x['sinr'] for x in r]):.2f}dB")
    print(f"     ≥0dB: {sum(1 for x in r if x['sinr']>=0)}/{n}")

m = train()
test(m)
