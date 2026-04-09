"""
ISAC v66 - 深度优化: 结合多种预编码方法
1. 尝试MRT (最大比传输) + 功率优化
2. 神经网络学习选择最佳预编码策略
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

M, K, P, Nt = 16, 4, 4, 4
Pmax = 30
N_req = 4

def mrt_precoding(H, Pmax):
    """MRT预编码 - 简单的信道匹配"""
    H_complex = H  # (M, K, Nt)
    
    w = np.zeros((M, K, Nt), dtype=complex)
    for k in range(K):
        # 信道共轭
        h_k = H_complex[:, k, :]  # (M, Nt)
        h_k_conj = np.conj(h_k)
        
        # 归一化
        norm = np.sqrt(np.sum(np.abs(h_k_conj) ** 2))
        if norm > 0:
            w[:, k, :] = h_k_conj / norm * np.sqrt(Pmax * 0.7 / K)
    
    return w

def mmse_precoding(H, Pmax, sigma2=0.5):
    """MMSE预编码"""
    H_complex = H
    H_stack = np.vstack([H_complex[m, :, :] for m in range(M)])
    
    HH = H_stack @ H_stack.conj().T
    HH_reg = HH + sigma2 * np.eye(M * Nt)
    
    try:
        HH_inv = np.linalg.inv(HH_reg)
        W_mmse = HH_inv @ H_stack
        power = np.sum(np.abs(W_mmse) ** 2)
        W_mmse = W_mmse * np.sqrt(Pmax * 0.65 / power)
        
        w = np.zeros((M, K, Nt), dtype=complex)
        for k in range(K):
            for m in range(M):
                w[m, k, :] = W_mmse[m*Nt:(m+1)*Nt, k]
        return w
    except:
        return mrt_precoding(H_complex, Pmax)

def compute_sinr(H, w):
    """计算SINR"""
    sinrs = []
    for k in range(K):
        signal = np.abs(np.sum(np.conj(w[:, k, :]) * H[:, k, :])) ** 2
        interference = sum(np.abs(np.sum(np.conj(w[:, j, :]) * H[:, k, :])) ** 2 for j in range(K) if j != k)
        sinrs.append(10 * np.log10(signal / (interference + 0.01) + 1e-8))
    return np.array(sinrs)

def generate_data(n_samples):
    """生成数据 - 测试不同预编码方法"""
    X_list, method_list = [], []
    
    for _ in range(n_samples):
        ap_pos = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        user_pos = np.random.uniform(-50, 50, (K, 2))
        
        H = np.zeros((M, K, Nt*2), dtype=np.float32)
        H_complex = np.zeros((M, K, Nt), dtype=complex)
        
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap_pos[m] - user_pos[k])**2)), 5)
                pl = (d / 10) ** -2
                h = np.sqrt(pl/2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
                H[m, k, :Nt] = np.real(h)
                H[m, k, Nt:] = np.imag(h)
                H_complex[m, k, :] = h
        
        # 测试不同预编码方法，选择最佳的
        methods = ['MRT', 'MMSE']
        best_method = None
        best_sinr = -100
        
        for method in methods:
            if method == 'MRT':
                w_test = mrt_precoding(H_complex, Pmax)
            else:
                w_test = mmse_precoding(H_complex, Pmax, sigma2=0.5)
            
            sinrs = compute_sinr(H_complex, w_test)
            min_sinr = min(sinrs)
            
            if min_sinr > best_sinr:
                best_sinr = min_sinr
                best_method = method
        
        # 编码方法 (0=MRI, 1=MMSE)
        method_id = 0 if best_method == 'MRT' else 1
        
        X_list.append(H.flatten())
        method_list.append(method_id)
    
    return np.array(X_list), np.array(method_list)

print("生成预编码选择数据...")
X, Method = generate_data(500)

class ISAC_v66(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M * K * Nt * 2, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 2), nn.Softmax(dim=1)  # 输出MRT/MMSE概率
        )
    
    def forward(self, x):
        return self.net(x)

def train_v66(epochs=3000, bs=32, lr=1e-4):
    model = ISAC_v66()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    Xt = torch.tensor(X, dtype=torch.float32)
    Mt = torch.tensor(Method, dtype=torch.long)
    
    for e in range(epochs):
        opt.zero_grad()
        idx = torch.randperm(len(X))[:bs]
        
        prob = model(Xt[idx])
        loss = F.cross_entropy(prob, Mt[idx]) * 50
        
        loss.backward()
        opt.step()
        
        if e % 600 == 0:
            print(f"Epoch {e}: loss={loss.item():.4f}")
    
    torch.save(model.state_dict(), 'isac_v66.pth')
    print("v66完成!")
    return model

def test_v66(model, n=100):
    model.eval()
    results = []
    
    for i in range(n):
        ap_pos = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        user_pos = np.random.uniform(-50, 50, (K, 2))
        
        H = np.zeros((M, K, Nt*2), dtype=np.float32)
        H_complex = np.zeros((M, K, Nt), dtype=complex)
        
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap_pos[m] - user_pos[k])**2)), 5)
                pl = (d / 10) ** -2
                h = np.sqrt(pl/2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
                H[m, k, :Nt] = np.real(h)
                H[m, k, Nt:] = np.imag(h)
                H_complex[m, k, :] = h
        
        # 网络选择预编码方法
        prob = model(torch.tensor(H.flatten().reshape(1, -1), dtype=torch.float32))
        method_id = prob.squeeze().argmax().item()
        
        if method_id == 0:
            w = mrt_precoding(H_complex, Pmax)
        else:
            w = mmse_precoding(H_complex, Pmax, sigma2=0.5)
        
        sinrs = compute_sinr(H_complex, w)
        results.append(min(sinrs))
    
    print(f"v66: SINR_min: {np.min(results):.2f}dB, 平均: {np.mean(results):.2f}dB")
    print(f"     ≥ 0dB: {sum(1 for r in results if r >= 0)}/{n}")

if __name__ == '__main__':
    model = train_v66()
    test_v66(model)
