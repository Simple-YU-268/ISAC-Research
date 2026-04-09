"""快速训练版本"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

M, K, P, Nt = 16, 10, 4, 4
sigma2 = 0.5
Pmax = 80

def mmse_comm(H, P_comm):
    M_sel = H.shape[0]
    Hs = H.reshape(M_sel * Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
    W = np.linalg.inv(HH) @ Hs
    p = np.sum(np.abs(W)**2)
    return W.reshape(M_sel, Nt, K) * np.sqrt(P_comm / p)

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
    return np.array([10*np.log10(np.abs(np.sum(np.conj(W_flat[:,k])@Hs[:,k]))**2/(sum(np.abs(np.sum(np.conj(W_flat[:,j])@Hs[:,k]))**2 for j in range(K) if j!=k)+sigma2)+1e-10) for k in range(K)])

def compute_sensing_snr(H_t, Z):
    return np.array([10*np.log10(sum(np.abs(np.sum(Z[m,p,:]*np.conj(H_t[m,p,:])))**2 for m in range(H_t.shape[0]))/(sigma2*np.sum(np.abs(Z)**2)+1e-10)+1e-10) for p in range(P)])

def generate():
    ap = np.array([[x,y] for x in np.linspace(-60,60,4) for y in np.linspace(-60,60,4)])
    u = np.random.uniform(-15,15,(K,2))
    t = np.random.uniform(-12,12,(P,2))
    H_u = np.array([[np.sqrt((max(np.sqrt((ap[m]-u[k])**2),5)/10)**-2.5/2)*(np.random.randn(Nt)+1j*np.random.randn(Nt)) for k in range(K)] for m in range(M)], dtype=complex)
    H_t = np.array([[np.sqrt((max(np.sqrt((ap[m]-t[p])**2),5)/10)**-2.5/2)*(np.random.randn(Nt)+1j*np.random.randn(Nt)) for p in range(P)] for m in range(M)], dtype=complex)
    return H_u, H_t

# 生成2000样本
X, Y = [], []
for i in range(2000):
    H_u, _ = generate()
    # 选择信道最强的5个AP
    sp = np.sum(np.abs(H_u)**2, axis=(2,3))
    ts = sp.sum(axis=1)
    top5 = np.argsort(-ts)[:5]
    label = np.zeros(M, dtype=np.float32)
    label[top5] = 1.0
    H_in = np.zeros((M,K,Nt*2), dtype=np.float32)
    H_in[:,:,:Nt] = np.real(H_u)
    H_in[:,:,Nt:] = np.imag(H_u)
    X.append(H_in.flatten())
    Y.append(label)

X, Y = np.array(X), np.array(Y)
print(f"数据: {len(X)} 样本")

# 网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(M*K*Nt*2,512), nn.LeakyReLU(0.2), nn.Linear(512,256), nn.LeakyReLU(0.2), nn.Linear(256,M), nn.Sigmoid())
    def forward(self,x): return self.fc(x)

m = Net()
o = torch.optim.AdamW(m.parameters(), lr=5e-5)
Xt, Yt = torch.tensor(X,dtype=torch.float32), torch.tensor(Y,dtype=torch.float32)

print("训练...")
for e in range(5000):
    o.zero_grad()
    idx = torch.randperm(len(X))[:64]
    loss = F.binary_cross_entropy(m(Xt[idx]), Yt[idx])*50 + ((m(Xt[idx]).sum(dim=1)-5).abs().mean()*10
    loss.backward(); o.step()
    if e%1000==0: print(f"Epoch {e}")

torch.save(m.state_dict(), 'isac_q.pth')
print("模型保存!")

# 测试
m.eval()
r = []
for _ in range(200):
    H_u, H_t = generate()
    H_in = np.zeros((M,K,Nt*2), dtype=np.float32)
    H_in[:,:,:Nt] = np.real(H_u); H_in[:,:,Nt:] = np.imag(H_u)
    top5 = np.argsort(-m(torch.tensor(H_in.flatten().reshape(1,-1),dtype=torch.float32).squeeze().detach().numpy()))[:5]
    am = np.zeros(M, dtype=bool); am[top5]=1
    W = mmse_comm(H_u[am], Pmax*0.75)
    Z = sensing_beam(H_t[am], Pmax*0.25)
    cs = compute_sinr(H_u[am], W)
    ss = compute_sensing_snr(H_t[am], Z)
    pw = np.sum(np.abs(W)**2)+np.sum(np.abs(Z)**2)
    r.append(all(s>=0 for s in cs) and all(s>=-22 for s in ss) and pw<=Pmax)

print(f"完全满足: {sum(r)}/200 ({sum(r)*100/200:.1f}%)")
