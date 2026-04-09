"""验证MMSE预编码的真实性能"""
import numpy as np

M, K, Nt = 16, 10, 4
Pmax = 30
sigma2 = 0.5

def gen_channel():
    ap = np.array([[x, y] for x in np.linspace(-60,60,4) for y in np.linspace(-60,60,4)])
    u = np.random.uniform(-50,50,(K,2))
    H = np.zeros((M,K,Nt), dtype=complex)
    for m in range(M):
        for k in range(K):
            d = max(np.sqrt(np.sum((ap[m]-u[k])**2)), 5)
            pl = (d/10)**-2.5
            H[m,k,:] = np.sqrt(pl)*(np.random.randn(Nt)+1j*np.random.randn(Nt))/np.sqrt(2)
    return H

def mmse_beam(H, Pmax):
    Hs = H.reshape(M*Nt, K)
    HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M*Nt)
    W = np.linalg.inv(HH) @ Hs
    p = np.sum(np.abs(W)**2)
    W = W * np.sqrt(Pmax / p)
    return W

def compute_sinr(H, W):
    Hs = H.reshape(M*Nt, K)
    sinrs = []
    for k in range(K):
        sig = np.abs(np.sum(np.conj(W[:,k]) @ Hs[:,k]))**2
        inter = sum(np.abs(np.sum(np.conj(W[:,j]) @ Hs[:,k]))**2 for j in range(K) if j!=k)
        sinrs.append(10*np.log10(sig/(inter+sigma2)+1e-10))
    return np.array(sinrs)

print("=== MMSE预编码测试 (sigma2=0.5, Pmax=30W) ===\n")

results = []
for trial in range(100):
    H = gen_channel()
    W = mmse_beam(H, Pmax)
    sinrs = compute_sinr(H, W)
    
    results.append({
        'min': np.min(sinrs),
        'mean': np.mean(sinrs),
        'all_ok': all(s >= 0 for s in sinrs),
        'ok_count': sum(s >= 0 for s in sinrs)
    })

print(f"100次测试结果:")
print(f"  SINR_min: 最小{np.min([r['min'] for r in results]):.2f}dB, 平均{np.mean([r['min'] for r in results]):.2f}dB")
print(f"  SINR_mean: 平均{np.mean([r['mean'] for r in results]):.2f}dB")
print(f"  全部10用户≥0dB: {sum(1 for r in results if r['all_ok'])}/100")
print(f"  平均满足用户数: {np.mean([r['ok_count'] for r in results]):.1f}/10")

# 不同功率测试
print("\n=== 不同功率测试 ===")
for pmax in [30, 50, 100, 200]:
    H = gen_channel()
    W = mmse_beam(H, pmax)
    sinrs = compute_sinr(H, W)
    print(f"Pmax={pmax}W: SINR_min={np.min(sinrs):.2f}dB, 满足{sum(s>=0 for s in sinrs)}/10用户")
