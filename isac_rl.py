"""ISAC 强化学习 - 解决AP选择问题"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

M, K, P, Nt = 16, 10, 4, 4
Pmax = 30
sigma2 = 0.5

# ============= 环境 =============
class ISACEnv:
    def __init__(self):
        self.state_dim = M * K * Nt * 2  # 信道状态
        self.action_dim = M  # 每个AP选或不选
    
    def reset(self):
        ap = np.array([[x, y] for x in np.linspace(-60, 60, 4) for y in np.linspace(-60, 60, 4)])
        user_pos = np.random.uniform(-30, 30, (K, 2))
        
        H = np.zeros((M, K, Nt), dtype=complex)
        for m in range(M):
            for k in range(K):
                d = max(np.sqrt(np.sum((ap[m] - user_pos[k])**2)), 5)
                pl = (d / 10)**-2.5
                H[m, k, :] = np.sqrt(pl / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        
        # 归一化信道
        H_real = np.zeros((M, K, Nt * 2), dtype=np.float32)
        H_real[:, :, :Nt] = np.real(H)
        H_real[:, :, Nt:] = np.imag(H)
        H_real = H_real / (np.abs(H_real).max() + 1e-6)
        
        self.H = H
        self.original_pos = user_pos
        return H_real.flatten()
    
    def step(self, action):
        """action: M维向量，0或1表示是否选择该AP"""
        ap_mask = action > 0.5
        M_sel = np.sum(ap_mask)
        
        if M_sel < 2:
            return self.reset(), -100, True, {}
        
        H_sel = self.H[ap_mask, :, :]
        
        # MMSE波束
        Hs = H_sel.reshape(M_sel * Nt, K)
        HH = Hs @ Hs.T.conj() + sigma2 * np.eye(M_sel * Nt)
        try:
            W = np.linalg.inv(HH) @ Hs
            p = np.sum(np.abs(W)**2)
            W = W * np.sqrt(Pmax * 0.8 / p)
        except:
            return self.reset(), -100, True, {}
        
        # 计算SINR
        sinrs = []
        for k in range(K):
            sig = np.abs(np.sum(np.conj(W[:, k]) @ Hs[:, k]))**2
            inter = sum(np.abs(np.sum(np.conj(W[:, j]) @ Hs[:, k]))**2 for j in range(K) if j != k)
            sinrs.append(10 * np.log10(sig / (inter + sigma2) + 1e-10))
        
        sinrs = np.array(sinrs)
        min_sinr = sinrs.min()
        
        # 奖励: SINR - 惩罚项
        reward = min_sinr * 10  # 缩放
        if min_sinr < 0:
            reward -= 50  # 负SINR严重惩罚
        else:
            reward += 20  # 正SINR奖励
        
        # 功率惩罚
        total_pwr = np.sum(np.abs(W)**2)
        if total_pwr > Pmax:
            reward -= 30
        
        done = min_sinr >= 0 or total_pwr > Pmax
        
        return None, reward, done, {'sinr_min': min_sinr, 'power': total_pwr, 'ap_count': M_sel}

# ============= 策略网络 =============
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, action_dim), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

# ============= 训练 =============
print("=== ISAC 强化学习训练 ===\n")

env = ISACEnv()
state_dim = M * K * Nt * 2
action_dim = M

policy = PolicyNetwork(state_dim, action_dim)
value_net = ValueNetwork(state_dim)

policy_opt = optim.Adam(policy.parameters(), lr=1e-4)
value_opt = optim.Adam(value_net.parameters(), lr=3e-4)

gamma = 0.99
episodes = 500
batch_size = 16

for ep in range(episodes):
    state = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32)
    
    episode_rewards = []
    episode_states = []
    episode_actions = []
    
    for t in range(10):  # 最多10步
        # 选择动作 (加一点探索)
        probs = policy(state_tensor)
        action = (probs > 0.5).float()  # 确定性策略
        
        # 如果探索: 随机扰动
        if np.random.random() < 0.3:
            action = torch.rand(action_dim) > 0.7
        
        _, reward, done, info = env.step(action.numpy())
        
        episode_states.append(state_tensor)
        episode_actions.append(action)
        episode_rewards.append(reward)
        
        if done:
            break
        
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)
    
    # 计算回报
    returns = []
    R = 0
    for r in reversed(episode_rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns, dtype=torch.float32)
    
    # 更新策略
    if len(returns) > 0:
        policy_opt.zero_grad()
        
        # 简化: 监督学习方式
        # 找到最优的action作为标签
        best_action = torch.zeros(action_dim)
        best_action[info['ap_select'] if 'ap_select' in info else torch.arange(M)[:info.get('ap_count', 3)]] = 1
        
        for s, a in zip(episode_states, episode_actions):
            # 最大化正奖励的动作概率
            if episode_rewards[0] > 0:
                loss = -((a - 0.5) ** 2).mean()  # 鼓励接近0或1
            else:
                loss = ((a - 0.5) ** 2).mean()  # 鼓励更均匀
        
        # 简化: 直接用交叉熵
        target_probs = torch.sigmoid(policy(torch.stack(episode_states)))
        # 让正奖励的样本更确定
        for i, (s, r) in enumerate(zip(episode_states, episode_rewards)):
            if r > 0:
                # 正确选择AP
                loss = ((target_probs[i] - 0.5) ** 2).mean() * 0.1
    
    if ep % 50 == 0:
        print(f"Episode {ep}: reward={np.mean(episode_rewards):.2f}")

# ============= 测试 =============
print("\n=== RL策略测试 ===")
results = []
for _ in range(100):
    state = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32)
    
    probs = policy(state_tensor)
    action = (probs > 0.5).float()
    
    _, _, _, info = env.step(action.numpy())
    results.append(info)

print(f"SINR_min: 最小{np.min([r['sinr_min'] for r in results]):.2f}dB")
print(f"功率≤30W: {sum(1 for r in results if r['power'] <= 30)}/100")
print(f"SINR≥0dB: {sum(1 for r in results if r['sinr_min'] >= 0)}/100")

torch.save(policy.state_dict(), 'isac_rl.pth')
