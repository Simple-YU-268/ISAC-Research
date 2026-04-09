# 无监督学习求解 Cell-Free ISAC 波束赋形

## 1. 问题背景

Cell-Free ISAC 联合优化问题:
- 变量: W (通信波束), Z (感知波束), b (AP选择)
- 约束: 通信SINR, 感知SINR, FIM跟踪精度, 功率限制
- 传统方法: CVX 迭代求解，计算复杂

## 2. 研究现状

### 无监督深度学习方法

| 方法 | 论文 | 特点 |
|------|------|------|
| Teacher-Student | arXiv:2412.18162 | 蒸馏CVX知识，性能接近 |
| IBF-Net | arXiv:2403.17324 | 图像化信道，FCN |
| Set Transformer | arXiv:2603.23618 | 分布式处理 |
| SACGNN | arXiv:2410.09963 | 异构图网络 |
| DRL (DDPG) | arXiv:2510.25496 | 动态优化 |

## 3. 研究空白

1. **Cell-Free 场景**: 大多数方法针对单基站，需要分布式
2. **二进制AP选择**: b∈{0,1} 难以用神经网络直接处理
3. **收敛稳定性**: 当前AO算法仍有震荡问题

## 4. Proposed Approach

### 方案: 无监督双阶段

**阶段1: 连续松弛**
- b ∈ [0,1] 连续松弛
- 神经网络输出 W, Z, b
- 损失函数包含所有约束违反

**阶段2: 二值化**
- 贪心 rounding
- 微调 W, Z

### 网络架构

```python
# 输入: H (M,K,Nt), G (M,M,P,Nt)
# 输出: W (M,K,Nt), Z (M,Nt,Nt), b (M,P)

class ISACNet(nn.Module):
    def __init__(self, M=16, K=8, P=4, Nt=4):
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(M*K*Nt + M*M*P*Nt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # 解码器 - W
        self.w_decoder = nn.Linear(256, M*K*Nt*2)
        
        # 解码器 - Z  
        self.z_decoder = nn.Linear(256, M*Nt*Nt*2)
        
        # 解码器 - b
        self.b_decoder = nn.Linear(256, M*P)
        
    def forward(self, H, G):
        x = torch.cat([H.flatten(), G.flatten()], dim=-1)
        h = self.encoder(x)
        
        # 复数转实数 (拼接实部和虚部)
        W = self.w_decoder(h).view(-1, M, K, Nt, 2)
        Z = self.z_decoder(h).view(-1, M, Nt, Nt, 2)
        b = torch.sigmoid(self.b_decoder(h))  # Sigmoid for [0,1]
        
        return W, Z, b
```

### 损失函数

```python
def loss_fn(W, Z, b, H, G, cfg):
    # 通信 SINR 损失
    comm_loss = max(0, gamma_k - SINR_comm)
    
    # 感知 SINR 损失  
    sens_loss = max(0, gamma_S - SINR_sens)
    
    # FIM 损失
    fim_loss = max(0, Gamma - FIM)
    
    # 功率损失
    pwr_loss = max(0, power - Pmax)
    
    # 正则化
    reg_loss = torch.sum(W**2) + torch.sum(Z**2)
    
    return comm_loss + sens_loss + fim_loss + pwr_loss + 0.01*reg_loss
```

## 5. 实验计划

1. 数据生成: 随机信道 + CVX 求解
2. 训练: 无监督端到端
3. 测试: 对比 CVX, AO Stable
4. 评估: 收敛速度, 精度, 泛化

## 6. 预期贡献

1. 提出适合 Cell-Free 的无监督神经网络
2. 解决 AP 二进制选择难题
3. 大幅降低推理时间 (1000x+)
