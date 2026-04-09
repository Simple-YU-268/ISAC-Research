# ISAC波束赋形ML求解方法研究

## 研究背景

Cell-Free ISAC (通感一体) 联合优化问题:

$$\min_{W,Z,b} \sum_{m,k} ||w_{m,k}||^2 + Tr(Z_m)$$

约束:
- 通信 SINR ≥ γ_k
- 感知 SINR ≥ γ_S  
- 跟踪精度 FIM ≥ Γ
- AP选择 (二进制)
- 功率约束

传统方法: 凸优化 (CVX) 迭代求解，计算复杂

## ML求解方法

### 1. 无监督深度学习
- 输入: 信道 H, G
- 网络: 全连接/Transformer/GNN
- 输出: W, Z, b
- 损失: 通信SINR + 感知FIM + 功率约束

### 2. Teacher-Student
- 先用CVX训练Teacher
- Student学习两者平衡

### 3. 强化学习 (DRL)
- DDPG, PPO等算法
- 适合动态场景

## 参考文献

1. arXiv:2412.18162 - Unsupervised Teacher-Student (Cell-Free)
2. arXiv:2403.17324 - IBF-Net (RIS-ISAC)
3. arXiv:2603.23618 - Set Transformer
4. arXiv:2410.09963 - SACGNN
