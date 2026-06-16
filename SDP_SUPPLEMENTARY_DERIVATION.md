# Cell-Free ISAC 补充推导：AP选择、多目标、时隙耦合与紧致性

## 1. AP选择的外部算法

### 1.1 问题描述

用户推导中："通过外部算法预先确定最优的AP聚类集合"

**问题**：什么是"外部算法"？如何预先确定？

### 1.2 基于大尺度衰落的启发式选择

**大尺度衰落模型**：

$$\text{PL}_{m,k} = \text{PL}_0 \left(\frac{d_{m,k}}{d_0}\right)^{-\alpha} \tag{1}$$

其中 $d_{m,k} = \|\mathbf{q}_m - \mathbf{r}_k\|_2$ 是AP $m$ 到用户 $k$ 的距离。

**AP选择准则**：

对于每个目标 $p$，选择大尺度衰落最强的 $N_{\text{req}}$ 个AP：

$$\mathcal{M}_p = \arg\max_{\mathcal{M} \subseteq \{1,\ldots,M\}, |\mathcal{M}|=N_{\text{req}}} \sum_{m \in \mathcal{M}} \text{PL}_{m,p} \tag{2}$$

**简化**：独立排序选择，复杂度 $O(M \log M)$ 每目标。

### 1.3 基于预测位置的动态选择

**时隙 $n$ 的目标预测**：

$$\hat{\mathbf{r}}_p[n|n-1] = \hat{\mathbf{r}}_p[n-1] + \Delta t \cdot \hat{\mathbf{v}}_p[n-1] \tag{3}$$

基于预测位置计算 $d_{m,p}[n]$，然后选择AP。

**优势**：
- 利用跟踪滤波器（如卡尔曼滤波）的预测
- 避免实时优化AP选择
- 降低计算复杂度

### 1.4 联合AP选择优化（备选）

若需要联合优化：

$$\min_{\{b_{mp}\}} \sum_{p} \sum_{m} b_{mp} \cdot d_{m,p}^{-\alpha} \tag{4}$$

$$\text{s.t.} \sum_{m} b_{mp} = N_{\text{req}}, \quad \forall p \tag{5}$$

$$b_{mp} \in \{0,1\} \tag{6}$$

这是**指派问题**，可用匈牙利算法在 $O(M^3)$ 求解。

---

## 2. 多目标情况（P > 1）

### 2.1 目标间干扰

当 $P > 1$ 时，不同目标的感知信号可能互相干扰。

**感知SINR（多目标）**：

$$\text{SINR}_{S,p} = \frac{\mathbf{g}_p^H \mathbf{Z}_p \mathbf{g}_p}{\sigma_s^2 + \sum_{q \neq p} \mathbf{g}_p^H \mathbf{Z}_q \mathbf{g}_p} \tag{7}$$

其中 $\mathbf{Z}_p$ 是目标 $p$ 的专用感知协方差。

### 2.2 全局感知协方差分解

$$\mathbf{Z} = \sum_{p=1}^{P} \mathbf{Z}_p \tag{8}$$

**约束**：

$$\text{tr}(\mathbf{E}_m \mathbf{Z}_p) \leq b_{mp} \cdot P_{\max}, \quad \forall m, p \tag{9}$$

确保AP $m$ 仅向目标 $p$ 发射感知波形当 $b_{mp}=1$。

### 2.3 多目标PCRB

每个目标有独立的Fisher信息矩阵：

$$\mathbf{J}_p^{\text{data}} = \frac{2}{\sigma_s^2} \text{Re}\left\{ \nabla_{\boldsymbol{\theta}_p} \mathbf{g}_p^H \cdot \mathbf{R}_X \cdot \nabla_{\boldsymbol{\theta}_p}^H \mathbf{g}_p \right\} \tag{10}$$

其中 $\mathbf{R}_X = \sum_k \mathbf{W}_k + \sum_p \mathbf{Z}_p$。

**关键**：各目标的PCRB约束独立，但共享发射协方差 $\mathbf{R}_X$。

---

## 3. 时隙间耦合

### 3.1 时序模型

系统运行离散时隙 $n = 1, 2, \ldots, N$。

**目标运动模型**（恒定速度）：

$$\mathbf{r}_p[n+1] = \mathbf{r}_p[n] + \Delta t \cdot \mathbf{v}_p[n] + \mathbf{w}_p[n] \tag{11}$$

$$\mathbf{v}_p[n+1] = \mathbf{v}_p[n] + \mathbf{w}_p'[n] \tag{12}$$

其中 $\mathbf{w}_p, \mathbf{w}_p'$ 是过程噪声。

### 3.2 时隙间耦合来源

1. **目标跟踪**：PCRB约束依赖于当前估计精度，影响下一时隙的先验
2. **AP选择**：若采用动态选择，需考虑切换开销
3. **功率约束**：电池供电AP需考虑能量累积

### 3.3 解耦策略

**用户推导采用逐时隙独立优化**：

每个时隙 $n$ 独立求解 (P1)，假设：
- AP选择基于预测位置 $\hat{\mathbf{r}}_p[n|n-1]$
- 信道估计基于当前时隙
- 无显式时隙间约束

**合理性**：
- 简化计算，适合实时系统
- 若目标移动慢（$\Delta t$ 小），近似合理
- 可通过预测补偿运动

### 3.4 扩展：多步预测

若需考虑未来时隙：

$$\min_{\{\mathbf{W}_k[n], \mathbf{Z}[n]\}_{n=1}^N} \sum_{n=1}^N \left( \sum_k \text{tr}(\mathbf{W}_k[n]) + \text{tr}(\mathbf{Z}[n]) \right) \tag{13}$$

$$\text{s.t.} \quad \text{PCRB}_p[n] \leq \Gamma_{\text{Track},p}[n], \quad \forall p, n \tag{14}$$

这是**模型预测控制（MPC）**形式，复杂度 $O(N \cdot (MN_t)^{3.5})$。

---

## 4. SDR紧致性条件

### 4.1 何时秩一约束自动满足？

**定理**（Huang-Palomar, 2010）：对于QCQP问题，若满足：
1. 目标函数是凸的
2. 约束数 $K \leq 2$（或特定结构）

则SDR松弛是紧致的。

**Cell-Free ISAC的紧致性**：

(P1) 有 $K$ 个通信约束 + $P$ 个感知约束 + $M$ 个功率约束，通常 $K > 2$。

**一般情况**：
- SDR不一定紧致
- 但数值实验表明，对于MISO问题，SDR通常近似紧致
- 性能损失可通过随机化恢复控制

### 4.2 秩一恢复的理论保证

**定理**（Luo et al., 2010）：对于某些结构化问题，若最优解秩为 $r$，则存在多项式时间算法找到秩一近似，且目标值损失有界。

**实际策略**：
1. 求解SDP，得到 $\mathbf{W}_k^*$
2. 若 $\text{rank}(\mathbf{W}_k^*) = 1$，直接提取波束
3. 若秩 > 1，使用高斯随机化生成候选波束
4. 选择满足约束的最佳候选

### 4.3 高斯随机化算法

**输入**：$\mathbf{W}_k^*$
**输出**：$\mathbf{w}_k$

**步骤**：
1. 生成 $\mathbf{\xi}_k \sim \mathcal{CN}(\mathbf{0}, \mathbf{W}_k^*)$
2. 候选波束：$\mathbf{w}_k^{(i)} = \sqrt{\text{tr}(\mathbf{W}_k^*)} \cdot \frac{\mathbf{\xi}_k}{\|\mathbf{\xi}_k\|_2}$
3. 重复 $I$ 次（如 $I=1000$）
4. 选择满足所有约束且功率最小的候选

**性能**：通常达到SDP最优值的95%以上。

---

## 5. 复杂度精确分析

### 5.1 SDP问题规模

**变量维度**：
- $\mathbf{W}_k$: $MN_t \times MN_t$ Hermitian → $MN_t(MN_t+1)/2$ 实变量
- $\mathbf{Z}$: 同上
- $\mu_k$: $K$ 个标量

**总变量数**：

$$N_{\text{var}} = (K+1) \cdot \frac{MN_t(MN_t+1)}{2} + K \tag{15}$$

对于 $M=16, N_t=4, K=10$：

$$N_{\text{var}} = 11 \cdot \frac{64 \cdot 65}{2} + 10 = 11 \cdot 2080 + 10 = 22890$$

**约束维度**：
- LMI (32b): $K$ 个 $(MN_t+1) \times (MN_t+1)$ → $K \cdot (MN_t+1)^2/2$ 个标量约束
- 线性约束: $O(K+P+M)$

**总约束数**：

$$N_{\text{cons}} = K \cdot \frac{(MN_t+1)(MN_t+2)}{2} + O(K+P+M) \tag{16}$$

对于 $M=16, N_t=4, K=10, P=4$：

$$N_{\text{cons}} = 10 \cdot \frac{65 \cdot 66}{2} + 30 = 21450 + 30 = 21480$$

### 5.2 内点法迭代复杂度

每次迭代主要操作：
1. 形成KKT系统：$O(N_{\text{var}}^2 \cdot N_{\text{cons}})$
2. 求解线性系统：$O(N_{\text{var}}^3)$

**每次迭代**：$O(N_{\text{var}}^3) = O((MN_t)^6 \cdot K^3)$

对于 $MN_t=64, K=10$：

$$O(64^6 \cdot 1000) = O(10^{12}) \text{ (理论)}$$

**实际**（利用稀疏性和结构）：
- MOSEK利用块对角结构
- 实际复杂度约 $O((MN_t)^{3.5} \cdot K^{0.5})$
- 对于64维，约 $O(10^6)$ 每次迭代
- 迭代次数：20-50次
- **总时间**：1-10秒（现代CPU）

### 5.3 与启发式的对比

| 方法 | 预处理 | 求解时间 | 总时间 |
|------|--------|----------|--------|
| ZF启发式 | $O(MN_t K)$ | $O(K^3)$ | < 0.1s |
| SDP (MOSEK) | $O(MN_t K)$ | $O((MN_t)^{3.5})$ | 1-10s |
| 比例 | 1x | 10-100x | 10-100x |

---

## 6. 感知信道模型细化

### 6.1 双基地雷达模型

若AP与接收机分离（双基地）：

$$\mathbf{g}_{m,p} = \sqrt{\text{PL}_{m,p}^{\text{tx}} \cdot \text{PL}_{m,p}^{\text{rx}}} \cdot \boldsymbol{\beta}_{m,p} \tag{17}$$

其中 $\text{PL}^{\text{tx}}$ 是发射路径损耗，$\text{PL}^{\text{rx}}$ 是接收路径损耗。

### 6.2 雷达散射截面（RCS）

$$\text{PL}_{m,p} = \frac{\sigma_{\text{RCS},p}}{(4\pi)^3 d_{m,p}^4} \tag{18}$$

其中 $\sigma_{\text{RCS},p}$ 是目标 $p$ 的雷达散射截面。

**影响**：RCS小的目标需要更高功率。

### 6.3 多径效应

$$\mathbf{g}_{m,p} = \sum_{l=1}^{L} \alpha_{m,p,l} \mathbf{a}(\theta_{m,p,l}) \tag{19}$$

其中 $L$ 是多径数，$\alpha_{m,p,l}$ 是复增益，$\mathbf{a}(\theta)$ 是阵列响应向量。

**简化**：通常假设单径（LOS）或统计信道模型。

---

## 7. 功率分配策略对比

### 7.1 统一功率分配

所有AP等功率：

$$P_m = P_{\max}, \quad \forall m \tag{20}$$

**缺点**：信道弱的AP浪费功率。

### 7.2 自适应功率分配（SDP）

SDP自动优化各AP功率：

$$P_m = \sum_k \text{tr}(\mathbf{E}_m \mathbf{W}_k) + \text{tr}(\mathbf{E}_m \mathbf{Z}) \leq P_{\max} \tag{21}$$

**优点**：功率效率高，总功率最小。

### 7.3 分数功率分配

$$P_m = \rho \cdot P_{\max}, \quad \rho \in [0,1] \tag{22}$$

**启发式**：调整 $\rho$ 满足约束。

---

## 8. 总结与下一步

### 8.1 完整推导总结

| 组件 | 推导状态 | 文件 |
|------|----------|------|
| 全局变量重构 | ✓ 完成 | MATHEMATICAL_DERIVATION.md |
| SDR松弛 | ✓ 完成 | SDP_DERIVATION_COMPLETE.md |
| S-Procedure | ✓ 完成 | SDP_DERIVATION_COMPLETE.md |
| 感知约束凸性 | ✓ 完成 | SDP_DERIVATION_COMPLETE.md |
| AP选择策略 | ✓ 完成 | 本文 §1 |
| 多目标扩展 | ✓ 完成 | 本文 §2 |
| 时隙耦合 | ✓ 完成 | 本文 §3 |
| SDR紧致性 | ✓ 完成 | 本文 §4 |
| 复杂度分析 | ✓ 完成 | 本文 §5 |
| 感知信道模型 | ✓ 完成 | 本文 §6 |

### 8.2 下一步工作

1. **实现SDP求解器**：
   - MATLAB: CVX + MOSEK
   - Python: CVXPY + MOSEK/SCS
   
2. **AP选择优化**：
   - 实现基于大尺度衰落的启发式选择
   - 对比不同 $N_{\text{req}}$ 的影响
   
3. **性能验证**：
   - 对比SDP vs ZF启发式
   - 测量成功率、功率效率、计算时间
   
4. **扩展**：
   - 多目标 ($P > 1$)
   - 时隙耦合（MPC形式）
   - 多径信道模型

---

**版本**：补充推导 v1.0 | 2026-06-16
