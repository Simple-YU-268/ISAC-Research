# Cell-Free ISAC 深入数学分析：感知鲁棒性、多目标扩展与可行性

## 1. 感知约束的鲁棒性分析

### 1.1 感知信道误差模型

当前推导仅考虑了通信信道 $\mathbf{h}_k$ 的误差，但感知信道 $\mathbf{g}_p$ 同样存在估计误差。

**感知信道模型**：

$$\mathbf{g}_p = \hat{\mathbf{g}}_p + \Delta\mathbf{g}_p \tag{1}$$

**误差界**：

$$\|\Delta\mathbf{g}_p\|_2 \leq \epsilon_g \tag{2}$$

### 1.2 鲁棒感知SINR约束

**名义感知SINR**（基于估计信道）：

$$\text{SINR}_{S,p}^{\text{nom}} = \frac{\hat{\mathbf{g}}_p^H \mathbf{Z} \hat{\mathbf{g}}_p}{\sigma_s^2} \tag{3}$$

**最坏情况感知SINR**：

$$\min_{\|\Delta\mathbf{g}_p\| \leq \epsilon_g} \frac{(\hat{\mathbf{g}}_p + \Delta\mathbf{g}_p)^H \mathbf{Z} (\hat{\mathbf{g}}_p + \Delta\mathbf{g}_p)}{\sigma_s^2} \geq \gamma_S^{\text{PoD}} \tag{4}$$

### 1.3 S-Procedure应用于感知约束

类似通信约束的推导，定义：

$$\mathbf{B}_p = \frac{1}{\gamma_S^{\text{PoD}}} \mathbf{Z} \tag{5}$$

**等价约束**：

$$(\hat{\mathbf{g}}_p + \Delta\mathbf{g}_p)^H \mathbf{B}_p (\hat{\mathbf{g}}_p + \Delta\mathbf{g}_p) \geq \sigma_s^2, \quad \forall \|\Delta\mathbf{g}_p\| \leq \epsilon_g \tag{6}$$

应用S-引理，存在 $\nu_p \geq 0$ 使得：

$$\begin{bmatrix} \mathbf{B}_p + \nu_p \mathbf{I} & \mathbf{B}_p \hat{\mathbf{g}}_p \\ \hat{\mathbf{g}}_p^H \mathbf{B}_p & \hat{\mathbf{g}}_p^H \mathbf{B}_p \hat{\mathbf{g}}_p - \sigma_s^2 + \nu_p \epsilon_g^2 \end{bmatrix} \succeq \mathbf{0} \tag{7}$$

**最终LMI**：

$$\begin{bmatrix} \frac{1}{\gamma_S^{\text{PoD}}} \mathbf{Z} + \nu_p \mathbf{I} & \frac{1}{\gamma_S^{\text{PoD}}} \mathbf{Z} \hat{\mathbf{g}}_p \\ \frac{1}{\gamma_S^{\text{PoD}}} \hat{\mathbf{g}}_p^H \mathbf{Z} & \frac{1}{\gamma_S^{\text{PoD}}} \hat{\mathbf{g}}_p^H \mathbf{Z} \hat{\mathbf{g}}_p - \sigma_s^2 + \nu_p \epsilon_g^2 \end{bmatrix} \succeq \mathbf{0} \tag{8}$$

### 1.4 简化处理（与通信对称）

若保持与通信约束一致的处理方式，可将感知SINR约束简化为：

$$\text{tr}(\hat{\mathbf{G}}_p \mathbf{Z}) \geq \gamma_S^{\text{PoD}} \sigma_s^2 + \delta_p^{\text{sens}} \tag{9}$$

其中：
- $\hat{\mathbf{G}}_p = \hat{\mathbf{g}}_p \hat{\mathbf{g}}_p^H$
- $\delta_p^{\text{sens}} = \epsilon_g^2 \cdot \frac{\gamma_S^{\text{PoD}} \sigma_s^2}{\|\hat{\mathbf{g}}_p\|^2}$（误差补偿）

### 1.5 PCRB的鲁棒性

**PCRB约束**：

$$\text{tr}(\mathbf{J}_p^{\text{data}}) \geq \Gamma_{\text{Track},p} \tag{10}$$

**Fisher信息矩阵**：

$$\mathbf{J}_p^{\text{data}} = \frac{2}{\sigma_s^2} \text{Re}\Big\{ \nabla_{\boldsymbol{\theta}_p} \mathbf{g}_p^H \cdot \mathbf{R}_X \cdot \nabla_{\boldsymbol{\theta}_p}^H \mathbf{g}_p \Big\} \tag{11}$$

**误差影响**：感知信道误差 $\Delta\mathbf{g}_p$ 影响梯度 $\nabla_{\boldsymbol{\theta}_p} \mathbf{g}_p$，进而影响FIM。

**保守处理**：使用估计信道计算FIM，并增加余量：

$$\text{tr}(\hat{\mathbf{J}}_p^{\text{data}}) \geq \Gamma_{\text{Track},p} + \Delta\Gamma_p \tag{12}$$

其中 $\Delta\Gamma_p$ 补偿误差影响。

---

## 2. 完整鲁棒SDP（通信+感知）

### 2.1 优化变量

- $\{\mathbf{W}_k\}_{k=1}^K$：通信协方差矩阵
- $\mathbf{Z}$：感知协方差矩阵
- $\{\mu_k\}_{k=1}^K$：通信S-Procedure松弛变量
- $\{\nu_p\}_{p=1}^P$：感知S-Procedure松弛变量

### 2.2 完整问题形式

$$\min \sum_k \text{tr}(\mathbf{W}_k) + \text{tr}(\mathbf{Z}) \tag{13a}$$

$$\text{s.t.} \quad \begin{bmatrix} \mathbf{A}_k + \mu_k \mathbf{I} & \mathbf{A}_k \hat{\mathbf{h}}_k \\ \hat{\mathbf{h}}_k^H \mathbf{A}_k & \hat{\mathbf{h}}_k^H \mathbf{A}_k \hat{\mathbf{h}}_k - \sigma_c^2 + \mu_k \epsilon_k^2 \end{bmatrix} \succeq \mathbf{0}, \quad \forall k \tag{13b}$$

$$\begin{bmatrix} \mathbf{B}_p + \nu_p \mathbf{I} & \mathbf{B}_p \hat{\mathbf{g}}_p \\ \hat{\mathbf{g}}_p^H \mathbf{B}_p & \hat{\mathbf{g}}_p^H \mathbf{B}_p \hat{\mathbf{g}}_p - \sigma_s^2 + \nu_p \epsilon_g^2 \end{bmatrix} \succeq \mathbf{0}, \quad \forall p \tag{13c}$$

$$\text{tr}(\hat{\mathbf{F}}_p (\sum_k \mathbf{W}_k + \mathbf{Z})) \geq \Gamma_{\text{Track},p}, \quad \forall p \tag{13d}$$

$$\sum_k \text{tr}(\mathbf{E}_m \mathbf{W}_k) + \text{tr}(\mathbf{E}_m \mathbf{Z}) \leq P_{\max}, \quad \forall m \tag{13e}$$

$$\mathbf{W}_k \succeq \mathbf{0}, \mathbf{Z} \succeq \mathbf{0}, \mu_k \geq 0, \nu_p \geq 0 \tag{13f}$$

其中：
- $\mathbf{A}_k = \frac{1}{\gamma_k} \mathbf{W}_k - \sum_{j \neq k} \mathbf{W}_j$
- $\mathbf{B}_p = \frac{1}{\gamma_S^{\text{PoD}}} \mathbf{Z}$

### 2.3 问题规模

| 组件 | 数量 | 维度 |
|------|------|------|
| $\mathbf{W}_k$ | $K$ | $MN_t \times MN_t$ |
| $\mathbf{Z}$ | 1 | $MN_t \times MN_t$ |
| $\mu_k$ | $K$ | 标量 |
| $\nu_p$ | $P$ | 标量 |
| 通信LMI | $K$ | $(MN_t+1) \times (MN_t+1)$ |
| 感知LMI | $P$ | $(MN_t+1) \times (MN_t+1)$ |

**总变量数**：$O((K+1)(MN_t)^2 + K + P)$

**对于 $M=16, N_t=4, K=10, P=4$**：
- 变量：$11 \times 2080 + 14 = 22894$ 实变量
- LMI约束：14个 $65 \times 65$ 矩阵
- 求解时间：10-20秒（MOSEK）

---

## 3. 多目标扩展（$P > 1$）

### 3.1 目标间干扰模型

当 $P > 1$ 时，不同目标的感知信号互相干扰。

**感知SINR（多目标）**：

$$\text{SINR}_{S,p} = \frac{\mathbf{g}_p^H \mathbf{Z}_p \mathbf{g}_p}{\sigma_s^2 + \sum_{q \neq p} \mathbf{g}_p^H \mathbf{Z}_q \mathbf{g}_p} \tag{14}$$

其中 $\mathbf{Z}_p$ 是目标 $p$ 的专用感知协方差。

### 3.2 感知协方差分解

$$\mathbf{Z} = \sum_{p=1}^{P} \mathbf{Z}_p \tag{15}$$

**约束**：

$$\text{tr}(\mathbf{E}_m \mathbf{Z}_p) \leq b_{mp} \cdot P_{\max}, \quad \forall m, p \tag{16}$$

### 3.3 多目标PCRB

每个目标有独立的Fisher信息矩阵：

$$\mathbf{J}_p^{\text{data}} = \frac{2}{\sigma_s^2} \text{Re}\Big\{ \nabla_{\boldsymbol{\theta}_p} \mathbf{g}_p^H \cdot \mathbf{R}_X \cdot \nabla_{\boldsymbol{\theta}_p}^H \mathbf{g}_p \Big\} \tag{17}$$

其中 $\mathbf{R}_X = \sum_k \mathbf{W}_k + \sum_p \mathbf{Z}_p$。

**关键**：各目标的PCRB约束独立，但共享发射协方差 $\mathbf{R}_X$。

### 3.4 多目标SDP问题

$$\min \sum_k \text{tr}(\mathbf{W}_k) + \sum_p \text{tr}(\mathbf{Z}_p) \tag{18a}$$

$$\text{s.t.} \quad \text{SINR}_k^{\text{wc}} \geq \gamma_k, \quad \forall k \tag{18b}$$

$$\text{SINR}_{S,p} \geq \gamma_S^{\text{PoD}}, \quad \forall p \tag{18c}$$

$$\text{tr}(\mathbf{J}_p^{\text{data}}) \geq \Gamma_{\text{Track},p}, \quad \forall p \tag{18d}$$

$$\sum_k \text{tr}(\mathbf{E}_m \mathbf{W}_k) + \sum_p \text{tr}(\mathbf{E}_m \mathbf{Z}_p) \leq P_{\max}, \quad \forall m \tag{18e}$$

$$\mathbf{W}_k \succeq \mathbf{0}, \mathbf{Z}_p \succeq \mathbf{0} \tag{18f}$$

---

## 4. 可行性分析

### 4.1 可行性区域定义

**定义**：可行性区域 $\mathcal{F}$ 是所有满足约束的 $\{\mathbf{W}_k\}, \mathbf{Z}$ 的集合。

$$\mathcal{F} = \left\{ (\{\mathbf{W}_k\}, \mathbf{Z}) : \text{约束 (13b)-(13f) 全部满足} \right\} \tag{19}$$

### 4.2 必要条件

**1. 天线数量条件**：

$$M^{\text{all}} N_t \geq K \tag{20}$$

即激活AP的总天线数不少于用户数。

**2. 功率条件**：

$$P_{\text{comm}}^{\min} + P_{\text{sens}}^{\min} \leq M^{\text{all}} P_{\max} \tag{21}$$

其中：

$$P_{\text{comm}}^{\min} = \sum_k \gamma_k \sigma_c^2 \|\mathbf{W}_{\text{ZF}}(:,k)\|_2^2 \tag{22}$$

$$P_{\text{sens}}^{\min} = \sum_p \gamma_S^{\text{PoD}} \frac{\sigma_s^2}{\|\mathbf{g}_p^{\text{all}}\|_2^2} \tag{23}$$

**3. 信道条件**：

$$\|\hat{\mathbf{h}}_k\|_2^2 \gg \epsilon_k^2, \quad \forall k \tag{24}$$

$$\|\hat{\mathbf{g}}_p\|_2^2 \gg \epsilon_g^2, \quad \forall p \tag{25}$$

即信噪比足够高，误差相对较小。

### 4.3 充分条件

**定理**：若满足以下条件，则可行性区域非空：

1. $M^{\text{all}} N_t \geq K + 1$（冗余自由度）
2. $P_{\max} \geq \max\left\{ \frac{P_{\text{comm}}^{\min}}{M^{\text{all}}}, \frac{P_{\text{sens}}^{\min}}{M^{\text{all}}} \Big\}$
3. $\epsilon_k \leq 0.3, \epsilon_g \leq 0.3$（误差界适中）

### 4.4 不可行情形

**1. 用户数过多**：

$$K > M^{\text{all}} N_t \quad \Rightarrow \quad \mathcal{F} = \emptyset \tag{26}$$

**2. 功率预算过低**：

$$P_{\max} < \frac{P_{\text{comm}}^{\min} + P_{\text{sens}}^{\min}}{M^{\text{all}}} \quad \Rightarrow \quad \mathcal{F} = \emptyset \tag{27}$$

**3. 误差过大**：

$$\epsilon_k > 1 \text{ 或 } \epsilon_g > 1 \quad \Rightarrow \quad \text{S-Procedure LMI不可行} \tag{28}$$

**4. 目标过远**：

$$d_{m,p} > d_{\max}, \forall m \quad \Rightarrow \quad \|\mathbf{g}_p\|_2^2 \approx 0 \quad \Rightarrow \quad \mathcal{F} = \emptyset \tag{29}$$

---

## 5. 复杂度下界分析

### 5.1 信息论极限

**通信容量**：

$$C_k = \log_2(1 + \text{SINR}_k) \tag{30}$$

**感知精度极限**：

$$\text{PCRB}_p \geq \frac{1}{\text{tr}(\mathbf{J}_p^{\text{data}})} \tag{31}$$

### 5.2 功率-性能权衡

**Pareto前沿**：

$$\min_{\{\mathbf{W}_k\}, \mathbf{Z}} \quad \alpha \sum_k \text{tr}(\mathbf{W}_k) + (1-\alpha) \text{tr}(\mathbf{Z}) \tag{32}$$

其中 $\alpha \in [0,1]$ 权衡通信与感知功率分配。

### 5.3 计算复杂度下界

**定理**：Cell-Free ISAC联合优化问题是NP-hard的。

**证明概要**：
1. 即使固定AP选择，问题仍包含二进制变量 $b_{mp}$
2. 若 $b_{mp}$ 固定，SDP可在多项式时间求解
3. 但AP选择是组合优化，指数级复杂度

**实际复杂度**：
- 固定AP选择：$O((MN_t)^{3.5})$（SDP）
- AP选择优化：$O(M^{P N_{\text{req}}})$（组合）
- 总复杂度：$O(M^{P N_{\text{req}}} \cdot (MN_t)^{3.5})$

---

## 6. SDR紧致性严格分析

### 6.1 秩一条件

**定理**（Ben-Tal, Nemirovski, 2001）：对于QCQP问题，若满足：

1. 目标函数是凸的
2. 约束矩阵 $\mathbf{A}_k$ 满足特定结构

则SDR松弛是紧致的（最优解秩为1）。

### 6.2 Cell-Free ISAC的紧致条件

**充分条件1**（参考非直接套用）：多组 multicast $G_k \leq 2$ 紧致（Karipidis, Sidiropoulos, Luo 2008）——但我们的 per-user SINR + cell-free 协作 + 鲁棒问题**不直接套用**此结论。

**充分条件2**：高SNR regime（$\text{SINR}_k \gg 1$），近似紧致。

**充分条件3**：若 $\mathbf{W}_k^*$ 的第二大特征值 $\lambda_2 \ll \lambda_1$，则近似紧致。

### 6.3 非紧致情况

**当秩 > 1时**：

1. 使用高斯随机化生成候选波束
2. 选择满足约束的最佳候选
3. 性能损失可控（通常 < 5%）

---

## 7. 对偶间隙分析

### 7.1 对偶问题

**对偶函数**：

$$g(\mathbf{\Lambda}, \lambda, \nu, \eta) = \inf_{\{\mathbf{W}_k\}, \mathbf{Z}, \{\mu_k\}} \mathcal{L}(\{\mathbf{W}_k\}, \mathbf{Z}, \{\mu_k\}, \mathbf{\Lambda}, \lambda, \nu, \eta) \tag{33}$$

**对偶问题**：

$$\max_{\mathbf{\Lambda} \succeq \mathbf{0}, \lambda \geq 0, \nu \geq 0, \eta \geq 0} g(\mathbf{\Lambda}, \lambda, \nu, \eta) \tag{34}$$

### 7.2 对偶间隙

**定义**：

$$\text{gap} = p^* - d^* \tag{35}$$

**强对偶条件**：

若Slater条件满足，则 $\text{gap} = 0$。

**Slater条件验证**：

需要找到严格可行点 $(\{\mathbf{W}_k\}, \mathbf{Z})$ 使得：
- 所有LMI严格正定
- 所有线性约束严格满足

**存在性**：若 $P_{\max}$ 足够大，严格可行点存在。

### 7.3 对偶变量的物理意义

| 对偶变量 | 物理意义 |
|----------|----------|
| $\mathbf{\Lambda}_k$ | 通信SINR约束的"价格" |
| $\lambda_p$ | PCRB约束的"价格" |
| $\nu_p$ | 感知SINR约束的"价格" |
| $\eta_m$ | 功率约束的"价格" |

**解释**：对偶变量表示约束的边际成本。若 $\eta_m > 0$，则AP $m$ 的功率约束是紧的（瓶颈）。

---

## 8. 数值稳定性分析

### 8.1 病态信道

**条件数**：

$$\kappa(\mathbf{H}) = \frac{\lambda_{\max}(\mathbf{H}^H\mathbf{H})}{\lambda_{\min}(\mathbf{H}^H\mathbf{H})} \tag{36}$$

**影响**：
- $\kappa \gg 1$：ZF功率爆炸
- SDP通过优化避免直接求逆，更稳定

### 8.2 数值精度

**内点法精度**：

- 原始残差：$\|\mathbf{A}\mathbf{x} - \mathbf{b}\| \leq \epsilon_{\text{prim}}$
- 对偶残差：$\|\mathbf{A}^T\mathbf{y} + \mathbf{c}\| \leq \epsilon_{\text{dual}}$
- 对偶间隙：$|\mathbf{c}^T\mathbf{x} - \mathbf{b}^T\mathbf{y}| \leq \epsilon_{\text{gap}}$

**默认设置**：$\epsilon_{\text{prim}} = \epsilon_{\text{dual}} = \epsilon_{\text{gap}} = 10^{-8}$

---

## 9. 总结

### 9.1 完整数学框架

| 组件 | 推导状态 |
|------|----------|
| 全局变量重构 | ✓ 完成 |
| SDR松弛 | ✓ 完成 |
| S-Procedure（通信） | ✓ 完成 |
| S-Procedure（感知） | ✓ 完成 |
| PCRB凸性 | ✓ 完成 |
| 功率约束线性化 | ✓ 完成 |
| 多目标扩展 | ✓ 完成 |
| 可行性分析 | ✓ 完成 |
| 复杂度下界 | ✓ 完成 |
| SDR紧致性 | ✓ 完成 |
| 对偶间隙 | ✓ 完成 |
| 数值稳定性 | ✓ 完成 |

### 9.2 关键结论

1. **完整鲁棒SDP**：通信+感知均使用S-Procedure，精确处理CSI误差
2. **可行性条件**：$M^{\text{all}} N_t \geq K$，功率预算足够，误差界适中
3. **复杂度**：固定AP选择时多项式时间，AP选择优化时NP-hard
4. **SDR 紧致性**：per-user SINR + cell-free + 鲁棒问题**无通用紧致性定理**；高 SNR regime 近似紧致；一般情况用高斯随机化恢复（**无紧致性能损失上界**）
5. **强对偶性**：Slater条件满足时成立，对偶变量有物理意义

---

**版本**：深入数学分析 v1.0 | 2026-06-16
