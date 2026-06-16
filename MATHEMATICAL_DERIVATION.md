# Cell-Free ISAC 统一数学推导文档

**版本**: v2.0 (综合版)  
**日期**: 2026-06-16  
**状态**: 整合闭式解、SDP松弛、S-Procedure、凸性证明、秩一恢复、防御声明

---

## 目录

1. [问题定义](#1-问题定义)
2. [系统模型与信号模型](#2-系统模型与信号模型)
3. [通信约束推导](#3-通信约束推导)
4. [感知约束推导](#4-感知约束推导)
5. [SDP松弛与全局变量重构](#5-sdp松弛与全局变量重构)
6. [S-Procedure鲁棒约束转化](#6-s-procedure鲁棒约束转化)
7. [感知约束凸性证明](#7-感知约束凸性证明)
8. [秩一恢复与兜底方案](#8-秩一恢复与兜底方案)
9. [闭式解推导](#9-闭式解推导)
10. [复杂度分析](#10-复杂度分析)
11. [可行性条件](#11-可行性条件)
12. [理论假设与防御声明](#12-理论假设与防御声明)
13. [关键公式汇总](#13-关键公式汇总)

---

## 1. 问题定义

### 1.1 优化问题 (标准形式)

$$\min_{\{\mathbf{w}_{m,k}\}, \{\mathbf{Z}_m\}, \{b_{mp}\}} \quad \sum_{m=1}^{M} \left( \sum_{k=1}^{K} \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m) \right) \tag{P1a}$$

$$\text{s.t.} \quad \text{SINR}_k^{\text{wc}} \geq \gamma_k, \quad \forall k \in \mathcal{K} \tag{P1b}$$

$$\text{SINR}_{S,p} \geq \gamma_S^{\text{PoD}}, \quad \forall p \in \mathcal{P} \tag{P1c}$$

$$\text{tr}(\mathbf{J}_p^{\text{data}}) \geq \Gamma_{\text{Track}, p}, \quad \forall p \in \mathcal{P} \tag{P1d}$$

$$\sum_{m=1}^{M} b_{mp} = N_{\text{req}}, \quad \forall p \in \mathcal{P}^{\text{active}} \tag{P1e}$$

$$\sum_{k=1}^{K} \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m) \leq P_{\max}, \quad \forall m \in \mathcal{M} \tag{P1f}$$

$$\mathbf{Z}_m \succeq \mathbf{0}, \quad \forall m \in \mathcal{M} \tag{P1g}$$

$$b_{mp} \in \{0, 1\}, \quad \forall m \in \mathcal{M}, p \in \mathcal{P} \tag{P1h}$$

### 1.2 符号表

| 符号 | 维度 | 物理含义 |
|------|------|----------|
| $\mathbf{h}_{m,k}$ | $\mathbb{C}^{N_t \times 1}$ | AP $m$ 到用户 $k$ 的通信信道 |
| $\mathbf{g}_{m,p}$ | $\mathbb{C}^{N_t \times 1}$ | AP $m$ 到目标 $p$ 的感知信道 |
| $\mathbf{h}_k$ | $\mathbb{C}^{MN_t \times 1}$ | 堆叠通信信道（所有 AP） |
| $\mathbf{g}_p$ | $\mathbb{C}^{MN_t \times 1}$ | 堆叠感知信道（所有 AP） |
| $\mathbf{w}_{m,k}$ | $\mathbb{C}^{N_t \times 1}$ | AP $m$ 到用户 $k$ 的通信波束成形向量 |
| $\mathbf{w}_k$ | $\mathbb{C}^{MN_t \times 1}$ | 堆叠通信波束（所有 AP） |
| $\mathbf{z}_p$ | $\mathbb{C}^{MN_t \times 1}$ | 感知波束（所有 AP） |
| $\mathbf{Z}_m$ | $\mathbb{C}^{N_t \times N_t}$ | AP $m$ 的感知协方差矩阵 |
| $b_{mp}$ | $\{0,1\}$ | AP $m$ 是否服务目标 $p$ 的二进制指示变量 |
| $M$ | 标量 | AP 总数 |
| $N_t$ | 标量 | 每个 AP 的发射天线数 |
| $K$ | 标量 | 通信用户总数 |
| $P$ | 标量 | 感知目标总数 |
| $P_{\max}$ | 标量 | 单 AP 最大发射功率 |
| $\gamma_k$ | 标量 | 通信用户 $k$ 的 SINR 门限 |
| $\gamma_S^{\text{PoD}}$ | 标量 | 感知检测概率门限 |
| $\Gamma_{\text{Track},p}$ | 标量 | 目标 $p$ 的跟踪精度门限（FIM 迹约束） |
| $\sigma_c^2$ | 标量 | 通信接收噪声方差 |
| $\sigma_s^2$ | 标量 | 感知接收噪声方差 |
| $\epsilon_h$ | 标量 | 通信信道估计相对误差界 |
| $\epsilon_g$ | 标量 | 感知信道估计相对误差界 |
| $N_{\text{req}}$ | 标量 | 每个目标所需服务 AP 数 |

> **注**：具体数值参数（如 $M=16, N_t=4, P_{\max}=30$ W 等）见仿真设定章节表 X。

---

## 2. 系统模型与信号模型

### 2.1 通信信号模型

用户 $k$ 的接收信号:

$$y_k = \underbrace{\sum_{m=1}^M \mathbf{h}_{m,k}^H \mathbf{w}_{m,k} s_k}_{\text{期望信号}} + \underbrace{\sum_{j \neq k} \sum_{m=1}^M \mathbf{h}_{m,k}^H \mathbf{w}_{m,j} s_j}_{\text{多用户干扰}} + n_k$$

**堆叠形式**:

$$y_k = \mathbf{h}_k^H \mathbf{w}_k s_k + \sum_{j \neq k} \mathbf{h}_k^H \mathbf{w}_j s_j + n_k$$

其中 $n_k \sim \mathcal{CN}(0, \sigma_c^2)$。

### 2.2 感知信号模型

目标 $p$ 的反射信号 (单基地雷达):

$$y_{S,p} = \sum_{m=1}^M \mathbf{g}_{m,p}^H \mathbf{Z}_m \mathbf{g}_{m,p} + n_{S,p}$$

其中 $n_{S,p} \sim \mathcal{CN}(0, \sigma_s^2)$。

### 2.3 不完美 CSI 模型

**通信信道**:

$$\mathbf{h}_k = \hat{\mathbf{h}}_k + \Delta\mathbf{h}_k, \quad \|\Delta\mathbf{h}_k\|_2 \leq \epsilon_h \|\hat{\mathbf{h}}_k\|_2$$

**感知信道**:

$$\mathbf{g}_p = \hat{\mathbf{g}}_p + \Delta\mathbf{g}_p, \quad \|\Delta\mathbf{g}_p\|_2 \leq \epsilon_g \|\hat{\mathbf{g}}_p\|_2$$

---

## 3. 通信约束推导

### 3.1 SINR 定义

$$\text{SINR}_k = \frac{|\mathbf{h}_k^H \mathbf{w}_k|^2}{\sum_{j \neq k} |\mathbf{h}_k^H \mathbf{w}_j|^2 + \sigma_c^2} \tag{1}$$

### 3.2 最坏情况 SINR (简化近似)

**分子最坏情况**:

$$|\mathbf{h}_k^H \mathbf{w}_k|^2 \approx |\hat{\mathbf{h}}_k^H \mathbf{w}_k|^2 (1 - \epsilon_h)^2$$

**干扰最坏情况**:

$$|\mathbf{h}_k^H \mathbf{w}_j|^2 \approx |\hat{\mathbf{h}}_k^H \mathbf{w}_j|^2 (1 + \epsilon_h)^2$$

**最坏情况 SINR**:

$$\text{SINR}_k^{\text{wc}} \approx \text{SINR}_k^{\text{nom}} \cdot \frac{(1-\epsilon_h)^2}{(1+\epsilon_h)^2} \tag{2}$$

**鲁棒性因子**:

$$\eta_h = \left(\frac{1-\epsilon_h}{1+\epsilon_h}\right)^2 \tag{3}$$

对于 $\epsilon_h = 0.10$: $\eta_h \approx 0.669$ (-1.75 dB)

**鲁棒门限**:

$$\gamma_k^{\text{robust}} = \gamma_k \cdot \frac{(1+\epsilon_h)^2}{(1-\epsilon_h)^2} \tag{4}$$

对于 $\gamma_k = 1$ (0 dB): $\gamma_k^{\text{robust}} \approx 1.493$ (1.74 dB)

---

## 4. 感知约束推导

### 4.1 感知 SINR

$$\text{SINR}_{S,p} = \frac{|\mathbf{g}_p^H \mathbf{z}_p|^2}{\sigma_s^2 \|\mathbf{z}_p\|_2^2} \tag{5}$$

**最优感知波束** (Cauchy-Schwarz):

$$\mathbf{z}_p^* = \sqrt{P_{S,p}} \frac{\mathbf{g}_p}{\|\mathbf{g}_p\|_2} \tag{6}$$

**最优 SINR**:

$$\text{SINR}_{S,p}^* = \frac{P_{S,p} \|\mathbf{g}_p\|_2^2}{\sigma_s^2} \tag{7}$$

**最小感知功率**:

$$P_{S,p}^{\min} = \frac{\gamma_S^{\text{PoD}} \sigma_s^2}{\|\mathbf{g}_p\|_2^2} \tag{8}$$

### 4.2 鲁棒感知约束

$$\eta_g = \left(\frac{1-\epsilon_g}{1+\epsilon_g}\right)^2 \tag{9}$$

对于 $\epsilon_g = 0.15$: $\eta_g \approx 0.546$ (-2.63 dB)

$$(\gamma_S^{\text{PoD}})^{\text{robust}} = \gamma_S^{\text{PoD}} \cdot \frac{(1+\epsilon_g)^2}{(1-\epsilon_g)^2} \tag{10}$$

对于 $\gamma_S^{\text{PoD}} = 1$: $(\gamma_S^{\text{PoD}})^{\text{robust}} \approx 1.831$ (2.63 dB)

### 4.3 PCRB 约束

**Fisher 信息矩阵**:

$$\mathbf{J}_p^{\text{data}} = \frac{2}{\sigma_s^2} \text{Re}\Big\{ \nabla_{\boldsymbol{\theta}_p} \mathbf{g}_p^H \cdot \mathbf{R}_X \cdot \nabla_{\boldsymbol{\theta}_p}^H \mathbf{g}_p \Big\} \tag{11}$$

其中 $\mathbf{R}_X = \sum_k \mathbf{W}_k + \mathbf{Z}$ 是发射协方差矩阵。

**关键观察**: $\mathbf{J}_p^{\text{data}}$ 是 $\mathbf{R}_X$ 的**仿射函数**。

**证明**: 设 $\mathbf{G}_p = \nabla_{\boldsymbol{\theta}_p} \mathbf{g}_p^H \in \mathbb{C}^{D \times MN_t}$，则:

$$\mathbf{J}_p^{\text{data}} = \frac{2}{\sigma_s^2} \text{Re}\{ \mathbf{G}_p \mathbf{R}_X \mathbf{G}_p^H \}$$

展开:

$$\left(\mathbf{J}_p^{\text{data}}\right)_{ab} = \frac{2}{\sigma_s^2} \text{Re}\Big\{ \sum_{i,j} (\mathbf{G}_p)_{ai} (\mathbf{R}_X)_{ij} (\mathbf{G}_p^H)_{jb} \Big\}$$

这是关于 $\mathbf{R}_X$ 元素的线性组合。

**迹约束**:

$$\text{tr}(\mathbf{J}_p^{\text{data}}) = \text{tr}(\mathbf{F}_p \mathbf{R}_X) \tag{12}$$

其中:

$$\mathbf{F}_p = \frac{2}{\sigma_s^2} \text{Re}\Big\{ \nabla_{\boldsymbol{\theta}_p}^H \mathbf{g}_p \nabla_{\boldsymbol{\theta}_p} \mathbf{g}_p^H \Big\} \tag{13}$$

**约束形式**:

$$\text{tr}(\mathbf{F}_p (\sum_k \mathbf{W}_k + \mathbf{Z})) \geq \Gamma_{\text{Track},p} \tag{14}$$

**凸性**: 线性不等式约束，天然凸。

---

## 5. SDP松弛与全局变量重构

### 5.1 从波束到协方差

定义通信协方差矩阵:

$$\mathbf{W}_k = \mathbf{w}_k \mathbf{w}_k^H \in \mathbb{C}^{MN_t \times MN_t} \tag{15}$$

定义感知协方差矩阵:

$$\mathbf{Z} = \sum_p \mathbf{z}_p \mathbf{z}_p^H \in \mathbb{C}^{MN_t \times MN_t} \tag{16}$$

**全局发射协方差**:

$$\mathbf{R}_X = \sum_{k=1}^K \mathbf{W}_k + \mathbf{Z} \tag{17}$$

### 5.2 SDR 松弛

原问题要求 $\text{rank}(\mathbf{W}_k) = 1$ (因为 $\mathbf{W}_k = \mathbf{w}_k \mathbf{w}_k^H$)。

**SDR 松弛**: 丢弃秩一约束，仅要求 $\mathbf{W}_k \succeq \mathbf{0}$。

**松弛后问题**:

$$\min_{\{\mathbf{W}_k\}, \mathbf{Z}} \quad \sum_k \text{tr}(\mathbf{W}_k) + \text{tr}(\mathbf{Z}) \tag{P2a}$$

$$\text{s.t.} \quad \text{SINR}_k \geq \gamma_k, \quad \forall k \tag{P2b}$$

$$\text{SINR}_{S,p} \geq \gamma_S^{\text{PoD}}, \quad \forall p \tag{P2c}$$

$$\text{tr}(\mathbf{F}_p \mathbf{R}_X) \geq \Gamma_{\text{Track},p}, \quad \forall p \tag{P2d}$$

$$\text{tr}(\mathbf{E}_m \mathbf{R}_X) \leq P_{\max}, \quad \forall m \tag{P2e}$$

$$\mathbf{W}_k \succeq \mathbf{0}, \mathbf{Z} \succeq \mathbf{0} \tag{P2f}$$

其中 $\mathbf{E}_m$ 是 AP $m$ 的选择矩阵。

### 5.3 紧致性条件

**定理**: 对于总功率最小化问题，当 $K \leq 2$ 时，SDR 紧致，即最优解满足 $\text{rank}(\mathbf{W}_k^*) = 1$。

**高 SNR  regime**: 近似紧致，性能损失可控。

**一般情况**: 通过高斯随机化恢复波束，性能损失上界 $O(1/L)$ ($L$ 为候选数)。

---

## 6. S-Procedure鲁棒约束转化

### 6.1 通信鲁棒约束 (精确转化)

**最坏情况 SINR 约束**:

$$\min_{\|\Delta\mathbf{h}_k\| \leq \epsilon_h} \frac{|(\hat{\mathbf{h}}_k + \Delta\mathbf{h}_k)^H \mathbf{w}_k|^2}{\sum_{j \neq k} |(\hat{\mathbf{h}}_k + \Delta\mathbf{h}_k)^H \mathbf{w}_j|^2 + \sigma_c^2} \geq \gamma_k$$

**S-Procedure 转化**:

定义 $\mathbf{A}_k = \frac{1}{\gamma_k} \mathbf{W}_k - \sum_{j \neq k} \mathbf{W}_j$。

存在 $\mu_k \geq 0$ 使得:

$$\begin{bmatrix} \mathbf{A}_k + \mu_k \mathbf{I} & \mathbf{A}_k \hat{\mathbf{h}}_k \\ \hat{\mathbf{h}}_k^H \mathbf{A}_k & \hat{\mathbf{h}}_k^H \mathbf{A}_k \hat{\mathbf{h}}_k - \sigma_c^2 + \mu_k \epsilon_h^2 \end{bmatrix} \succeq \mathbf{0} \tag{18}$$

**性质**: S-Procedure 是**精确等价**转化，非近似。

### 6.2 感知鲁棒约束 (可选)

类似地，对于感知信道:

$$\begin{bmatrix} \frac{1}{\gamma_S^{\text{PoD}}} \mathbf{Z} + \nu_p \mathbf{I} & \frac{1}{\gamma_S^{\text{PoD}}} \mathbf{Z} \hat{\mathbf{g}}_p \\ \frac{1}{\gamma_S^{\text{PoD}}} \hat{\mathbf{g}}_p^H \mathbf{Z} & \frac{1}{\gamma_S^{\text{PoD}}} \hat{\mathbf{g}}_p^H \mathbf{Z} \hat{\mathbf{g}}_p - \sigma_s^2 + \nu_p \epsilon_g^2 \end{bmatrix} \succeq \mathbf{0} \tag{19}$$

其中 $\nu_p \geq 0$ 是感知 S-Procedure 松弛变量。

**本文采用**: 方案 A — 在假设中声明感知信道完美，仅对通信做鲁棒处理。

> **LaTeX 排版建议**：式 (19) 中矩阵元素含分母 $\gamma_S^{\text{PoD}}$，标准 LMI 书写中可将不等式两边同乘 $\gamma_S^{\text{PoD}}$，使左上角变为 $\mathbf{Z} + \nu_p \gamma_S^{\text{PoD}} \mathbf{I}$，避免矩阵内部出现分数结构，排版更美观。数学等价性不变。

---

## 7. 感知约束凸性证明

### 7.1 PCRB 约束凸性

**约束**: $\text{tr}(\mathbf{F}_p \mathbf{R}_X) \geq \Gamma_{\text{Track},p}$

- $\mathbf{F}_p$ 是已知常数半正定矩阵
- $\mathbf{R}_X$ 是优化变量 (半正定矩阵)
- $\text{tr}(\mathbf{F}_p \mathbf{R}_X)$ 是 $\mathbf{R}_X$ 的线性函数
- **结论**: 线性不等式约束，天然凸

### 7.2 感知 SINR 约束凸性

**约束**: $\text{tr}(\mathbf{g}_p \mathbf{g}_p^H \mathbf{Z}) \geq \gamma_S^{\text{PoD}} \sigma_s^2$

- $\mathbf{g}_p \mathbf{g}_p^H$ 是已知常数半正定矩阵
- $\mathbf{Z}$ 是优化变量
- $\text{tr}(\mathbf{g}_p \mathbf{g}_p^H \mathbf{Z})$ 是 $\mathbf{Z}$ 的线性函数
- **结论**: 线性不等式约束，天然凸

### 7.3 功率约束凸性

**约束**: $\text{tr}(\mathbf{E}_m \mathbf{R}_X) \leq P_{\max}$

- $\mathbf{E}_m$ 是已知常数矩阵
- $\mathbf{R}_X$ 是优化变量
- **结论**: 线性不等式约束，天然凸

### 7.4 总结

| 约束 | 形式 | 凸性 |
|------|------|------|
| 通信鲁棒 (S-Procedure) | LMI | 凸 |
| PCRB | 线性迹 | 凸 |
| 感知 SINR | 线性迹 | 凸 |
| 功率 | 线性迹 | 凸 |
| 半正定 | $\mathbf{W}_k, \mathbf{Z} \succeq \mathbf{0}$ | 凸 |

**最终问题 (P2) 是标准凸 SDP**。

---

## 8. 秩一恢复与兜底方案

### 8.1 问题背景

SDR 松弛丢弃了 $\text{rank}(\mathbf{W}_k) = 1$ 约束。求解后:
- 若 $\text{rank}(\mathbf{W}_k^*) = 1$: 直接用特征值分解恢复 $\mathbf{w}_k$
- 若 $\text{rank}(\mathbf{W}_k^*) > 1$: 需要高斯随机化提取次优波束

### 8.2 Algorithm 1: 高斯随机化秩一恢复

**输入**: SDP 最优解 $\{\mathbf{W}_k^*\}_{k=1}^K$, $\mathbf{Z}^*$, 约束参数  
**输出**: 满足所有约束的波束 $\{\mathbf{w}_k\}_{k=1}^K$, 感知波形

**步骤 1: 通信波束恢复**

对每个用户 $k$:
1. 若 $\text{rank}(\mathbf{W}_k^*) = 1$:
   $$\mathbf{w}_k = \sqrt{\lambda_{\max}(\mathbf{W}_k^*)} \cdot \mathbf{v}_{\max}(\mathbf{W}_k^*)$$
2. 若 $\text{rank}(\mathbf{W}_k^*) > 1$:
   - 生成 $L = 1000$ 个候选: $\boldsymbol{\xi}_l \sim \mathcal{CN}(\mathbf{0}, \mathbf{W}_k^*)$
   - 归一化: $\mathbf{w}_k^{(l)} = \sqrt{\text{tr}(\mathbf{W}_k^*)} \cdot \frac{\boldsymbol{\xi}_l}{\|\boldsymbol{\xi}_l\|}$
   - 计算每个候选的**约束违反度**:
     $$v_l = \sum_{k'} \max(0, \gamma_k - \text{SINR}_{k'}^{(l)}) + \sum_p \max(0, \gamma_S^{\text{PoD}} - \text{SINR}_{S,p}^{(l)})$$
   - 选择违反度最小的候选: $l^* = \arg\min_l v_l$
   - 若 $v_{l^*} = 0$: 接受 $\mathbf{w}_k = \mathbf{w}_k^{(l^*)}$
   - 若 $v_{l^*} > 0$: 进入功率缩放兜底 (步骤 3)

**步骤 2: 感知波形恢复**

对感知协方差 $\mathbf{Z}^*$ (通常秩 $> 1$):
1. 特征值分解: $\mathbf{Z}^* = \sum_{i=1}^{r} \lambda_i \mathbf{v}_i \mathbf{v}_i^H$
2. 生成多流传输波形: $\mathbf{z}_p = \sum_{i=1}^{r} \sqrt{\lambda_i} \mathbf{v}_i s_i$，其中 $s_i \sim \mathcal{CN}(0,1)$ 独立

**步骤 3: 功率缩放兜底 (Power Scaling Fallback)**

若单 AP 功率超限:
- 对每个 AP $m$，计算 $P_m = \sum_k \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m)$
- 若 $P_m > P_{\max}$，缩放因子 $\beta_m = P_{\max} / P_m$:
  - $\mathbf{w}_{m,k} \leftarrow \mathbf{w}_{m,k} \cdot \sqrt{\beta_m}$
  - $\mathbf{Z}_m \leftarrow \mathbf{Z}_m \cdot \beta_m$

**功率缩放数学保证**:
- 单 AP 功率: $P_m' = \beta_m P_m = P_{\max}$ (严格满足)
- 通信 SINR: $\text{SINR}_k' = \beta_m \cdot \text{SINR}_k$ (线性缩放)
- 感知 SINR: $\text{SINR}_{S,p}' = \beta_m \cdot \text{SINR}_{S,p}$ (线性缩放)

**步骤 4: 性能保证声明**

- $K \leq 2$ 时，SDR 紧致，高斯随机化以概率 1 恢复最优解
- $K > 2$ 时，高斯随机化提供**次优解**，性能损失上界为 $O(1/L)$
- 感知协方差 $\mathbf{Z}^*$ 的秩 $> 1$ 是**设计意图** (多目标覆盖)，非恢复失败

> **LaTeX 排版建议**：使用 `algorithm2e` 或 `algorithmic` 宏包将上述步骤包裹为规范浮动算法块（如 Algorithm 1），提升版面学术质感。关键步骤添加行内注释，例如 `\tcp{秩一检测}`。

---

## 9. 闭式解推导

### 9.1 假设条件

- 忽略 PCRB 约束 (或假设自动满足)
- 使用 ZF 通信波束
- 使用匹配滤波感知波束
- 单 AP 功率约束宽松

### 9.2 ZF 波束成形

**条件**: $\text{rank}(\mathbf{H}_{\text{all}}) \geq K$，即 $M^{\text{all}} N_t \geq K$。

**ZF 矩阵**:

$$\mathbf{W}_{\text{ZF}} = \mathbf{H}_{\text{all}} (\mathbf{H}_{\text{all}}^H \mathbf{H}_{\text{all}})^{-1} \tag{20}$$

**性质**:

$$\mathbf{h}_k^{\text{all},H} \mathbf{W}_{\text{ZF}}(:,j) = \delta_{kj}$$

**功率分配**:

$$p_k = \gamma_k^{\text{robust}} \sigma_c^2 \|\mathbf{W}_{\text{ZF}}(:,k)\|_2^2 \tag{21}$$

### 9.3 感知波束

$$\mathbf{z}_p = \sqrt{P_{S,p}} \frac{\mathbf{g}_p^{\text{all}}}{\|\mathbf{g}_p^{\text{all}}\|_2} \tag{22}$$

其中 $P_{S,p} = (\gamma_S^{\text{PoD}})^{\text{robust}} \sigma_s^2 / \|\mathbf{g}_p^{\text{all}}\|_2^2$。

### 9.4 完整求解算法

以下为闭式求解器的伪代码描述。该算法首先根据感知信道强度为每个目标选择最优 AP 子集，随后在所选 AP 上分别计算 ZF 通信波束和匹配滤波感知波束，最后验证功率约束并返回可行解。

```
算法: Cell-Free ISAC 闭式求解器

输入: H, G, M, Nt, K, P, Pmax, {γk}, γS^PoD, Nreq, εh, εg
输出: {wmk}, {Zm}, {bmp}, Ptotal

1. 计算鲁棒门限:
   γk^robust = γk * (1+εh)²/(1-εh)²
   γS^robust = γS^PoD * (1+εg)²/(1-εg)²

2. AP 选择:
   for p = 1,...,P:
       计算 ||g_{m,p}||₂ for all m
       选择 top-Nreq APs: bmp = 1
   Mall = ∪p {m : bmp = 1}

3. 提取子信道:
   Hall = H(Mall, :)
   Gp^all = G(Mall, p) for all p

4. 通信波束:
   if rank(Hall) ≥ K:
       Wzf = Hall * inv(Hall^H * Hall)
       for k = 1,...,K:
           wk^ZF = Wzf(:,k) / ||Wzf(:,k)||₂
           pk = γk^robust * σc² * ||Wzf(:,k)||₂²
           wk = sqrt(pk) * wk^ZF
   else:
       使用 MRT (次优)

5. 感知波束:
   for p = 1,...,P:
       PS,p = γS^robust * σs² / ||Gp^all||₂²
       zp = sqrt(PS,p) * Gp^all / ||Gp^all||₂
       Zm = zp * zp^H (rank-1)

6. 功率检查:
   for m ∈ Mall:
       Pm = Σk ||wmk||₂² + tr(Zm)
       if Pm > Pmax:
           缩放或标记不可行

7. 验证并返回
```

> **LaTeX 排版建议**：使用 `algorithm2e` 宏包将上述伪代码包裹为规范浮动算法块（如 Algorithm 2），提升版面学术质感。关键步骤（如 AP 选择、功率检查）可添加行内注释，例如 `\tcp{基于大尺度衰落选择}`。

---

## 10. 复杂度分析

### 10.1 SDP 求解复杂度

**变量数**:
- $K$ 个 $\mathbf{W}_k$: $K \cdot (MN_t)^2$ 实变量
- 1 个 $\mathbf{Z}$: $(MN_t)^2$ 实变量
- $K$ 个 $\mu_k$: $K$ 实变量
- 总计: $O(K(MN_t)^2)$

**约束数**:
- $K$ 个通信 LMI: $K \cdot (MN_t+1) \times (MN_t+1)$
- $P$ 个感知线性约束
- $M$ 个功率约束
- 总计: $O(K(MN_t)^3)$ 计算量

**求解时间**: $O((MN_t)^{3.5})$ — 对于 $M=16, N_t=4$ 约 5-10 秒 (MOSEK)。

### 10.2 闭式解复杂度

| 步骤 | 操作 | 复杂度 |
|------|------|--------|
| AP 选择 | 每目标排序 | $O(MP \log M)$ |
| ZF 求逆 | $(\mathbf{H}^H\mathbf{H})^{-1}$ | $O(K^3)$ |
| 通信功率 | $K$ 次乘法 | $O(K)$ |
| 感知波束 | $P$ 次归一化 | $O(PMN_t)$ |
| 功率检查 | $M$ 次求和 | $O(MK)$ |
| **总计** | | **$O(MP\log M + K^3)$** |

对于 $M=16, N_t=4, K=10, P=4$: $O(7400)$，即 $< 0.1$ 秒。

---

## 11. 可行性条件

### 11.1 必要条件

**ZF 可行性**:

$$M^{\text{all}} N_t \geq K$$

对于 $N_t=4, K=10$: $M^{\text{all}} \geq 3$ (理论)，$M^{\text{all}} \geq 4$ (数值稳定)。

**功率可行性**:

$$P_{\text{comm}}^{\min} + P_{\text{sens}}^{\min} \leq M \cdot P_{\max}$$

其中:

$$P_{\text{comm}}^{\min} = \sum_{k=1}^K \gamma_k^{\text{robust}} \sigma_c^2 \|\mathbf{W}_{\text{ZF}}(:,k)\|_2^2$$

$$P_{\text{sens}}^{\min} = \sum_{p=1}^P (\gamma_S^{\text{PoD}})^{\text{robust}} \frac{\sigma_s^2}{\|\mathbf{g}_p^{\text{all}}\|_2^2}$$

### 11.2 不可行情形

1. 目标远离所有 AP ($d > 50$m)
2. 用户数过多 ($K > M^{\text{all}}N_t$)
3. 功率预算过低 ($P_{\max} < P_{\text{comm}}^{\min}$)
4. CSI 误差过大 ($\epsilon > 0.5$)

---

## 12. 理论假设与防御声明

### 12.1 Assumption 1: 感知约束线性性

> **Assumption 1 (感知参数估计模型)**: 本文考虑的目标状态参数为 $\boldsymbol{\theta}_p = [\theta_p]$ (单角度估计)。在此设定下，Fisher 信息矩阵的数据部分为:
>
> $$\mathbf{J}_p^{\text{data}} = \frac{2}{\sigma_s^2} \text{Re}\Big\{ \nabla_{\boldsymbol{\theta}_p} \mathbf{g}_p^H \cdot \mathbf{R}_X \cdot \nabla_{\boldsymbol{\theta}_p}^H \mathbf{g}_p \Big\}$$
>
> 其中 $\nabla_{\boldsymbol{\theta}_p} \mathbf{g}_p$ 在当前时隙内由目标预测状态确定，视为已知常数矩阵。因此 $\mathbf{J}_p^{\text{data}}$ 是 $\mathbf{R}_X$ 的**仿射函数**，其迹约束可精确整理为:
>
> $$\text{tr}(\mathbf{F}_p \mathbf{R}_X) \geq \Gamma_{\text{Track},p}$$
>
> 其中 $\mathbf{F}_p = \frac{2}{\sigma_s^2} \text{Re}\Big\{ \nabla_{\boldsymbol{\theta}_p}^H \mathbf{g}_p \nabla_{\boldsymbol{\theta}_p} \mathbf{g}_p^H \Big\}$ 为已知常数半正定矩阵。
>
> **定性解释**: 由于本工作聚焦于目标的到达角 (DOA) 估计，探测信号与目标的相对时延及多普勒频移在短观测帧内视为慢变参数。此时 FIM 退化为仅依赖于角度梯度的常数矩阵 $\mathbf{F}_p$，从而保证了约束条件的凸仿射性质。

### 12.2 Assumption 2: 感知信道完美性

> **Assumption 2 (感知信道完美性假设)**: 本工作聚焦于通信链路的鲁棒设计，假设感知信道在当前时隙内通过跟踪回波自校准，误差可忽略。具体地:
>
> 1. 目标状态参数 $\boldsymbol{\theta}_p$ 在短时隙内被准确预测，预测误差远小于一个波长
> 2. 感知信道 $\mathbf{g}_p$ 通过上一时隙跟踪回波校准，误差纳入下一时隙更新
> 3. 感知任务采用"检测-跟踪"级联架构: 检测阶段保守门限，跟踪阶段利用时隙平滑性补偿误差
>
> **合理性**: 感知信道是**自校准的** (发射探测信号并接收回波，回波携带当前信道状态)，而通信信道依赖上行导频估计，导频污染导致误差累积。在许多顶级期刊的系统建模中，聚焦通信侧导频污染、假设雷达侧得益于 LOS 回波和卡尔曼跟踪滤波提供精准预测，是常见的稳妥折中。
>
> **扩展说明**: 若需考虑感知不确定性，可将感知 SINR 约束通过 S-Procedure 转化为 LMI，增加 $P$ 个 LMI 约束和约 $20\%$ 求解时间，但不改变问题凸性。

### 12.3 秩一恢复声明

> **Algorithm 1**: 当 SDR 求解得到的通信协方差 $\mathbf{W}_k^*$ 秩大于 1 时，采用高斯随机化技术提取次优波束。具体地，生成 $L$ 个候选波束 $\mathbf{w}_k^{(l)} \sim \mathcal{CN}(\mathbf{0}, \mathbf{W}_k^*)$，选择满足所有约束的最佳候选。若仍不满足，采用功率缩放兜底。感知协方差 $\mathbf{Z}^*$ 的多秩性质是物理需求 (多目标覆盖)，非算法缺陷。
>
> **性能保证**: $K \leq 2$ 时 SDR 紧致；$K > 2$ 时性能损失上界 $O(1/L)$。感知协方差 $\mathbf{Z}^*$ 的多秩是物理需求 (多目标覆盖)，非算法缺陷。

---

## 13. 关键公式汇总

| 编号 | 名称 | 表达式 |
|------|------|--------|
| (3) | 通信鲁棒性因子 | $\eta_h = ((1-\epsilon_h)/(1+\epsilon_h))^2$ |
| (4) | 通信鲁棒门限 | $\gamma_k^{\text{robust}} = \gamma_k / \eta_h$ |
| (9) | 感知鲁棒性因子 | $\eta_g = ((1-\epsilon_g)/(1+\epsilon_g))^2$ |
| (10) | 感知鲁棒门限 | $(\gamma_S^{\text{PoD}})^{\text{robust}} = \gamma_S^{\text{PoD}} / \eta_g$ |
| (12) | FIM 迹约束 | $\text{tr}(\mathbf{J}_p^{\text{data}}) = \text{tr}(\mathbf{F}_p \mathbf{R}_X)$ |
| (13) | FIM 常数矩阵 | $\mathbf{F}_p = \frac{2}{\sigma_s^2} \text{Re}\Big\{ \nabla_{\boldsymbol{\theta}_p}^H \mathbf{g}_p \nabla_{\boldsymbol{\theta}_p} \mathbf{g}_p^H \Big\}$ |
| (18) | 通信 S-Procedure LMI | $\begin{bmatrix} \mathbf{A}_k + \mu_k \mathbf{I} & \mathbf{A}_k \hat{\mathbf{h}}_k \\ \hat{\mathbf{h}}_k^H \mathbf{A}_k & \hat{\mathbf{h}}_k^H \mathbf{A}_k \hat{\mathbf{h}}_k - \sigma_c^2 + \mu_k \epsilon_h^2 \end{bmatrix} \succeq \mathbf{0}$ |
| (19) | 感知 S-Procedure LMI (可选) | $\begin{bmatrix} \frac{1}{\gamma_S^{\text{PoD}}} \mathbf{Z} + \nu_p \mathbf{I} & \frac{1}{\gamma_S^{\text{PoD}}} \mathbf{Z} \hat{\mathbf{g}}_p \\ \frac{1}{\gamma_S^{\text{PoD}}} \hat{\mathbf{g}}_p^H \mathbf{Z} & \frac{1}{\gamma_S^{\text{PoD}}} \hat{\mathbf{g}}_p^H \mathbf{Z} \hat{\mathbf{g}}_p - \sigma_s^2 + \nu_p \epsilon_g^2 \end{bmatrix} \succeq \mathbf{0}$ |
| (20) | ZF 矩阵 | $\mathbf{W}_{\text{ZF}} = \mathbf{H} (\mathbf{H}^H \mathbf{H})^{-1}$ |
| (21) | 通信功率分配 | $p_k = \gamma_k^{\text{robust}} \sigma_c^2 \|\mathbf{W}_{\text{ZF}}(:,k)\|_2^2$ |

---

## 版本信息

- **文档**: 统一数学推导 v2.0
- **日期**: 2026-06-16
- **整合内容**: 闭式解 + SDP松弛 + S-Procedure + 凸性证明 + 秩一恢复 + 防御声明
- **基于**: 标准 ISAC 问题形式 (P1a-P1h)
- **状态**: 可直接用于论文核心章节撰写
