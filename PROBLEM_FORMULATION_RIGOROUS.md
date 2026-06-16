# Cell-Free ISAC Complete Problem Formulation (Rigorous)

本文档是本项目唯一保留的完整问题定义，统一描述 Cell-Free Integrated Sensing and Communication (ISAC) 系统的问题描述、系统模型、优化变量、目标函数和约束条件。

**版本**: v2.0 (Rigorous)  
**日期**: 2026-06-16  
**状态**: 补全版 — 包含完整数学推导、等价形式、复杂度分析

---

## 目录

1. [Notation & Symbol Table](#1-notation--symbol-table)
2. [Problem Description](#2-problem-description)
3. [System Model](#3-system-model)
4. [Signal Model](#4-signal-model)
5. [Optimization Problem](#5-optimization-problem)
6. [Constraint Analysis](#6-constraint-analysis)
7. [Equivalent Forms](#7-equivalent-forms)
8. [Feasibility Analysis](#8-feasibility-analysis)
9. [Algorithmic Decomposition](#9-algorithmic-decomposition)
10. [Complexity Analysis](#10-complexity-analysis)
11. [Current Implementation](#11-current-implementation)

---

## 1. Notation & Symbol Table

### 1.1 Sets and Indices

| Symbol | Description | Range |
|--------|-------------|-------|
| $\mathcal{M}$ | AP 集合 | $\{1, 2, \ldots, M\}$ |
| $\mathcal{K}$ | 用户集合 | $\{1, 2, \ldots, K\}$ |
| $\mathcal{P}$ | 目标集合 | $\{1, 2, \ldots, P\}$ |
| $\mathcal{A}$ | 激活 AP 子集 | $\mathcal{A} \subseteq \mathcal{M}, |\mathcal{A}| = N_{\text{active}}$ |
| $m$ | AP 索引 | $m \in \mathcal{M}$ |
| $k$ | 用户索引 | $k \in \mathcal{K}$ |
| $p$ | 目标索引 | $p \in \mathcal{P}$ |
| $n$ | 天线索引 | $n \in \{1, 2, \ldots, N_t\}$ |

### 1.2 Scalars

| Symbol | Description | Typical Value | Unit |
|--------|-------------|---------------|------|
| $M$ | AP 数量 | 16 | — |
| $N_t$ | 每 AP 天线数 | 4 | — |
| $K$ | 通信用户数 | 10 | — |
| $P$ | 感知目标数 | 4 | — |
| $N_{\text{active}}$ | 激活 AP 数 | 7–16 | — |
| $P_{\max}$ | 系统总功率预算 | 30 | W |
| $P_{m,\max}$ | 单 AP 功率上限 | 30 | W |
| $\gamma_{\text{comm}}$ | 通信 SINR 门限 | 0 (linear: 1) | dB |
| $\gamma_{\text{sens}}$ | 感知 SNR 门限 | 0 (linear: 1) | dB |
| $\Gamma$ | CRB 门限 | 1 | m² 或无量纲 |
| $\sigma_c^2$ | 通信噪声功率 | $10^{-9}$ | W (-60 dBm) |
| $\sigma_s^2$ | 感知噪声功率 | $10^{-9}$ | W (-60 dBm) |
| $\epsilon_h$ | 通信 CSI 误差界 | 0.10 | — |
| $\epsilon_g$ | 感知 CSI 误差界 | 0.15 | — |
| $\eta$ | 路径损耗指数 | 2.5 | — |
| $d_0$ | 参考距离 | 10 | m |
| $\rho$ | 通信功率占比 | $[0, 1]$ | — |
| $\alpha$ | AP 选择权重 | 0.5 | — |

### 1.3 Vectors and Matrices

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| $\mathbf{h}_{m,k}$ | $\mathbb{C}^{N_t \times 1}$ | AP $m$ 到用户 $k$ 的信道 |
| $\mathbf{g}_{m,p}$ | $\mathbb{C}^{N_t \times 1}$ | AP $m$ 到目标 $p$ 的信道 |
| $\mathbf{h}_k$ | $\mathbb{C}^{MN_t \times 1}$ | 所有 AP 到用户 $k$ 的堆叠信道 |
| $\mathbf{g}_p$ | $\mathbb{C}^{MN_t \times 1}$ | 所有 AP 到目标 $p$ 的堆叠信道 |
| $\mathbf{w}_{m,k}$ | $\mathbb{C}^{N_t \times 1}$ | AP $m$ 到用户 $k$ 的通信波束 |
| $\mathbf{w}_k$ | $\mathbb{C}^{MN_t \times 1}$ | 面向用户 $k$ 的堆叠通信波束 |
| $\mathbf{z}_p$ | $\mathbb{C}^{MN_t \times 1}$ | 面向目标 $p$ 的感知波束 |
| $\mathbf{Z}_m$ | $\mathbb{C}^{N_t \times N_t}$ | AP $m$ 的感知协方差矩阵 |
| $\mathbf{W}$ | $\mathbb{C}^{M \times N_t \times K}$ | 通信波束张量 |
| $\mathbf{H}$ | $\mathbb{C}^{MN_t \times K}$ | 通信信道矩阵 $[\mathbf{h}_1, \ldots, \mathbf{h}_K]$ |
| $\mathbf{G}$ | $\mathbb{C}^{MN_t \times P}$ | 感知信道矩阵 $[\mathbf{g}_1, \ldots, \mathbf{g}_P]$ |
| $\mathbf{a}$ | $\{0,1\}^M$ | AP 选择向量 |
| $\mathbf{q}_m$ | $\mathbb{R}^{2 \times 1}$ | AP $m$ 的二维位置 |
| $\mathbf{u}_k$ | $\mathbb{R}^{2 \times 1}$ | 用户 $k$ 的二维位置 |
| $\mathbf{r}_p$ | $\mathbb{R}^{2 \times 1}$ | 目标 $p$ 的二维位置 |

### 1.4 Operators

| Symbol | Description |
|--------|-------------|
| $(\cdot)^H$ | Hermitian (共轭转置) |
| $(\cdot)^T$ | 转置 |
| $(\cdot)^*$ | 复共轭 |
| $\|\cdot\|_2$ | 欧氏范数 (向量) / 谱范数 (矩阵) |
| $\|\cdot\|_F$ | Frobenius 范数 |
| $\text{tr}(\cdot)$ | 矩阵迹 |
| $\text{rank}(\cdot)$ | 矩阵秩 |
| $\succeq$ | 半正定 (Löwner 序) |
| $\mathbb{E}[\cdot]$ | 期望 |
| $\text{Re}(\cdot)$ | 实部 |
| $\text{diag}(\cdot)$ | 对角矩阵 |
| $\text{blkdiag}(\cdot)$ | 块对角矩阵 |
| $\mathcal{CN}(\boldsymbol{\mu}, \mathbf{\Sigma})$ | 复高斯分布 |
| $\mathbf{I}_n$ | $n \times n$ 单位矩阵 |
| $\mathbf{0}$ | 零矩阵/向量 |

---

## 2. Problem Description

### 2.1 系统概述

考虑一个 Cell-Free ISAC 系统，其中 $M$ 个分布式接入点 (AP) 通过前传链路连接到中央处理单元 (CPU)。每个 AP 配备 $N_t$ 根发射天线。系统同时服务 $K$ 个单天线通信用户，并对 $P$ 个感知目标进行探测/定位。

**关键特征**:
- **Cell-Free 架构**: AP 分布式部署，无蜂窝边界
- **ISAC 双功能**: 同一波形同时用于通信和感知
- **集中式处理**: CPU 联合设计所有 AP 的波束成形
- **不完美 CSI**: 信道估计存在有界误差

### 2.2 优化目标

**主目标**: 在满足所有服务质量约束的前提下，最小化系统总发射功率。

**等价目标** (固定功率时): 最小化约束违反度，最大化可行性概率。

### 2.3 非凸性来源

该问题本质上是非凸的，主要来源:

1. **SINR 约束**: 分式结构 + 多用户干扰耦合
2. **SNR 约束**: 二次型 + 波束-信道耦合
3. **二进制约束**: $a_m \in \{0,1\}$ 构成组合优化
4. **变量耦合**: $W, Z, \mathbf{a}, \rho$ 相互耦合
5. **鲁棒约束**: 无穷范数约束 (worst-case) 难以直接处理
6. **CRB 约束**: 涉及矩阵逆，非凸

---

## 3. System Model

### 3.1 Network Topology

**AP 部署**:
```
4 × 4 均匀网格
覆盖区域: [-60, 60] × [-60, 60] m²
AP 间距: 40 m
```

**用户分布**:
```
均匀随机分布在 [-50, 50] × [-50, 50] m²
```

**目标分布**:
```
均匀随机分布在 [-30, 30] × [-30, 30] m² (默认)
或 [-R, R] × [-R, R] m² (参数化)
```

**最小距离约束**: $d \geq 5$ m (避免路径损耗奇点)

### 3.2 Channel Model

#### 3.2.1 通信信道

AP $m$ 到用户 $k$ 的信道:

$$\mathbf{h}_{m,k} = \sqrt{\text{PL}(d_{m,k})} \cdot \boldsymbol{\alpha}_{m,k} \in \mathbb{C}^{N_t \times 1}$$

其中:
- 路径损耗: $\text{PL}(d) = \left(\frac{d}{d_0}\right)^{-\eta}$
- 小尺度衰落: $\boldsymbol{\alpha}_{m,k} \sim \mathcal{CN}(\mathbf{0}, \mathbf{I}_{N_t})$
- 距离: $d_{m,k} = \|\mathbf{q}_m - \mathbf{u}_k\|_2$

**堆叠信道** (所有 AP):

$$\mathbf{h}_k = [\mathbf{h}_{1,k}^T, \mathbf{h}_{2,k}^T, \ldots, \mathbf{h}_{M,k}^T]^T \in \mathbb{C}^{MN_t \times 1}$$

#### 3.2.2 感知信道

AP $m$ 到目标 $p$ 的信道:

$$\mathbf{g}_{m,p} = \sqrt{\text{PL}(d_{m,p})} \cdot \boldsymbol{\beta}_{m,p} \in \mathbb{C}^{N_t \times 1}$$

其中:
- $\boldsymbol{\beta}_{m,p} \sim \mathcal{CN}(\mathbf{0}, \mathbf{I}_{N_t})$
- $d_{m,p} = \|\mathbf{q}_m - \mathbf{r}_p\|_2$

**堆叠信道**:

$$\mathbf{g}_p = [\mathbf{g}_{1,p}^T, \mathbf{g}_{2,p}^T, \ldots, \mathbf{g}_{M,p}^T]^T \in \mathbb{C}^{MN_t \times 1}$$

#### 3.2.3 信道矩阵

$$\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_K] \in \mathbb{C}^{MN_t \times K}$$

$$\mathbf{G} = [\mathbf{g}_1, \mathbf{g}_2, \ldots, \mathbf{g}_P] \in \mathbb{C}^{MN_t \times P}$$

### 3.3 Imperfect CSI Model

#### 3.3.1 误差模型

真实信道由估计值和误差组成:

$$\mathbf{h}_{m,k} = \hat{\mathbf{h}}_{m,k} + \Delta\mathbf{h}_{m,k}$$

$$\mathbf{g}_{m,p} = \hat{\mathbf{g}}_{m,p} + \Delta\mathbf{g}_{m,p}$$

#### 3.3.2 有界不确定性集合

**通信 CSI 误差**:

$$\mathcal{H}_k = \left\{\Delta\mathbf{h}_k : \|\Delta\mathbf{h}_k\|_2 \leq \epsilon_h \|\hat{\mathbf{h}}_k\|_2 \right\}$$

**感知 CSI 误差**:

$$\mathcal{G}_p = \left\{\Delta\mathbf{g}_p : \|\Delta\mathbf{g}_p\|_2 \leq \epsilon_g \|\hat{\mathbf{g}}_p\|_2 \right\}$$

#### 3.3.3 Worst-Case 等价形式

对于任意向量 $\mathbf{x}, \mathbf{y}$ 和误差 $\Delta\mathbf{y}$ 满足 $\|\Delta\mathbf{y}\| \leq \epsilon\|\mathbf{y}\|$:

$$\min_{\|\Delta\mathbf{y}\| \leq \epsilon\|\mathbf{y}\|} |\mathbf{x}^H(\mathbf{y} + \Delta\mathbf{y})|^2 = |\mathbf{x}^H\mathbf{y}|^2 \cdot (1 - \epsilon)^2$$

$$\max_{\|\Delta\mathbf{y}\| \leq \epsilon\|\mathbf{y}\|} |\mathbf{x}^H(\mathbf{y} + \Delta\mathbf{y})|^2 = |\mathbf{x}^H\mathbf{y}|^2 \cdot (1 + \epsilon)^2$$

**证明**: 由 Cauchy-Schwarz 不等式,
$|\mathbf{x}^H\Delta\mathbf{y}| \leq \|\mathbf{x}\| \cdot \|\Delta\mathbf{y}\| \leq \|\mathbf{x}\| \cdot \epsilon\|\mathbf{y}\|$

当 $\Delta\mathbf{y} = -\epsilon\frac{\|\mathbf{y}\|}{\|\mathbf{x}\|}\mathbf{x}$ 时取等 (最小化情形)。

---

## 4. Signal Model

### 4.1 发射信号

AP $m$ 的发射信号:

$$\mathbf{x}_m = \sum_{k=1}^K \mathbf{w}_{m,k} s_k + \sum_{p=1}^P \mathbf{z}_{m,p} c_p$$

其中:
- $s_k$: 用户 $k$ 的数据符号, $\mathbb{E}[|s_k|^2] = 1$
- $c_p$: 感知信号 (雷达波形), $\mathbb{E}[|c_p|^2] = 1$
- $\mathbf{w}_{m,k}$: 通信波束
- $\mathbf{z}_{m,p}$: 感知波束

**总发射功率**:

$$P_m = \sum_{k=1}^K \|\mathbf{w}_{m,k}\|_2^2 + \sum_{p=1}^P \|\mathbf{z}_{m,p}\|_2^2$$

### 4.2 通信信号

用户 $k$ 的接收信号:

$$y_k = \underbrace{\mathbf{h}_k^H \mathbf{w}_k s_k}_{\text{期望信号}} + \underbrace{\sum_{j \neq k} \mathbf{h}_k^H \mathbf{w}_j s_j}_{\text{多用户干扰}} + \underbrace{\mathbf{h}_k^H \mathbf{z}_{\text{sens}}}_{\text{感知干扰}} + \underbrace{n_k}_{\text{噪声}}$$

其中 $\mathbf{z}_{\text{sens}} = \sum_p \mathbf{z}_p c_p$ 是感知信号分量。

**简化模型** (感知作为噪声处理):

$$y_k \approx \mathbf{h}_k^H \mathbf{w}_k s_k + \sum_{j \neq k} \mathbf{h}_k^H \mathbf{w}_j s_j + n_k$$

**通信 SINR**:

$$\text{SINR}_k = \frac{|\mathbf{h}_k^H \mathbf{w}_k|^2}{\sum_{j \neq k} |\mathbf{h}_k^H \mathbf{w}_j|^2 + \sigma_c^2}$$

### 4.3 感知信号

#### 4.3.1 波束形式

目标 $p$ 的接收信号 (单基地/自发自收):

$$y_p = \mathbf{g}_p^H \mathbf{z}_p + n_p$$

**感知 SNR**:

$$\text{SNR}_p = \frac{|\mathbf{g}_p^H \mathbf{z}_p|^2}{\sigma_s^2 \|\mathbf{z}_p\|_2^2}$$

#### 4.3.2 协方差形式

若使用协方差矩阵 $\mathbf{Z}_m = \mathbb{E}[\mathbf{x}_m^{\text{sens}} (\mathbf{x}_m^{\text{sens}})^H]$:

$$\text{SNR}_p = \frac{\sum_{m=1}^M |\mathbf{g}_{m,p}^H \mathbf{Z}_m \mathbf{g}_{m,p}|^2}{\sigma_s^2 \sum_{m=1}^M \text{tr}(\mathbf{Z}_m)}$$

**注**: 当 $\mathbf{Z}_m = \mathbf{z}_{m,p}\mathbf{z}_{m,p}^H$ (rank-1) 时，两种形式等价。

### 4.4 信号与干扰的关系

**关键假设**: 通信和感知使用同一波形 (ISAC 共享波形)，因此:

- 感知信号对通信用户构成干扰
- 通信信号对感知目标构成干扰 (自干扰)

**自干扰消除**: 假设通过全双工技术或时间/频率分离消除，因此感知 SNR 公式中不包含通信信号干扰。

---

## 5. Optimization Problem

### 5.1 完整问题 (P1)

**目标函数**:

$$\min_{\{\mathbf{w}_{m,k}\}, \{\mathbf{Z}_m\}, \{b_{mp}\}} \quad \sum_{m=1}^{M} \left( \sum_{k=1}^{K} \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m) \right) \tag{5a}$$

**约束条件**:

**(C1) 通信 QoS 约束 (最坏情况 SINR)**:

$$\text{SINR}_k^{\text{wc}} \geq \gamma_k, \quad \forall k \in \mathcal{K} \tag{5b}$$

其中最坏情况 SINR 定义为:

$$\text{SINR}_k^{\text{wc}} = \min_{\Delta\mathbf{h}_k \in \mathcal{H}_k} \frac{|(\hat{\mathbf{h}}_k + \Delta\mathbf{h}_k)^H \mathbf{w}_k|^2}{\sum_{j \neq k} |(\hat{\mathbf{h}}_k + \Delta\mathbf{h}_k)^H \mathbf{w}_j|^2 + \sigma_c^2}$$

**(C2) 感知 SINR 约束 (检测概率)**:

$$\text{SINR}_{S,p} \geq \gamma_S^{\text{PoD}}, \quad \forall p \in \mathcal{P} \tag{5c}$$

其中感知 SINR:

$$\text{SINR}_{S,p} = \frac{\left|\sum_{m=1}^M \mathbf{g}_{m,p}^H \mathbf{Z}_m \mathbf{g}_{m,p}\right|^2}{\sigma_s^2 \sum_{m=1}^M \text{tr}(\mathbf{Z}_m)}$$

或波束形式:

$$\text{SINR}_{S,p} = \frac{|\mathbf{g}_p^H \mathbf{z}_p|^2}{\sigma_s^2 \|\mathbf{z}_p\|_2^2}$$

**(C3) 感知 PCRB 约束 (跟踪精度)**:

$$\text{tr}\left(\mathbf{J}_p^{\text{data}}\right) \geq \Gamma_{\text{Track}, p}, \quad \forall p \in \mathcal{P} \tag{5d}$$

其中 $\mathbf{J}_p^{\text{data}}$ 是感知数据 Fisher 信息矩阵:

$$\mathbf{J}_p^{\text{data}} = \frac{2}{\sigma_s^2} \sum_{m=1}^M \sum_{k=1}^K \text{Re}\left\{ \nabla_{\boldsymbol{\theta}_p} (\mathbf{g}_{m,p}^H \mathbf{w}_{m,k}) \cdot \nabla_{\boldsymbol{\theta}_p}^H (\mathbf{g}_{m,p}^H \mathbf{w}_{m,k}) \right\}$$

**(C4) AP 选择约束**:

$$\sum_{m=1}^M b_{mp} = N_{\text{req}}, \quad \forall p \in \mathcal{P}^{\text{active}} \tag{5e}$$

其中 $b_{mp} \in \{0,1\}$ 表示 AP $m$ 是否参与目标 $p$ 的感知服务。

**(C5) 单 AP 功率约束**:

$$\sum_{k=1}^K \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m) \leq P_{\max}, \quad \forall m \in \mathcal{M} \tag{5f}$$

**(C6) 半正定约束**:

$$\mathbf{Z}_m \succeq \mathbf{0}, \quad \forall m \in \mathcal{M} \tag{5g}$$

**(C7) 二进制约束**:

$$b_{mp} \in \{0, 1\}, \quad \forall m \in \mathcal{M}, p \in \mathcal{P} \tag{5h}$$

### 5.2 与标准形式的对应关系

| 本文档 | 标准形式 | 说明 |
|--------|----------|------|
| $\gamma_k$ | $\gamma_k$ | 通信用户 $k$ 的 SINR 门限 |
| $\gamma_S^{\text{PoD}}$ | $\gamma_S^{\text{PoD}}$ | 感知检测概率门限 |
| $\Gamma_{\text{Track}, p}$ | $\Gamma_{\text{Track}, p}$ | 感知跟踪精度门限 |
| $N_{\text{req}}$ | $N_{\text{req}}$ | 每目标所需协作 AP 数 |
| $P_{\max}$ | $P_{\max}$ | 单 AP 功率上限 |
| $\mathcal{P}^{\text{active}}$ | $\mathcal{P}_n^{\text{active}}$ | 活跃目标集合 (时隙 $n$) |
| $b_{mp}$ | $b_{mp}[n]$ | AP-目标关联变量 |

### 5.3 变量定义域

| 变量 | 定义域 | 维度 | 说明 |
|------|--------|------|------|
| $\mathbf{w}_{m,k}$ | $\mathbb{C}^{N_t \times 1}$ | $N_t$ 复变量 | AP $m$ 到用户 $k$ 的波束 |
| $\mathbf{Z}_m$ | $\mathbb{C}^{N_t \times N_t}, \succeq \mathbf{0}$ | $N_t^2$ 复变量 | AP $m$ 的感知协方差 |
| $b_{mp}$ | $\{0, 1\}$ | 二进制 | AP $m$ 服务目标 $p$ |

**总变量数**:
- 连续变量: $MKN_t + MN_t^2$ 个复变量
- 二进制变量: $MP$ 个
- 对于 $M=16, N_t=4, K=10, P=4$: 
  - 连续: $640 + 256 = 896$
  - 二进制: $64$

---

## 6. Constraint Analysis

### 6.1 通信 SINR 约束 (5b) 的等价形式

**分式约束**:

$$\frac{|\mathbf{h}_k^H \mathbf{w}_k|^2}{\sum_{j \neq k} |\mathbf{h}_k^H \mathbf{w}_j|^2 + \sigma_c^2} \geq \gamma_{\text{comm}}$$

**等价于** (假设分母 > 0):

$$|\mathbf{h}_k^H \mathbf{w}_k|^2 \geq \gamma_{\text{comm}} \left(\sum_{j \neq k} |\mathbf{h}_k^H \mathbf{w}_j|^2 + \sigma_c^2\right)$$

**展开**:

$$\mathbf{w}_k^H \mathbf{h}_k \mathbf{h}_k^H \mathbf{w}_k - \gamma_{\text{comm}} \sum_{j \neq k} \mathbf{w}_j^H \mathbf{h}_k \mathbf{h}_k^H \mathbf{w}_j \geq \gamma_{\text{comm}} \sigma_c^2$$

**矩阵形式**:

定义 $\mathbf{H}_k = \mathbf{h}_k \mathbf{h}_k^H$ (rank-1 PSD 矩阵):

$$\text{tr}(\mathbf{H}_k \mathbf{w}_k \mathbf{w}_k^H) - \gamma_{\text{comm}} \sum_{j \neq k} \text{tr}(\mathbf{H}_k \mathbf{w}_j \mathbf{w}_j^H) \geq \gamma_{\text{comm}} \sigma_c^2$$

**注**: 这是关于 $\mathbf{W}_k = \mathbf{w}_k \mathbf{w}_k^H$ 的线性约束，但 $\mathbf{W}_k$ 需满足 $\text{rank}(\mathbf{W}_k) = 1$。

### 6.2 鲁棒通信约束 (5b) 的精确形式

**Worst-case SINR**:

$$\text{SINR}_k^{\text{wc}} = \min_{\|\Delta\mathbf{h}_k\| \leq \epsilon_h \|\hat{\mathbf{h}}_k\|} \frac{|(\hat{\mathbf{h}}_k + \Delta\mathbf{h}_k)^H \mathbf{w}_k|^2}{\sum_{j \neq k} |(\hat{\mathbf{h}}_k + \Delta\mathbf{h}_k)^H \mathbf{w}_j|^2 + \sigma_c^2}$$

**保守近似** (分子取下界，分母取上界):

$$\text{SINR}_k^{\text{wc}} \approx \frac{|\hat{\mathbf{h}}_k^H \mathbf{w}_k|^2 (1-\epsilon_h)^2}{\sum_{j \neq k} |\hat{\mathbf{h}}_k^H \mathbf{w}_j|^2 (1+\epsilon_h)^2 + \sigma_c^2}$$

**简化近似** (仅分子缩放):

$$\text{SINR}_k^{\text{wc}} \approx \text{SINR}_k^{\text{nom}} \cdot \frac{(1-\epsilon_h)^2}{(1+\epsilon_h)^2}$$

**鲁棒性因子**:

$$\eta_h = \frac{(1-\epsilon_h)^2}{(1+\epsilon_h)^2} = \frac{1-\epsilon_h}{1+\epsilon_h} \cdot \frac{1-\epsilon_h}{1+\epsilon_h}$$

对于 $\epsilon_h = 0.10$:

$$\eta_h = \frac{0.9}{1.1} \cdot \frac{0.9}{1.1} = 0.818^2 \approx 0.669$$

**等价约束** (使用近似):

$$\text{SINR}_k^{\text{nom}} \geq \frac{\gamma_{\text{comm}}}{\eta_h} = \frac{\gamma_{\text{comm}}(1+\epsilon_h)^2}{(1-\epsilon_h)^2}$$

对于 $\gamma_{\text{comm}} = 1$ (0 dB):

$$\text{SINR}_k^{\text{nom}} \geq \frac{1}{0.669} \approx 1.496 \text{ (1.75 dB)}$$

即需要额外 **1.75 dB** 的功率余量。

### 6.3 感知 SINR 约束 (5c) — 检测概率

**协方差形式**:

$$\text{SINR}_{S,p} = \frac{\left|\sum_{m=1}^M \mathbf{g}_{m,p}^H \mathbf{Z}_m \mathbf{g}_{m,p}\right|^2}{\sigma_s^2 \sum_{m=1}^M \text{tr}(\mathbf{Z}_m)} \geq \gamma_S^{\text{PoD}}$$

**波束形式** (rank-1 协方差 $\mathbf{Z}_m = \mathbf{z}_{m,p}\mathbf{z}_{m,p}^H$):

$$\text{SINR}_{S,p} = \frac{|\mathbf{g}_p^H \mathbf{z}_p|^2}{\sigma_s^2 \|\mathbf{z}_p\|_2^2} \geq \gamma_S^{\text{PoD}}$$

**等价于**:

$$|\mathbf{g}_p^H \mathbf{z}_p|^2 \geq \gamma_S^{\text{PoD}} \sigma_s^2 \|\mathbf{z}_p\|_2^2$$

**最优波束** (匹配滤波):

对于固定功率 $\|\mathbf{z}_p\|_2^2 = P_{\text{sens},p}$，最大化 SNR 的波束为:

$$\mathbf{z}_p^* = \sqrt{P_{\text{sens},p}} \frac{\mathbf{g}_p}{\|\mathbf{g}_p\|_2}$$

**最优 SNR**:

$$\text{SINR}_{S,p}^* = \frac{P_{\text{sens},p} \|\mathbf{g}_p\|_2^2}{\sigma_s^2}$$

**最小功率需求**:

$$P_{\text{sens},p} \geq \frac{\gamma_S^{\text{PoD}} \sigma_s^2}{\|\mathbf{g}_p\|_2^2}$$

### 6.4 鲁棒感知约束 (5c) 的精确形式

类似通信鲁棒约束:

$$\text{SINR}_{S,p}^{\text{wc}} \approx \text{SINR}_{S,p}^{\text{nom}} \cdot \frac{(1-\epsilon_g)^2}{(1+\epsilon_g)^2}$$

**鲁棒性因子**:

$$\eta_g = \frac{(1-\epsilon_g)^2}{(1+\epsilon_g)^2}$$

对于 $\epsilon_g = 0.15$:

$$\eta_g = \frac{0.85}{1.15} \cdot \frac{0.85}{1.15} = 0.739^2 \approx 0.546$$

**等价约束**:

$$\text{SINR}_{S,p}^{\text{nom}} \geq \frac{\gamma_S^{\text{PoD}}}{\eta_g} = \frac{\gamma_S^{\text{PoD}}(1+\epsilon_g)^2}{(1-\epsilon_g)^2}$$

对于 $\gamma_S^{\text{PoD}} = 1$ (0 dB):

$$\text{SINR}_{S,p}^{\text{nom}} \geq \frac{1}{0.546} \approx 1.832 \text{ (2.63 dB)}$$

即需要额外 **2.63 dB** 的功率余量。

### 6.5 PCRB 约束 (5d) — 跟踪精度

#### 6.5.1 感知数据 Fisher 信息矩阵

$$\mathbf{J}_p^{\text{data}} = \frac{2}{\sigma_s^2} \sum_{m=1}^M \sum_{k=1}^K \text{Re}\left\{\nabla_{\boldsymbol{\theta}_p} (\mathbf{g}_{m,p}^H \mathbf{w}_{m,k}) \cdot \nabla_{\boldsymbol{\theta}_p}^H (\mathbf{g}_{m,p}^H \mathbf{w}_{m,k})\right\}$$

其中 $\boldsymbol{\theta}_p = [x_p, y_p]^T$ 是目标位置。

#### 6.5.2 PCRB 约束形式

约束要求 Fisher 信息矩阵的迹足够大:

$$\text{tr}(\mathbf{J}_p^{\text{data}}) \geq \Gamma_{\text{Track}, p}$$

这等价于要求定位误差的下界 (PCRB) 足够小:

$$\text{PCRB}_p = \text{tr}\left((\mathbf{J}_p^{\text{data}})^{-1}\right) \leq \frac{2}{\Gamma_{\text{Track}, p}}$$

**简化形式** (忽略交叉项):

$$\text{tr}(\mathbf{J}_p^{\text{data}}) \approx \frac{2}{\sigma_s^2} \sum_{m=1}^M \sum_{k=1}^K |\mathbf{g}_{m,p}^H \mathbf{w}_{m,k}|^2 \geq \Gamma_{\text{Track}, p}$$

**等价于**:

$$\sum_{m=1}^M \sum_{k=1}^K |\mathbf{g}_{m,p}^H \mathbf{w}_{m,k}|^2 \geq \frac{\sigma_s^2 \Gamma_{\text{Track}, p}}{2}$$

### 6.6 功率约束 (5f)

| 约束 | 数学形式 | 说明 |
|------|----------|------|
| 单 AP 功率 | $\sum_k \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m) \leq P_{\max}$ | 每 AP 功率上限 |

**注**: 标准形式中使用 $P_{\max}$ 作为单 AP 功率上限 (而非系统总功率)。系统总功率为 $\sum_m P_m$。

### 6.7 AP 选择约束 (5e)

#### 6.7.1 按目标选择

与固定 $N_{\text{active}}$ 不同，标准形式要求每个目标 $p$ 由恰好 $N_{\text{req}}$ 个 AP 服务:

$$\sum_{m=1}^M b_{mp} = N_{\text{req}}, \quad \forall p \in \mathcal{P}^{\text{active}}$$

#### 6.7.2 选择策略

对于每个目标 $p$，选择信道最强的 $N_{\text{req}}$ 个 AP:

$$b_{mp} = \begin{cases} 1, & \text{if } m \in \text{top-}N_{\text{req}} \text{ by } \|\mathbf{g}_{m,p}\|_2 \\ 0, & \text{otherwise} \end{cases}$$

**总激活 AP 数**: 最多 $MP$ (若每个目标选择不同 AP)，最少 $N_{\text{req}}$ (若全部共享)。

#### 6.7.3 与固定 AP 选择的区别

| 方式 | 变量 | 约束 | 说明 |
|------|------|------|------|
| 固定 AP | $a_m$ | $\sum_m a_m = N_{\text{active}}$ | 所有目标共享同一 AP 集合 |
| 按目标选择 | $b_{mp}$ | $\sum_m b_{mp} = N_{\text{req}}$ | 每个目标可有自己的 AP 集合 |

---

## 7. Equivalent Forms

### 7.1 SDR 松弛形式 (P2)

将 rank-1 约束松弛，定义 $\mathbf{W}_{m,k} = \mathbf{w}_{m,k} \mathbf{w}_{m,k}^H$:

$$\min_{\{\mathbf{W}_{m,k}\}, \{\mathbf{Z}_m\}, \{b_{mp}\}} \quad \sum_{m=1}^M \sum_{k=1}^K \text{tr}(\mathbf{W}_{m,k}) + \sum_{m=1}^M \text{tr}(\mathbf{Z}_m)$$

**s.t.**

$$\text{tr}(\mathbf{H}_k \mathbf{W}_k) - \gamma_k \sum_{j \neq k} \text{tr}(\mathbf{H}_k \mathbf{W}_j) \geq \gamma_k \sigma_c^2, \quad \forall k$$

$$\text{tr}(\mathbf{G}_p \mathbf{Z}_p) \geq \gamma_S^{\text{PoD}} \sigma_s^2, \quad \forall p$$

$$\text{tr}(\mathbf{J}_p^{\text{data}}) \geq \Gamma_{\text{Track}, p}, \quad \forall p$$

$$\sum_{k=1}^K \text{tr}(\mathbf{W}_{m,k}) + \text{tr}(\mathbf{Z}_m) \leq P_{\max}, \quad \forall m$$

$$\sum_{m=1}^M b_{mp} = N_{\text{req}}, \quad \forall p$$

$$\mathbf{W}_{m,k} \succeq \mathbf{0}, \quad \mathbf{Z}_m \succeq \mathbf{0}, \quad b_{mp} \in \{0,1\}$$

**注**: 松弛后若最优解满足 $\text{rank}(\mathbf{W}_{m,k}) = 1$，则 SDR 精确。

### 7.2 固定 AP 选择子问题 (P3)

给定 AP-目标关联 $\{b_{mp}\}$，定义激活 AP 集合:

$$\mathcal{M}_p = \{m : b_{mp} = 1\}, \quad |\mathcal{M}_p| = N_{\text{req}}$$

$$\mathcal{M}^{\text{all}} = \bigcup_p \mathcal{M}_p$$

提取子信道并求解简化问题。

### 7.3 最小违反度形式 (P4)

$$\min_{\{\mathbf{w}_{m,k}\}, \{\mathbf{Z}_m\}, \{b_{mp}\}} \quad \text{violation}$$

其中:

$$\text{violation} = \sum_{k=1}^K \max(0, \gamma_k - \text{SINR}_k^{\text{wc}}) + \sum_{p=1}^P \max(0, \gamma_S^{\text{PoD}} - \text{SINR}_{S,p}) + \sum_{p=1}^P \max(0, \Gamma_{\text{Track}, p} - \text{tr}(\mathbf{J}_p^{\text{data}})) + \sum_{m=1}^M \max(0, P_m - P_{\max})$$

### 7.4 拉格朗日对偶 (概述)

**拉格朗日函数**:

$$\mathcal{L} = \sum_{m,k} \|\mathbf{w}_{m,k}\|_2^2 + \sum_m \text{tr}(\mathbf{Z}_m) + \sum_k \lambda_k (\gamma_k - \text{SINR}_k^{\text{wc}}) + \sum_p \mu_p (\gamma_S^{\text{PoD}} - \text{SINR}_{S,p}) + \sum_p \nu_p (\Gamma_{\text{Track}, p} - \text{tr}(\mathbf{J}_p^{\text{data}})) + \sum_m \omega_m (P_m - P_{\max})$$

**对偶问题**:

$$\max_{\boldsymbol{\lambda}, \boldsymbol{\mu}, \boldsymbol{\nu}, \boldsymbol{\omega} \geq \mathbf{0}} \quad g(\boldsymbol{\lambda}, \boldsymbol{\mu}, \boldsymbol{\nu}, \boldsymbol{\omega})$$

**注**: 由于 SINR 约束的非凸性，强对偶性一般不成立，对偶间隙 > 0。

---

## 8. Feasibility Analysis

### 8.1 可行性条件

#### 8.1.1 必要条件

**ZF 可行性**:

若采用 ZF 波束成形，需要:

$$\text{rank}(\mathbf{H}_{\mathcal{M}^{\text{all}}}) \geq K$$

即 $|\mathcal{M}^{\text{all}}| \cdot N_t \geq K$。

对于 $N_t = 4, K = 10$:

$$|\mathcal{M}^{\text{all}}| \geq \lceil 10/4 \rceil = 3$$

实际中需要 $|\mathcal{M}^{\text{all}}| \geq 4$ 以保证数值稳定性。

**功率可行性**:

总功率需求:

$$P_{\text{total}}^{\min} = P_{\text{comm}}^{\min} + P_{\text{sens}}^{\min}$$

其中:

$$P_{\text{comm}}^{\min} = \sum_{k=1}^K \frac{\gamma_k^{\text{robust}} \sigma_c^2}{|\mathbf{h}_k^H \mathbf{w}_k^{\text{ZF}}|^2}$$

$$P_{\text{sens}}^{\min} = \sum_{p=1}^P \frac{(\gamma_S^{\text{PoD}})^{\text{robust}} \sigma_s^2}{\|\mathbf{g}_p\|_2^2}$$

**可行条件**:

$$P_{\text{total}}^{\min} \leq M \cdot P_{\max}$$

(注意: 标准形式中 $P_{\max}$ 是单 AP 上限，系统总功率上限为 $M \cdot P_{\max}$)

#### 8.1.2 充分条件 (宽松)

若所有用户和目标都靠近 AP (距离 < 10m)，则信道增益大，功率需求小，可行性高。

### 8.2 瓶颈分析

根据实验结果，约束瓶颈优先级:

1. **感知 SINR** (最严格): 目标远离 AP 时路径损耗大
2. **通信 SINR** (中等): 多用户干扰和 CSI 误差
3. **单 AP 功率** (中等): 每 AP 功率上限 $P_{\max}$
4. **PCRB** (未充分验证): Fisher 信息矩阵迹约束
5. **AP 选择** (低): 通常可满足

### 8.3 成功率影响因素

| 参数 | 影响 | 敏感度 |
|------|------|--------|
| $\gamma_S^{\text{PoD}}$ | 高 | 3dB → 0dB: 70% → 95% |
| $P_{\max}$ | 中 | 30W → 40W: 95% → 100% |
| $M$ | 中 | M=16 → M=20: 95% → 100% |
| $K$ | 高 | K=10 → K=20: 95% → 50% |
| $N_{\text{req}}$ | 中 | 增加协作 AP 数提升分集 |
| 目标范围 | 中 | ±50m → ±20m: 95% → 100% |
| $\epsilon$ | 低 | 0.15 → 0.30: 95% → 75% |

---

## 9. Algorithmic Decomposition

### 9.1 分解策略

由于完整问题的非凸性和组合特性，采用以下分解:

```
Layer 1: AP Selection (组合优化)
    ↓
Layer 2: Power Split (连续优化)
    ↓
Layer 3: Beamforming (凸/闭式)
    ↓
Layer 4: Verification (约束检查)
```

### 9.2 各层算法

#### 9.2.1 AP 选择 (Layer 1)

**按目标选择**:

对于每个目标 $p \in \mathcal{P}^{\text{active}}$:

1. 计算 AP $m$ 到目标 $p$ 的信道强度: $\|\mathbf{g}_{m,p}\|_2^2$
2. 选择最强的 $N_{\text{req}}$ 个 AP: $b_{mp} = 1$
3. 其余 AP: $b_{mp} = 0$

**复杂度**: $O(MP \log M)$ (对每个目标排序)

**简化策略** (所有目标共享 AP 集合):

1. 计算联合得分: $\text{score}_m = \sum_p \|\mathbf{g}_{m,p}\|_2^2$
2. 选择前 $N_{\text{active}}$ 个 AP
3. 所有目标共享该集合

#### 9.2.2 通信波束 (Layer 3)

**ZF (Zero-Forcing)**:

$$\mathbf{W}_{\text{ZF}} = \mathbf{H}_{\mathcal{A}} (\mathbf{H}_{\mathcal{A}}^H \mathbf{H}_{\mathcal{A}})^{-1}$$

$$\mathbf{w}_k = \frac{\mathbf{W}_{\text{ZF}}(:,k)}{\|\mathbf{W}_{\text{ZF}}(:,k)\|_2} \cdot \sqrt{p_k}$$

**功率分配**:

$$p_k = \frac{\gamma_k^{\text{robust}} \sigma_c^2}{|\mathbf{h}_k^H \mathbf{w}_k^{\text{ZF}}|^2}$$

其中鲁棒门限:

$$\gamma_k^{\text{robust}} = \frac{\gamma_k (1+\epsilon_h)^2}{(1-\epsilon_h)^2}$$

**复杂度**: $O(K^3)$ (矩阵求逆)

**注**: ZF 是低复杂度启发式，非全局最优。最优需 SDR 或迭代优化。

#### 9.2.3 感知波束 (Layer 3)

**匹配滤波 (MRT)**:

$$\mathbf{z}_p = \sqrt{P_{\text{sens},p}} \frac{\mathbf{g}_p}{\|\mathbf{g}_p\|_2}$$

**功率分配**:

$$P_{\text{sens},p} = \frac{(\gamma_S^{\text{PoD}})^{\text{robust}} \sigma_s^2}{\|\mathbf{g}_p\|_2^2}$$

其中鲁棒门限:

$$(\gamma_S^{\text{PoD}})^{\text{robust}} = \frac{\gamma_S^{\text{PoD}} (1+\epsilon_g)^2}{(1-\epsilon_g)^2}$$

**复杂度**: $O(MN_t)$

**注**: 对于 rank-1 感知信道，匹配滤波是最优的。

#### 9.2.4 功率分配 (Layer 2)

**总功率**:

$$P_{\text{total}} = P_{\text{comm}} + P_{\text{sens}} = \sum_{k=1}^K p_k + \sum_{p=1}^P P_{\text{sens},p}$$

**单 AP 功率检查**:

对于每个 AP $m$:

$$P_m = \sum_{k=1}^K \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m) \leq P_{\max}$$

若 $P_m > P_{\max}$，需要缩放或重新分配功率。

**优化策略**:

1. 计算 $P_{\text{comm}}^{\min}$ 和 $P_{\text{sens}}^{\min}$
2. 若所有 AP 满足 $P_m \leq P_{\max}$，成功
3. 否则尝试减少 AP 数量或放宽门限

### 9.3 完整算法流程

```
算法: Cell-Free ISAC 求解器
输入: H, G, M, Nt, K, P, Pmax, γ_comm, γ_sens, ε_h, ε_g
输出: A*, W*, Z*, P_total*

1. 计算鲁棒门限:
   γ_comm^robust = γ_comm / η_h
   γ_sens^robust = γ_sens / η_g

2. for N_active in [16, 14, 12, 10, 8, 6, 4]:
3.   选择 top-N_active APs (按 score_m)
4.   提取子信道 H_A, G_A
5.   
6.   if rank(H_A) < K:
7.     continue  (ZF 不可行)
8.   
9.   计算 ZF 波束 W_zf = H_A * inv(H_A^H * H_A)
10.  归一化并分配功率: w_k = W_zf(:,k) / ||W_zf(:,k)|| * sqrt(p_k)
11.  P_comm = sum(p_k)
12.  
13.  计算匹配滤波感知波束: z_p = sqrt(P_sens,p) * g_p / ||g_p||
14.  P_sens = sum(P_sens,p)
15.  P_total = P_comm + P_sens
16.  
17.  if P_total <= P_max:
18.    验证所有约束 (SINR, SNR, CRB, 单AP功率)
19.    if 全部满足:
20.      记录解并 break

21. return 最优解
```

---

## 10. Complexity Analysis

### 10.1 完整问题 (P1)

**变量数**:
- 连续: $KMN_t + MN_t^2 + 1 = O(MN_t(K + N_t))$
- 二进制: $M$

**约束数**:
- SINR: $K$
- SNR: $P$
- CRB: $P$
- 功率: $M + 1$
- 半正定: $M$

**复杂度**: 非凸 + 组合 = NP-hard (一般情形)

**SDR 松弛后**: $O((MN_t)^{3.5})$ (内点法)

对于 $M=16, N_t=4$: $O(64^{3.5}) \approx O(10^6)$ — 不可行。

### 10.2 分解算法

| 步骤 | 操作 | 复杂度 |
|------|------|--------|
| AP 排序 | sort | $O(M \log M)$ |
| ZF 求逆 | matrix inversion | $O(K^3)$ |
| 功率分配 | K 次除法 | $O(K)$ |
| 感知波束 | 向量归一化 | $O(MN_t)$ |
| 约束验证 | 矩阵乘法 | $O(K^2 MN_t + P MN_t)$ |
| **总计** | | **$O(M \log M + K^3 + K^2 MN_t)$** |

对于 $M=16, N_t=4, K=10$:

$$O(16 \log 16 + 1000 + 100 \cdot 64) \approx O(7400)$$

**与 SDR 对比**: 复杂度降低约 $10^3$ 倍。

### 10.3 多 AP 候选搜索

若尝试 $N_{\text{active}} \in \{16, 14, 12, 10, 8, 6, 4\}$ (7 个值):

$$\text{总复杂度} = 7 \times O(M \log M + K^3) = O(7K^3)$$

对于 $K=10$: $7 \times 1000 = 7000$ 次操作 — 可忽略。

---

## 11. Current Implementation

### 11.1 已实现

| 组件 | 文件 | 状态 |
|------|------|------|
| 完整求解器 | `isac_final_solver.m` | ✅ ZF + 匹配滤波 + 多 AP 候选 |
| 参数化测试 | `run_experiments.m` | ✅ 7 组参数扫描 |
| 实验汇总 | `EXPERIMENT_SUMMARY.md` | ✅ |
| 问题定义 | `COMPLETE_PROBLEM_FORMULATION.md` | ✅ (本文档) |

### 11.2 未实现 (Gap)

| 组件 | 优先级 | 说明 |
|------|--------|------|
| SDR 求解器 | 低 | 复杂度太高，仅用于对比 |
| 严格 CRB 验证 | 中 | 当前使用简化形式 |
| 单 AP 功率约束 | 中 | 当前 $P_{m,\max} = P_{\max}$，不活跃 |
| 感知协方差 $Z_m$ | 低 | 当前使用波束形式 |
| 联合优化 $\rho$ | 低 | 当前直接计算最小功率 |
| 交替优化 | 中 | 可进一步提升性能 |
| 对偶分析 | 低 | 理论分析 |

### 11.3 验证清单

- [x] 通信 SINR 逐用户检查
- [x] 感知 SNR 逐目标检查 (简化)
- [x] 总功率检查
- [x] 鲁棒 CSI 近似 (安全余量)
- [ ] CRB 逐目标严格检查
- [ ] 单 AP 功率逐 AP 检查
- [ ] 半正定约束验证
- [ ] 对偶间隙分析

---

## 附录 A: 关键公式速查

### A.1 鲁棒性因子

$$\eta_h = \frac{(1-\epsilon_h)^2}{(1+\epsilon_h)^2}, \quad \eta_g = \frac{(1-\epsilon_g)^2}{(1+\epsilon_g)^2}$$

### A.2 ZF 功率分配

$$p_k = \frac{\gamma_{\text{comm}}^{\text{robust}} \sigma_c^2}{|\mathbf{h}_k^H \mathbf{w}_k^{\text{ZF}}|^2}, \quad \mathbf{w}_k^{\text{ZF}} = \frac{\mathbf{W}_{\text{ZF}}(:,k)}{\|\mathbf{W}_{\text{ZF}}(:,k)\|_2}$$

### A.3 匹配滤波感知功率

$$P_{\text{sens},p} = \frac{\gamma_{\text{sens}}^{\text{robust}} \sigma_s^2}{\|\mathbf{g}_p\|_2^2}$$

### A.4 Violation 计算

$$\text{violation} = \max\left(0, \frac{\gamma_{\text{comm}}}{\text{SINR}^{\text{wc}}} - 1, \frac{\gamma_{\text{sens}}}{\text{SNR}^{\text{wc}}} - 1, \frac{P_{\text{total}}}{P_{\max}} - 1\right)$$

### A.5 成功率定义

$$\text{Success} = \mathbb{1}(\text{violation} \leq 0)$$

$$\text{Success Rate} = \frac{1}{T} \sum_{t=1}^T \text{Success}_t$$

---

## 附录 B: 参数配置表

### B.1 默认配置 (大规模口径)

| 参数 | 值 | 说明 |
|------|-----|------|
| M | 16 | AP 数量 |
| Nt | 4 | 每 AP 天线数 |
| K | 10 | 通信用户数 |
| P | 4 | 感知目标数 |
| Pmax | 30 W | 总功率预算 |
| Pm,max | 30 W | 单 AP 功率上限 |
| γcomm | 0 dB | 通信 SINR 门限 |
| γsens | 0 dB | 感知 SNR 门限 |
| Γ | 1 | CRB 门限 |
| σc² | 10⁻⁹ W | 通信噪声 |
| σs² | 10⁻⁹ W | 感知噪声 |
| εh | 0.10 | 通信 CSI 误差 |
| εg | 0.15 | 感知 CSI 误差 |
| η | 2.5 | 路径损耗指数 |
| d0 | 10 m | 参考距离 |
| α | 0.5 | AP 选择权重 |
| Nactive | 动态 | 激活 AP 数 |

### B.2 严格工业验证口径

| 参数 | 值 | 说明 |
|------|-----|------|
| γcomm | 10 dB | 严格通信质量 |
| γsens | 10 dB | 严格感知质量 |
| Pmax | 3.2 W | 低功率场景 |
| εh | 0.05 | 高精度 CSI |
| εg | 0.05 | 高精度 CSI |

---

## 文档信息

- **作者**: Simple Yu
- **版本**: v2.0 Rigorous
- **创建日期**: 2026-06-16
- **Git Commit**: 待提交
- **状态**: 完整问题定义 + 等价形式 + 复杂度分析

---
