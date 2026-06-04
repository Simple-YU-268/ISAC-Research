# Cell-Free ISAC Complete Problem Formulation

本文档是本项目唯一保留的完整问题定义，统一描述 Cell-Free Integrated Sensing and Communication (ISAC) 系统的问题描述、系统模型、优化变量、目标函数和约束条件。

---

## 1. Problem Description

考虑一个 Cell-Free ISAC 系统。多个分布式接入点 (Access Points, APs) 同时服务多个通信用户，并对多个感知目标进行探测、定位或跟踪。系统需要联合设计通信波束、感知波束或感知协方差、AP 选择和通信/感知功率分配。

目标是在满足通信服务质量、感知质量、定位精度、AP 选择和功率预算约束的条件下，最小化系统总发射功率，或在固定功率预算下最大化整体可行性。

该问题本质上是一个联合非凸优化问题，主要非凸性来自：

- 通信 SINR 约束中的多用户干扰项
- 感知 SNR/CRB 约束中的二次型
- AP 选择变量的二进制约束
- 通信波束、感知波束和功率分配之间的耦合
- 不完美 CSI 下的最坏情况鲁棒约束

---

## 2. System Model

### 2.1 Network Topology

系统包含：

| Symbol | Description |
|---|---|
| `M` | AP 数量 |
| `Nt` | 每个 AP 的天线数 |
| `K` | 通信用户数量 |
| `P` | 感知目标数量 |
| `P_max` | 系统总发射功率预算 |
| `P_m,max` | 单个 AP 的最大发射功率 |

项目中推荐使用的大规模口径为：

```text
M = 16
K = 10
P = 4
Nt = 4
P_max = 30 W
```

典型二维部署：

```text
AP:     4 x 4 网格, 覆盖 [-60, 60] x [-60, 60] m^2
Users:  均匀随机分布在 [-50, 50] x [-50, 50] m^2
Targets: 均匀随机分布在 [-30, 30] x [-30, 30] m^2
```

---

### 2.2 Channel Model

AP `m` 到用户 `k` 的通信信道为：

```text
h_{m,k} in C^{Nt x 1}
```

AP `m` 到目标 `p` 的感知信道为：

```text
g_{m,p} in C^{Nt x 1}
```

采用几何路径损耗和瑞利衰落模型：

```text
h_{m,k} = sqrt(PL(d_{m,k})) * alpha_{m,k}
g_{m,p} = sqrt(PL(d_{m,p})) * beta_{m,p}
```

其中：

```text
alpha_{m,k}, beta_{m,p} ~ CN(0, I)
PL(d) = (d / d0)^(-eta)
d0 = 10 m
eta = 2.5
d >= 5 m
```

距离由二维欧氏距离给出：

```text
d_{m,k} = ||q_m - u_k||_2
d_{m,p} = ||q_m - r_p||_2
```

其中 `q_m` 是 AP 位置，`u_k` 是用户位置，`r_p` 是目标位置。

---

### 2.3 Imperfect CSI Model

系统考虑不完美信道状态信息 (CSI)。真实信道由估计信道和估计误差组成：

```text
h_{m,k} = h_hat_{m,k} + Delta h_{m,k}
g_{m,p} = g_hat_{m,p} + Delta g_{m,p}
```

误差集合采用有界不确定性模型：

```text
||Delta h_k|| <= epsilon_h ||h_hat_k||
||Delta g_p|| <= epsilon_g ||g_hat_p||
```

典型取值：

```text
epsilon_h = 0.10
epsilon_g = 0.15
```

---

## 3. Signal Model

### 3.1 Communication Signal

用户 `k` 的接收信号为：

```text
y_k = h_k^H w_k s_k + sum_{j != k} h_k^H w_j s_j + n_k
```

其中：

| Symbol | Description |
|---|---|
| `h_k` | 所有 AP 到用户 `k` 的堆叠信道 |
| `w_k` | 面向用户 `k` 的堆叠通信波束 |
| `s_k` | 用户 `k` 的数据符号，满足 `E[|s_k|^2] = 1` |
| `n_k` | 通信噪声，`n_k ~ CN(0, sigma_c^2)` |

通信 SINR 为：

```text
SINR_k =
|h_k^H w_k|^2 /
(sum_{j != k} |h_k^H w_j|^2 + sigma_c^2)
```

---

### 3.2 Sensing Signal

目标 `p` 的感知信号可表示为：

```text
y_p = sum_m g_{m,p}^H Z_m g_{m,p} + n_p
```

其中：

| Symbol | Description |
|---|---|
| `g_{m,p}` | AP `m` 到目标 `p` 的感知信道 |
| `Z_m` | AP `m` 的感知协方差矩阵 |
| `n_p` | 感知噪声，`n_p ~ CN(0, sigma_s^2)` |

若使用感知波束 `z_p`，目标 `p` 的感知 SNR 可写为：

```text
SNR_p =
|g_p^H z_p|^2 /
(sigma_s^2 ||z_p||^2)
```

若使用协方差形式，可写为：

```text
SNR_p =
sum_m |g_{m,p}^H Z_m g_{m,p}|^2 /
(sigma_s^2 sum_m tr(Z_m))
```

---

## 4. Optimization Variables

完整问题中的优化变量包括：

| Variable | Domain / Shape | Description |
|---|---|---|
| `W` | `C^{M x Nt x K}` | 通信波束矩阵 |
| `w_k` | `C^{M Nt x 1}` | 面向用户 `k` 的堆叠通信波束 |
| `Z_m` | `C^{Nt x Nt}` | AP `m` 的感知协方差矩阵 |
| `Z` | `{Z_m}_{m=1}^M` | 全部感知协方差 |
| `z_p` | `C^{M Nt x 1}` | 面向目标 `p` 的感知波束 |
| `a_m` | `{0, 1}` | AP `m` 是否被选择 |
| `a_{m,p}` | `{0, 1}` | AP `m` 是否服务目标 `p` |
| `rho` | `[0, 1]` | 通信功率占比 |

通信/感知功率分配：

```text
P_comm = rho P_max
P_sens = (1 - rho) P_max
```

---

## 5. Objective Function

最完整的功率最小化目标为：

```text
min_{W,Z,a,rho}
    ||W||_F^2 + sum_{m=1}^M tr(Z_m)
```

其中：

```text
||W||_F^2 = sum_m sum_k ||w_{m,k}||^2
```

在固定功率预算实验中，也可以等价地使用约束违反度最小化目标：

```text
min violation(W, Z, a, rho)
```

其中 violation 汇总通信 SINR 缺口、感知 SNR 缺口、CRB 超标和功率超标。

---

## 6. Constraints

### 6.1 Communication SINR Constraint

每个用户必须满足通信 SINR 门限：

```text
SINR_k >= gamma_comm, for all k = 1,...,K
```

常见门限：

```text
gamma_comm = 0 dB   大规模 v2.2 口径
gamma_comm = 10 dB  严格工业验证口径
```

---

### 6.2 Robust Communication Constraint

在不完美 CSI 下，要求最坏情况 SINR 仍满足门限：

```text
min_{||Delta h_k|| <= epsilon_h ||h_hat_k||}
SINR_k(W, h_hat_k + Delta h_k) >= gamma_comm,
for all k
```

常用保守近似：

```text
SINR_k^wc approx SINR_k^nominal * (1 - epsilon_h)^2 / (1 + epsilon_h)^2
```

---

### 6.3 Sensing SNR Constraint

每个目标必须满足感知 SNR 门限：

```text
SNR_p >= gamma_sens, for all p = 1,...,P
```

常见门限：

```text
gamma_sens = 3 dB   大规模 v2.2 口径
gamma_sens = 10 dB  严格工业验证口径
```

---

### 6.4 Robust Sensing Constraint

在目标信道不确定时，要求最坏情况感知 SNR 仍满足门限：

```text
min_{||Delta g_p|| <= epsilon_g ||g_hat_p||}
SNR_p(Z, g_hat_p + Delta g_p) >= gamma_sens,
for all p
```

---

### 6.5 CRB Constraint

定位或参数估计精度由 Cramer-Rao Bound (CRB) 约束表示：

```text
CRB_p <= Gamma, for all p = 1,...,P
```

项目中使用的简化形式为：

```text
CRB_p = sigma_s^2 / Fisher_p
```

其中一种 Fisher 信息近似为：

```text
Fisher_p = sum_m sum_k |g_{m,p}^H w_{m,k}|^2
```

因此：

```text
CRB_p =
sigma_s^2 /
(sum_m sum_k |g_{m,p}^H w_{m,k}|^2)
```

常见门限：

```text
Gamma = 1 m
Gamma = 10
```

CRB 约束也可转化为感知信号强度下界：

```text
|g_p^H z_p|^2 >= sigma_s^2 / Gamma
```

---

### 6.6 Total Power Constraint

系统总发射功率不能超过预算：

```text
sum_m (sum_k ||w_{m,k}||^2 + tr(Z_m)) <= P_max
```

即：

```text
||W||_F^2 + sum_m tr(Z_m) <= P_max
```

常见取值：

```text
P_max = 30 W   大规模 v2.2 口径
P_max = 3.2 W  严格工业验证口径
```

---

### 6.7 Per-AP Power Constraint

每个 AP 的发射功率也不能超过单 AP 上限：

```text
sum_k ||w_{m,k}||^2 + tr(Z_m) <= P_m,max,
for all m = 1,...,M
```

---

### 6.8 AP Selection Constraint

AP 选择变量为二进制：

```text
a_m in {0,1}
```

固定激活 AP 数：

```text
sum_m a_m = N_active
```

若按目标选择协作 AP：

```text
a_{m,p} in {0,1}
sum_m a_{m,p} = N_req, for all p
```

常见设置：

```text
N_req = 4
N_active = 7
```

AP 选择通常由通信和感知联合得分决定：

```text
score_m =
alpha sum_k ||h_{m,k}||^2
+ (1 - alpha) sum_p ||g_{m,p}||^2
```

选择得分最高的 AP。

---

### 6.9 Positive Semidefinite Constraint

若使用感知协方差矩阵，必须满足：

```text
Z_m >= 0, for all m
```

即每个 `Z_m` 都是 Hermitian positive semidefinite matrix。

---

### 6.10 Power Split Constraint

通信/感知功率分配因子满足：

```text
0 <= rho <= 1
```

并且：

```text
P_comm = rho P_max
P_sens = (1 - rho) P_max
```

---

## 7. Complete Optimization Problem

最完整的问题可统一写为：

```text
min_{W,Z,a,rho}
    ||W||_F^2 + sum_m tr(Z_m)

s.t.
    min_{Delta h_k in H_k}
    SINR_k(W, h_hat_k + Delta h_k)
    >= gamma_comm,
    for all k

    min_{Delta g_p in G_p}
    SNR_p(Z, g_hat_p + Delta g_p)
    >= gamma_sens,
    for all p

    CRB_p(W, Z, g_hat_p) <= Gamma,
    for all p

    ||W||_F^2 + sum_m tr(Z_m) <= P_max

    sum_k ||w_{m,k}||^2 + tr(Z_m) <= P_m,max,
    for all m

    sum_m a_m = N_active

    a_m in {0,1},
    for all m

    Z_m >= 0,
    for all m

    0 <= rho <= 1
```

其中不确定集合为：

```text
H_k = {Delta h_k : ||Delta h_k|| <= epsilon_h ||h_hat_k||}
G_p = {Delta g_p : ||Delta g_p|| <= epsilon_g ||g_hat_p||}
```

---

## 8. Success Criteria

### 8.1 Single-Slot Success

一个时隙成功当且仅当以下条件同时满足：

```text
SINR_k >= gamma_comm, for all k
SNR_p >= gamma_sens, for all p
CRB_p <= Gamma, for all p
P_total <= P_max
P_m <= P_m,max, for all m
AP selection constraint satisfied
```

### 8.2 Frame-Level Success

对于包含 `T` 个时隙的时帧：

```text
frame_success = all(slot_success_t for t = 1,...,T)
```

因此时帧成功率会随 `T` 增大而快速下降。

---

## 9. Algorithmic Decomposition

完整联合问题难以直接求解，因此项目采用分解和近似：

1. AP 选择：基于通信/感知信道强度的贪心选择。
2. 通信波束：使用 MMSE 或鲁棒 MMSE 闭式近似。
3. 感知波束：使用匹配滤波、SVD 或 CRB 引导的感知波束。
4. 功率分配：固定比例或优化 `rho`。
5. 鲁棒性：通过安全裕量、最坏情况近似和保守 AP 选择处理不完美 CSI。
6. 动态目标：通过 Kalman/EKF 预测目标位置，并使用 PC-CRLB 指导下一时隙波束设计。

---

## 10. Current Implementation Gap

当前仓库中不同脚本实现了该完整问题的不同子集。推荐入口 `src/v2.2/cellfree_isac_v22_robust_large.py` 主要实现：

- 16 AP, 10 users, 4 targets
- 二维几何信道
- AP 选择
- MMSE 通信波束
- 匹配滤波感知波束
- 通信 SINR 检查
- 总功率检查

但它没有完整实现以下验证：

- 感知 SNR 逐目标检查
- CRB 逐目标检查
- 每 AP 功率检查
- 严格的最坏情况鲁棒 CSI 约束
- 正半定协方差变量 `Z_m`

因此，后续算法和代码应以本文档的完整建模为准，并逐步补齐实现和验证。

新增 MATLAB 求解入口 `isac_complete_solver.m` 按本文档实现完整约束验证，并使用分解式近似流程搜索激活 AP 数和通信/感知功率分配。该脚本用于完整建模下的可行性求解和约束瓶颈诊断。
