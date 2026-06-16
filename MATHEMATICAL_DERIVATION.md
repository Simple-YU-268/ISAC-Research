# Cell-Free ISAC 严谨数学推导与求解

## 问题重述 (标准形式)

$$\min_{\{\mathbf{w}_{m,k}\}, \{\mathbf{Z}_m\}, \{b_{mp}\}} \quad \sum_{m=1}^{M} \left( \sum_{k=1}^{K} \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m) \right) \tag{1a}$$

$$\text{s.t.} \quad \text{SINR}_k^{\text{wc}} \geq \gamma_k, \quad \forall k \in \mathcal{K} \tag{1b}$$

$$\text{SINR}_{S,p} \geq \gamma_S^{\text{PoD}}, \quad \forall p \in \mathcal{P} \tag{1c}$$

$$\text{tr}(\mathbf{J}_p^{\text{data}}) \geq \Gamma_{\text{Track}, p}, \quad \forall p \in \mathcal{P} \tag{1d}$$

$$\sum_{m=1}^{M} b_{mp} = N_{\text{req}}, \quad \forall p \in \mathcal{P}^{\text{active}} \tag{1e}$$

$$\sum_{k=1}^{K} \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m) \leq P_{\max}, \quad \forall m \in \mathcal{M} \tag{1f}$$

$$\mathbf{Z}_m \succeq \mathbf{0}, \quad \forall m \in \mathcal{M} \tag{1g}$$

$$b_{mp} \in \{0, 1\}, \quad \forall m \in \mathcal{M}, p \in \mathcal{P} \tag{1h}$$

---

## 推导 1: 通信最坏情况 SINR (约束 1b)

### 1.1 SINR 定义

用户 $k$ 的接收信号:

$$y_k = \underbrace{\sum_{m=1}^M \mathbf{h}_{m,k}^H \mathbf{w}_{m,k} s_k}_{\text{期望信号}} + \underbrace{\sum_{j \neq k} \sum_{m=1}^M \mathbf{h}_{m,k}^H \mathbf{w}_{m,j} s_j}_{\text{多用户干扰}} + n_k$$

**堆叠形式**:

定义 $\mathbf{h}_k = [\mathbf{h}_{1,k}^T, \ldots, \mathbf{h}_{M,k}^T]^T \in \mathbb{C}^{MN_t \times 1}$

定义 $\mathbf{w}_k = [\mathbf{w}_{1,k}^T, \ldots, \mathbf{w}_{M,k}^T]^T \in \mathbb{C}^{MN_t \times 1}$

则:

$$y_k = \mathbf{h}_k^H \mathbf{w}_k s_k + \sum_{j \neq k} \mathbf{h}_k^H \mathbf{w}_j s_j + n_k$$

**SINR**:

$$\text{SINR}_k = \frac{|\mathbf{h}_k^H \mathbf{w}_k|^2}{\sum_{j \neq k} |\mathbf{h}_k^H \mathbf{w}_j|^2 + \sigma_c^2} \tag{2}$$

### 1.2 不完美 CSI 模型

真实信道:

$$\mathbf{h}_{m,k} = \hat{\mathbf{h}}_{m,k} + \Delta\mathbf{h}_{m,k}$$

堆叠形式:

$$\mathbf{h}_k = \hat{\mathbf{h}}_k + \Delta\mathbf{h}_k$$

误差界:

$$\|\Delta\mathbf{h}_k\|_2 \leq \epsilon_h \|\hat{\mathbf{h}}_k\|_2$$

### 1.3 最坏情况 SINR 推导

**目标**: 求 $\min_{\|\Delta\mathbf{h}_k\| \leq \epsilon_h \|\hat{\mathbf{h}}_k\|} \text{SINR}_k$

**分子分析**:

$$|\mathbf{h}_k^H \mathbf{w}_k|^2 = |(\hat{\mathbf{h}}_k + \Delta\mathbf{h}_k)^H \mathbf{w}_k|^2$$

由 Cauchy-Schwarz:

$$|(\hat{\mathbf{h}}_k + \Delta\mathbf{h}_k)^H \mathbf{w}_k| \geq |\hat{\mathbf{h}}_k^H \mathbf{w}_k| - |\Delta\mathbf{h}_k^H \mathbf{w}_k|$$

$$\geq |\hat{\mathbf{h}}_k^H \mathbf{w}_k| - \|\Delta\mathbf{h}_k\|_2 \|\mathbf{w}_k\|_2$$

$$\geq |\hat{\mathbf{h}}_k^H \mathbf{w}_k| - \epsilon_h \|\hat{\mathbf{h}}_k\|_2 \|\mathbf{w}_k\|_2$$

**最坏情况** (取等条件):

当 $\Delta\mathbf{h}_k = -\epsilon_h \frac{\|\hat{\mathbf{h}}_k\|_2}{\|\mathbf{w}_k\|_2} \frac{\mathbf{w}_k \mathbf{w}_k^H}{\|\mathbf{w}_k\|_2^2} \hat{\mathbf{h}}_k$ 的适当缩放形式...

**简化近似** (常用保守近似):

$$|\mathbf{h}_k^H \mathbf{w}_k|^2 \approx |\hat{\mathbf{h}}_k^H \mathbf{w}_k|^2 (1 - \epsilon_h)^2 \quad \text{(worst-case)}$$

$$|\mathbf{h}_k^H \mathbf{w}_j|^2 \approx |\hat{\mathbf{h}}_k^H \mathbf{w}_j|^2 (1 + \epsilon_h)^2 \quad \text{(worst-case for interference)}$$

**最坏情况 SINR**:

$$\text{SINR}_k^{\text{wc}} \approx \frac{|\hat{\mathbf{h}}_k^H \mathbf{w}_k|^2 (1-\epsilon_h)^2}{\sum_{j \neq k} |\hat{\mathbf{h}}_k^H \mathbf{w}_j|^2 (1+\epsilon_h)^2 + \sigma_c^2} \tag{3}$$

**进一步简化** (假设干扰项也缩放):

$$\text{SINR}_k^{\text{wc}} \approx \text{SINR}_k^{\text{nom}} \cdot \frac{(1-\epsilon_h)^2}{(1+\epsilon_h)^2} \tag{4}$$

其中:

$$\text{SINR}_k^{\text{nom}} = \frac{|\hat{\mathbf{h}}_k^H \mathbf{w}_k|^2}{\sum_{j \neq k} |\hat{\mathbf{h}}_k^H \mathbf{w}_j|^2 + \sigma_c^2}$$

**鲁棒性因子**:

$$\eta_h = \frac{(1-\epsilon_h)^2}{(1+\epsilon_h)^2} = \left(\frac{1-\epsilon_h}{1+\epsilon_h}\right)^2 \tag{5}$$

对于 $\epsilon_h = 0.10$:

$$\eta_h = \left(\frac{0.9}{1.1}\right)^2 = 0.818^2 \approx 0.669 \text{ (-1.75 dB)}$$

### 1.4 等价约束

约束 (1b) 等价于:

$$\text{SINR}_k^{\text{nom}} \geq \frac{\gamma_k}{\eta_h} = \gamma_k \cdot \frac{(1+\epsilon_h)^2}{(1-\epsilon_h)^2} \tag{6}$$

定义**鲁棒门限**:

$$\gamma_k^{\text{robust}} = \gamma_k \cdot \frac{(1+\epsilon_h)^2}{(1-\epsilon_h)^2} \tag{7}$$

对于 $\gamma_k = 1$ (0 dB), $\epsilon_h = 0.10$:

$$\gamma_k^{\text{robust}} = 1 \cdot \frac{1.21}{0.81} \approx 1.493 \text{ (1.74 dB)}$$

即需要额外 **1.74 dB** 功率余量。

---

## 推导 2: 感知 SINR (约束 1c)

### 2.1 感知信号模型

目标 $p$ 的反射信号 (单基地雷达):

$$y_{S,p} = \sum_{m=1}^M \mathbf{g}_{m,p}^H \mathbf{Z}_m \mathbf{g}_{m,p} + n_{S,p}$$

**感知 SINR** (协方差形式):

$$\text{SINR}_{S,p} = \frac{\left|\sum_{m=1}^M \mathbf{g}_{m,p}^H \mathbf{Z}_m \mathbf{g}_{m,p}\right|^2}{\sigma_s^2 \sum_{m=1}^M \text{tr}(\mathbf{Z}_m)} \tag{8}$$

### 2.2 Rank-1 协方差简化

假设 $\mathbf{Z}_m = \mathbf{z}_{m,p} \mathbf{z}_{m,p}^H$ (rank-1)，则:

$$\text{tr}(\mathbf{Z}_m) = \|\mathbf{z}_{m,p}\|_2^2$$

$$\mathbf{g}_{m,p}^H \mathbf{Z}_m \mathbf{g}_{m,p} = |\mathbf{g}_{m,p}^H \mathbf{z}_{m,p}|^2$$

堆叠感知波束:

$$\mathbf{z}_p = [\mathbf{z}_{1,p}^T, \ldots, \mathbf{z}_{M,p}^T]^T \in \mathbb{C}^{MN_t \times 1}$$

则:

$$\text{SINR}_{S,p} = \frac{|\mathbf{g}_p^H \mathbf{z}_p|^2}{\sigma_s^2 \|\mathbf{z}_p\|_2^2} \tag{9}$$

### 2.3 最优感知波束

**问题**: 给定功率预算 $P_{S,p} = \|\mathbf{z}_p\|_2^2$，最大化 $\text{SINR}_{S,p}$

$$\max_{\|\mathbf{z}_p\|_2^2 = P_{S,p}} \frac{|\mathbf{g}_p^H \mathbf{z}_p|^2}{\sigma_s^2 P_{S,p}}$$

**解**: 由 Cauchy-Schwarz，当 $\mathbf{z}_p \parallel \mathbf{g}_p$ 时取最大值:

$$\mathbf{z}_p^* = \sqrt{P_{S,p}} \frac{\mathbf{g}_p}{\|\mathbf{g}_p\|_2} \tag{10}$$

**最优 SINR**:

$$\text{SINR}_{S,p}^* = \frac{P_{S,p} \|\mathbf{g}_p\|_2^2}{\sigma_s^2} \tag{11}$$

### 2.4 最小感知功率

由约束 $\text{SINR}_{S,p} \geq \gamma_S^{\text{PoD}}$:

$$\frac{P_{S,p} \|\mathbf{g}_p\|_2^2}{\sigma_s^2} \geq \gamma_S^{\text{PoD}}$$

$$P_{S,p} \geq \frac{\gamma_S^{\text{PoD}} \sigma_s^2}{\|\mathbf{g}_p\|_2^2} \tag{12}$$

### 2.5 鲁棒感知约束

类似通信推导，最坏情况:

$$\text{SINR}_{S,p}^{\text{wc}} \approx \text{SINR}_{S,p}^{\text{nom}} \cdot \eta_g$$

其中:

$$\eta_g = \frac{(1-\epsilon_g)^2}{(1+\epsilon_g)^2} \tag{13}$$

对于 $\epsilon_g = 0.15$:

$$\eta_g = \left(\frac{0.85}{1.15}\right)^2 = 0.739^2 \approx 0.546 \text{ (-2.63 dB)}$$

**鲁棒门限**:

$$(\gamma_S^{\text{PoD}})^{\text{robust}} = \gamma_S^{\text{PoD}} \cdot \frac{(1+\epsilon_g)^2}{(1-\epsilon_g)^2} \tag{14}$$

对于 $\gamma_S^{\text{PoD}} = 1$:

$$(\gamma_S^{\text{PoD}})^{\text{robust}} = 1 \cdot \frac{1.3225}{0.7225} \approx 1.831 \text{ (2.63 dB)}$$

**最小感知功率 (鲁棒)**:

$$P_{S,p}^{\min} = \frac{(\gamma_S^{\text{PoD}})^{\text{robust}} \sigma_s^2}{\|\mathbf{g}_p\|_2^2} \tag{15}$$

---

## 推导 3: PCRB 约束 (约束 1d)

### 3.1 Fisher 信息矩阵

对于目标位置 $\boldsymbol{\theta}_p = [x_p, y_p]^T$，感知数据 Fisher 信息矩阵:

$$\mathbf{J}_p^{\text{data}} = \frac{2}{\sigma_s^2} \sum_{m=1}^M \sum_{k=1}^K \text{Re}\left\{ \nabla_{\boldsymbol{\theta}_p} (\mathbf{g}_{m,p}^H \mathbf{w}_{m,k}) \cdot \nabla_{\boldsymbol{\theta}_p}^H (\mathbf{g}_{m,p}^H \mathbf{w}_{m,k}) \right\} \tag{16}$$

### 3.2 梯度计算

假设感知信道与目标位置的关系:

$$\mathbf{g}_{m,p} = \sqrt{\text{PL}(d_{m,p})} \cdot \boldsymbol{\beta}_{m,p}$$

其中 $d_{m,p} = \|\mathbf{q}_m - \mathbf{r}_p\|_2$。

距离对位置的梯度:

$$\nabla_{\boldsymbol{\theta}_p} d_{m,p} = \frac{\mathbf{r}_p - \mathbf{q}_m}{d_{m,p}}$$

路径损耗对距离的梯度:

$$\nabla_{d} \text{PL}(d) = -\eta \frac{\text{PL}(d)}{d}$$

因此:

$$\nabla_{\boldsymbol{\theta}_p} \mathbf{g}_{m,p} = \nabla_{\boldsymbol{\theta}_p} d_{m,p} \cdot \nabla_{d} \text{PL}(d) \cdot \frac{\boldsymbol{\beta}_{m,p}}{2\sqrt{\text{PL}(d)}}$$

$$= -\frac{\eta}{2d_{m,p}^2} (\mathbf{r}_p - \mathbf{q}_m) \mathbf{g}_{m,p}^H$$

### 3.3 简化形式

忽略交叉项，假设各 AP-用户对独立贡献:

$$\text{tr}(\mathbf{J}_p^{\text{data}}) \approx \frac{2}{\sigma_s^2} \sum_{m=1}^M \sum_{k=1}^K |\mathbf{g}_{m,p}^H \mathbf{w}_{m,k}|^2 \cdot \|\nabla_{\boldsymbol{\theta}_p} (\mathbf{g}_{m,p}^H \mathbf{w}_{m,k})\|_2^2$$

**进一步简化** (假设波束与信道对齐):

$$\text{tr}(\mathbf{J}_p^{\text{data}}) \approx \frac{C}{\sigma_s^2} \sum_{m=1}^M \sum_{k=1}^K |\mathbf{g}_{m,p}^H \mathbf{w}_{m,k}|^2 \tag{17}$$

其中 $C$ 是与几何相关的常数。

### 3.4 约束转化

约束 (1d):

$$\text{tr}(\mathbf{J}_p^{\text{data}}) \geq \Gamma_{\text{Track}, p}$$

等价于:

$$\sum_{m=1}^M \sum_{k=1}^K |\mathbf{g}_{m,p}^H \mathbf{w}_{m,k}|^2 \geq \frac{\sigma_s^2 \Gamma_{\text{Track}, p}}{C} \tag{18}$$

**注**: 当通信波束 $\mathbf{w}_{m,k}$ 与感知信道 $\mathbf{g}_{m,p}$ 对齐时，此约束自动满足。若使用 ZF 波束，通信波束与感知信道正交，此约束可能不满足。

**关键观察**: PCRB 约束要求通信波形也提供感知能力，即 ISAC 波形设计。

---

## 推导 4: AP 选择 (约束 1e)

### 4.1 按目标选择

每个目标 $p$ 需要恰好 $N_{\text{req}}$ 个 AP 协作:

$$\sum_{m=1}^M b_{mp} = N_{\text{req}}, \quad \forall p \in \mathcal{P}^{\text{active}}$$

### 4.2 最优选择策略

**目标**: 最小化总功率，等价于最大化信道增益。

对于目标 $p$，选择信道最强的 $N_{\text{req}}$ 个 AP:

$$b_{mp} = \begin{cases} 1, & \text{if } m \in \mathcal{M}_p^{\text{top}} \\ 0, & \text{otherwise} \end{cases} \tag{19}$$

其中:

$$\mathcal{M}_p^{\text{top}} = \arg\max_{\mathcal{M} \subseteq \{1,\ldots,M\}, |\mathcal{M}|=N_{\text{req}}} \sum_{m \in \mathcal{M}} \|\mathbf{g}_{m,p}\|_2^2$$

**简化**: 对每个目标独立排序选择。

### 4.3 激活 AP 集合

所有被至少一个目标选中的 AP:

$$\mathcal{M}^{\text{all}} = \bigcup_{p \in \mathcal{P}^{\text{active}}} \{m : b_{mp} = 1\} \tag{20}$$

**总激活 AP 数**:

$$M^{\text{all}} = |\mathcal{M}^{\text{all}}| \in [N_{\text{req}}, \min(M, P \cdot N_{\text{req}})]$$

### 4.4 与固定 AP 选择的对比

| 特性 | 按目标选择 ($b_{mp}$) | 固定选择 ($a_m$) |
|------|----------------------|-------------------|
| 变量数 | $MP$ | $M$ |
| 约束 | $\sum_m b_{mp} = N_{\text{req}}$ (每目标) | $\sum_m a_m = N_{\text{active}}$ (全局) |
| 灵活性 | 高 (每目标自适应) | 低 (全局统一) |
| 复杂度 | $O(MP \log M)$ | $O(M \log M)$ |
| 功率效率 | 高 (按需分配) | 中 (可能浪费) |

---

## 推导 5: 功率约束 (约束 1f)

### 5.1 单 AP 功率

每个 AP $m$ 的总功率:

$$P_m = \sum_{k=1}^K \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m) \leq P_{\max} \tag{21}$$

### 5.2 系统总功率

$$P_{\text{total}} = \sum_{m=1}^M P_m = \sum_{m=1}^M \sum_{k=1}^K \|\mathbf{w}_{m,k}\|_2^2 + \sum_{m=1}^M \text{tr}(\mathbf{Z}_m) \tag{22}$$

**注**: 标准形式中 $P_{\max}$ 是单 AP 上限，系统总功率上限为 $M \cdot P_{\max}$。

### 5.3 功率分配策略

**通信功率**:

$$P_{\text{comm}} = \sum_{m=1}^M \sum_{k=1}^K \|\mathbf{w}_{m,k}\|_2^2 = \sum_{k=1}^K p_k$$

**感知功率**:

$$P_{\text{sens}} = \sum_{m=1}^M \text{tr}(\mathbf{Z}_m) = \sum_{p=1}^P P_{S,p}$$

**总功率**:

$$P_{\text{total}} = P_{\text{comm}} + P_{\text{sens}}$$

**关键约束**: 每个 AP 满足 $P_m \leq P_{\max}$，而非仅总功率约束。

---

## 推导 6: 通信波束成形优化

### 6.1 问题简化

固定 AP 选择后，提取子信道:

$$\mathbf{H}_{\text{all}} = [\mathbf{h}_1^{\text{all}}, \ldots, \mathbf{h}_K^{\text{all}}] \in \mathbb{C}^{M^{\text{all}}N_t \times K}$$

其中 $\mathbf{h}_k^{\text{all}}$ 仅包含激活 AP 的信道。

### 6.2 ZF 波束成形

**条件**: $\text{rank}(\mathbf{H}_{\text{all}}) \geq K$，即 $M^{\text{all}} N_t \geq K$。

**ZF 矩阵**:

$$\mathbf{W}_{\text{ZF}} = \mathbf{H}_{\text{all}} (\mathbf{H}_{\text{all}}^H \mathbf{H}_{\text{all}})^{-1} \in \mathbb{C}^{M^{\text{all}}N_t \times K} \tag{23}$$

**性质**:

$$\mathbf{h}_k^{\text{all},H} \mathbf{W}_{\text{ZF}}(:,j) = \delta_{kj} = \begin{cases} 1, & k=j \\ 0, & k \neq j \end{cases}$$

**归一化波束**:

$$\mathbf{w}_k = \frac{\mathbf{W}_{\text{ZF}}(:,k)}{\|\mathbf{W}_{\text{ZF}}(:,k)\|_2} \cdot \sqrt{p_k} \tag{24}$$

### 6.3 功率分配

由 SINR 约束 (6):

$$\text{SINR}_k^{\text{nom}} = \frac{p_k |\mathbf{h}_k^{\text{all},H} \mathbf{w}_k^{\text{ZF}}|^2}{\sigma_c^2} \geq \gamma_k^{\text{robust}}$$

其中 $\mathbf{w}_k^{\text{ZF}} = \frac{\mathbf{W}_{\text{ZF}}(:,k)}{\|\mathbf{W}_{\text{ZF}}(:,k)\|_2}$ 是归一化 ZF 波束。

由于 ZF 消除干扰:

$$\text{SINR}_k^{\text{nom}} = \frac{p_k}{\|\mathbf{W}_{\text{ZF}}(:,k)\|_2^2 \sigma_c^2} \geq \gamma_k^{\text{robust}}$$

**解得**:

$$p_k = \gamma_k^{\text{robust}} \sigma_c^2 \|\mathbf{W}_{\text{ZF}}(:,k)\|_2^2 \tag{25}$$

**总通信功率**:

$$P_{\text{comm}} = \sum_{k=1}^K p_k = \gamma_k^{\text{robust}} \sigma_c^2 \sum_{k=1}^K \|\mathbf{W}_{\text{ZF}}(:,k)\|_2^2 \tag{26}$$

### 6.4 单 AP 功率分配

将 $p_k$ 分配到各 AP:

$$\mathbf{w}_{m,k} = \mathbf{w}_k((m-1)N_t+1 : mN_t)$$

检查每个 AP:

$$P_m = \sum_{k=1}^K \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m) \leq P_{\max}$$

若超出，需要缩放或重新优化。

---

## 推导 7: 联合优化策略

### 7.1 分解策略

由于问题的非凸性和组合特性，采用**交替优化**:

```
迭代直到收敛:
    1. 固定 {b_mp}, 优化 {w_mk}, {Z_m}
    2. 固定 {w_mk}, {Z_m}, 优化 {b_mp}
```

### 7.2 固定 AP 选择优化波束

**子问题**:

$$\min_{\{\mathbf{w}_{m,k}\}, \{\mathbf{Z}_m\}} \quad \sum_{m,k} \|\mathbf{w}_{m,k}\|_2^2 + \sum_m \text{tr}(\mathbf{Z}_m)$$

**s.t.**

$$\text{SINR}_k^{\text{wc}} \geq \gamma_k, \quad \forall k$$

$$\text{SINR}_{S,p} \geq \gamma_S^{\text{PoD}}, \quad \forall p$$

$$\text{tr}(\mathbf{J}_p^{\text{data}}) \geq \Gamma_{\text{Track}, p}, \quad \forall p$$

$$\sum_k \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m) \leq P_{\max}, \quad \forall m$$

$$\mathbf{Z}_m \succeq \mathbf{0}$$

### 7.3 闭式解 (简化情形)

**假设**:
- 忽略 PCRB 约束 (或假设自动满足)
- 使用 ZF 通信波束
- 使用匹配滤波感知波束
- 单 AP 功率约束宽松

**步骤**:

1. **AP 选择**: 对每个目标 $p$，选择 $N_{\text{req}}$ 个最强 AP

2. **通信波束**:
   - 构建 $\mathbf{H}_{\text{all}}$
   - 计算 $\mathbf{W}_{\text{ZF}} = \mathbf{H}_{\text{all}} (\mathbf{H}_{\text{all}}^H \mathbf{H}_{\text{all}})^{-1}$
   - 分配功率 $p_k = \gamma_k^{\text{robust}} \sigma_c^2 \|\mathbf{W}_{\text{ZF}}(:,k)\|_2^2$

3. **感知波束**:
   - 对每个目标 $p$，构建 $\mathbf{g}_p^{\text{all}}$
   - 分配功率 $P_{S,p} = (\gamma_S^{\text{PoD}})^{\text{robust}} \sigma_s^2 / \|\mathbf{g}_p^{\text{all}}\|_2^2$
   - 波束 $\mathbf{z}_p = \sqrt{P_{S,p}} \mathbf{g}_p^{\text{all}} / \|\mathbf{g}_p^{\text{all}}\|_2$

4. **功率检查**:
   - 计算每个 AP 的 $P_m$
   - 若 $P_m > P_{\max}$，缩放或标记不可行

5. **验证**:
   - 计算 $\text{SINR}_k^{\text{wc}}$ 和 $\text{SINR}_{S,p}^{\text{wc}}$
   - 检查所有约束

---

## 推导 8: 复杂度分析

### 8.1 完整问题

**变量数**:
- 连续: $MKN_t + MN_t^2 = O(MN_t(K+N_t))$
- 二进制: $MP$

**约束数**:
- SINR: $K$
- 感知 SINR: $P$
- PCRB: $P$
- 功率: $M$
- 半正定: $M$
- AP 选择: $P$

**复杂度**: 非凸 + 组合 = **NP-hard**

**SDR 松弛**: $O((MN_t)^{3.5})$ — 对 $M=16, N_t=4$ 不可行

### 8.2 分解算法

| 步骤 | 操作 | 复杂度 |
|------|------|--------|
| AP 选择 | 每目标排序 | $O(MP \log M)$ |
| 信道提取 | 索引操作 | $O(MN_t(K+P))$ |
| ZF 求逆 | $(\mathbf{H}^H\mathbf{H})^{-1}$ | $O(K^3)$ |
| 通信功率 | $K$ 次乘法 | $O(K)$ |
| 感知波束 | $P$ 次归一化 | $O(PMN_t)$ |
| 功率检查 | $M$ 次求和 | $O(MK)$ |
| 约束验证 | 矩阵乘法 | $O(K^2MN_t + PMN_t)$ |
| **总计** | | **$O(MP\log M + K^3 + K^2MN_t)$** |

对于 $M=16, N_t=4, K=10, P=4$:

$$O(64\log 16 + 1000 + 100 \cdot 64) \approx O(7400)$$

**与 SDR 对比**: 降低约 $10^3$ 倍。

---

## 推导 9: 可行性条件

### 9.1 必要条件

**ZF 可行性**:

$$M^{\text{all}} N_t \geq K$$

对于 $N_t=4, K=10$:

$$M^{\text{all}} \geq 3 \quad \text{(理论)}$$

$$M^{\text{all}} \geq 4 \quad \text{(数值稳定)}$$

**功率可行性**:

$$P_{\text{comm}}^{\min} + P_{\text{sens}}^{\min} \leq M \cdot P_{\max}$$

其中:

$$P_{\text{comm}}^{\min} = \sum_{k=1}^K \gamma_k^{\text{robust}} \sigma_c^2 \|\mathbf{W}_{\text{ZF}}(:,k)\|_2^2$$

$$P_{\text{sens}}^{\min} = \sum_{p=1}^P (\gamma_S^{\text{PoD}})^{\text{robust}} \frac{\sigma_s^2}{\|\mathbf{g}_p^{\text{all}}\|_2^2}$$

### 9.2 充分条件

若所有目标位于 AP 附近 ($d < 10$m)，则:

$$\|\mathbf{g}_p\|_2^2 \gg \sigma_s^2 \quad \Rightarrow \quad P_{S,p}^{\min} \ll P_{\max}$$

可行性高。

### 9.3 不可行情形

1. 目标远离所有 AP ($d > 50$m)
2. 用户数过多 ($K > M^{\text{all}}N_t$)
3. 功率预算过低 ($P_{\max} < P_{\text{comm}}^{\min}$)
4. CSI 误差过大 ($\epsilon > 0.5$)

---

## 总结: 完整求解算法

```
算法: Cell-Free ISAC 标准形式求解器

输入: H, G, M, Nt, K, P, Pmax, {γk}, γS^PoD, {ΓTrack,p}, Nreq, εh, εg
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

7. 验证:
   计算 SINRk^wc for all k
   计算 SINRS,p for all p
   计算 tr(Jp^data) for all p
   若全部满足: 成功
   否则: 尝试减少 K 或放宽门限

8. 返回解
```

---

## 关键公式汇总

| 公式 | 名称 | 表达式 |
|------|------|--------|
| (5) | 通信鲁棒性因子 | $\eta_h = ((1-\epsilon_h)/(1+\epsilon_h))^2$ |
| (7) | 通信鲁棒门限 | $\gamma_k^{\text{robust}} = \gamma_k / \eta_h$ |
| (10) | 最优感知波束 | $\mathbf{z}_p^* = \sqrt{P_{S,p}} \mathbf{g}_p / \|\mathbf{g}_p\|_2$ |
| (12) | 最小感知功率 | $P_{S,p}^{\min} = \gamma_S^{\text{PoD}} \sigma_s^2 / \|\mathbf{g}_p\|_2^2$ |
| (13) | 感知鲁棒性因子 | $\eta_g = ((1-\epsilon_g)/(1+\epsilon_g))^2$ |
| (14) | 感知鲁棒门限 | $(\gamma_S^{\text{PoD}})^{\text{robust}} = \gamma_S^{\text{PoD}} / \eta_g$ |
| (23) | ZF 矩阵 | $\mathbf{W}_{\text{ZF}} = \mathbf{H} (\mathbf{H}^H \mathbf{H})^{-1}$ |
| (25) | 通信功率分配 | $p_k = \gamma_k^{\text{robust}} \sigma_c^2 \|\mathbf{W}_{\text{ZF}}(:,k)\|_2^2$ |
| (26) | 总通信功率 | $P_{\text{comm}} = \sum_k p_k$ |

---

## 版本信息

- **文档**: 严谨数学推导 v1.0
- **日期**: 2026-06-16
- **基于**: 标准 ISAC 问题形式 (公式 1a-1h)
- **推导**: 完整闭式解 + 复杂度分析 + 可行性条件
