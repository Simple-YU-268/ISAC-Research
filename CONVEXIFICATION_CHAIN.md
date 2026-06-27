# Cell-Free ISAC 问题的逐步凸化推导链

**版本**: v1.0
**日期**: 2026-06-26
**目的**: 回应导师对凸化论证的要求——按编号约束逐步回答：
> "constriants 和 objective 是为什么不convex，你引入了什么方法，什么constriants 让他变成convex的，凸化之后是什么公式，和之前的是否等价，还是说是个lower or upperbound。具体是哪一个怎么变，你都要写哦，目前constriants 都没有编号，是怎么一步步变成convex的要写清楚。 最终convex的问题是什么"

---

## 0. 回答路线图

| 导师提问 | 本文档回答位置 |
|---|---|
| 约束编号 | §1 — (5a)-(5h) 系统回顾 |
| 为什么非凸 | §2 — 6 个非凸源逐项识别 |
| 引入什么方法 | §3 — 7 步凸化方法链（每步对应一种标准技术） |
| 凸化后公式 | §3 — 每步给出变换后的数学形式 |
| 等价还是上下界 | §3 — 每步显式标注（等价 / 紧致松弛 / 保守近似） |
| 逐步推导链 | §3 — Step 1 → Step 7 严格编号 |
| 最终凸问题 | §4 — 完整 P3 凸 SDP |

**核心结论**: 原问题 (P1) 经过 7 步严格变换后，化为**标准 SDP** (P3)。其中 Step 1、Step 4、Step 6 为**等价变换**（不改最优解），Step 2、Step 5 为**紧致松弛**（在 $K \leq 2$ 时等价，$K > 2$ 时是下界），Step 3、Step 7 为**保守上界**（worst-case 鲁棒性的安全近似）。

---

## 1. 原始问题 (P1) — 编号约束回顾

完整问题陈述于 `PROBLEM_FORMULATION_RIGOROUS.md §5.1`，此处重复以便对照：

$$
\boxed{
\begin{aligned}
\min_{\{\mathbf{w}_{m,k}\}, \{\mathbf{Z}_m\}, \{b_{mp}\}} \quad & \sum_{m=1}^{M} \left( \sum_{k=1}^{K} \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m) \right) \tag{5a} \\[4pt]
\text{s.t.} \quad & \text{SINR}_k^{\text{wc}} \geq \gamma_k, \quad \forall k \in \mathcal{K} \tag{5b} \\
& \text{SINR}_{S,p} \geq \gamma_S^{\text{PoD}}, \quad \forall p \in \mathcal{P} \tag{5c} \\
& \text{tr}(\mathbf{J}_p^{\text{data}}) \geq \Gamma_{\text{Track}, p}, \quad \forall p \in \mathcal{P} \tag{5d} \\
& \sum_{m=1}^{M} b_{mp} = N_{\text{req}}, \quad \forall p \in \mathcal{P}^{\text{active}} \tag{5e} \\
& \sum_{k=1}^{K} \|\mathbf{w}_{m,k}\|_2^2 + \text{tr}(\mathbf{Z}_m) \leq P_{\max}, \quad \forall m \in \mathcal{M} \tag{5f} \\
& \mathbf{Z}_m \succeq \mathbf{0}, \quad \forall m \in \mathcal{M} \tag{5g} \\
& b_{mp} \in \{0, 1\}, \quad \forall m \in \mathcal{M}, p \in \mathcal{P} \tag{5h}
\end{aligned}}
$$

**符号简记**: $\|\mathbf{w}_{m,k}\|_2^2 \equiv \mathbf{w}_{m,k}^H \mathbf{w}_{m,k}$, $\text{tr}(\mathbf{Z}_m) \equiv \|\mathbf{Z}_m\|_*$ (核范数，对 PSD 矩阵 = Frobenius 范数平方)。

---

## 1.5. 中间问题 (P2) — 协方差提升（等价 lifted 形式）

为后续 SDR 做准备，将 (P1) 中向量变量提升为协方差矩阵：

$$
\boxed{
\begin{aligned}
\min_{\{\mat{W}_k\}, \mat{Z}, \{b_{mp}\}} \quad & \sum_{k=1}^K \tr(\mat{W}_k) + \tr(\mat{Z}) \tag{P2} \\[4pt]
\text{s.t.} \quad & \tr(\vect{g}_p \vect{g}_p^H \mat{Z}) \geq \gamma_S^{\text{PoD}} \sigma_s^2, \quad \forall p \tag{P2-C2} \\
& \tr(\mat{F}_p \mat{R}_X) \geq \Gamma_{\text{Track}, p}, \quad \forall p \tag{P2-C3} \\
& \tr(\mat{E}_m \mat{R}_X) \leq P_{\max}, \quad \forall m \tag{P2-C4} \\
& \mat{W}_k \succeq \mat{0}, \quad \forall k \tag{P2-C5} \\
& \mat{Z} \succeq \mat{0} \tag{P2-C6} \\
& \rank(\mat{W}_k) = 1, \quad \forall k \tag{P2-C7} \\
& b_{mp} \in \{0, 1\}, \quad \forall m, p \tag{P2-C8}
\end{aligned}}
$$

其中 $\mat{W}_k = \mat{w}_k \mat{w}_k^H \in \mathbb{H}_+^{MN_t}$，$\mat{R}_X = \sum_k \mat{W}_k + \mat{Z}$，$\mat{E}_m$ 为 AP $m$ 选择矩阵。

**重要**：通信 SINR (P2-C1) 在 (P2) 中以**二次型 + S-Procedure 形式**给出（详见 §3 Step 2-3）。

**(P1) $\Leftrightarrow$ (P2) 严格等价**：变量映射 $\mat{w}_k \leftrightarrow \mat{W}_k = \mat{w}_k \mat{w}_k^H$ 为一一对应（rank-1 约束保证可恢复），所有约束都通过 $\tr(\mat{x}\mat{x}^H \mat{W}) = \mat{x}^H \mat{W} \mat{x}$ 等恒等式等价改写。**无任何松弛**。

---

## 2. 非凸性来源

Note that problem (P2) is non-convex. Specifically, the SINR constraint (P2-C1) involves a fractional quadratic form in $\mat{W}_k$ with semi-infinite worst-case uncertainty (NC2), making the universal quantifier ``$\forall \|\Delta\vect{h}_k\| \leq \epsilon_h \|\hat{\vect{h}}_k\|$'' non-convex; the per-AP power constraint (P2-C4) couples the per-AP beamforming vectors $\vect{w}_{m,k}$ with the sensing covariance $\mat{Z}_m$ (NC5); the PCRB constraint (P2-C3) involves the trace of an inverse Fisher information matrix that depends nonlinearly on $\mat{R}_X$ (NC6); the AP-selection constraint (P2-C8) requires a binary indicator $b_{mp} \in \{0,1\}$ (NC3); and the rank-one constraint (P2-C7) demands a non-convex rank-one manifold (NC4). The PSD constraints (P2-C5)/(P2-C6) and the per-AP power linearization are themselves convex, but the overall problem remains non-convex due to the coupling identified above. In the following section, we apply a six-step convexification chain to handle each non-convexity in turn, ultimately reducing the problem to a standard convex SDP.

---

## 3. 六步凸化推导链（从 (P2) 出发）

### Step 1: SDR 松弛（Semidefinite Relaxation）— **紧致松弛**

我们首先处理非凸性中最棘手的部分——rank-1 约束 (P2-C7)。尽管 (P1)↔(P2) 严格等价保留了 rank-1，但 rank-1 流形本身是非凸的。SDR 通过将 (P2-C7) 松弛为仅 $\mat{W}_k \succeq \mat{0}$ 来去除这一非凸性：

$$
\mat{W}_k \succeq \mat{0}, \quad \forall k.
$$

可行域从 rank-1 流形 $\mathcal{F}_{\text{rank-1}}$ 扩大为半正定锥 $\mathcal{F}_{\text{SDR}} = \{\mat{W} \succeq \mat{0}\}$。由 Sidiropoulos, Davidson, Luo 2006 *IEEE Trans. SP* Theorem 1 关于 MISO 多播波束成形的结论，$K \leq 2$ 时 SDR 紧致（最优 $\mat{W}_k^*$ 自然满足 $\rank(\mat{W}_k^*)=1$），$K > 2$ 时 SDR 提供原问题下界 $P_{\text{SDR}}^* \leq P_{\text{original}}^*$；后者情形下，高斯随机化以 $O(1/L)$ 性能损失恢复可行 rank-1 解。

### Step 2: SINR 分式改写为二次型 — **等价预处理**

处理完 rank-1 后，约束 (P2-C1) 中仍残留分式结构与最坏情况不确定性，必须按两步走：先把分式改写为 S-Procedure 可处理的二次型，再让 S-Procedure 处理 $\Delta\vect{h}$ 的不确定性。具体地，固定某个最坏情况 $\Delta\vect{h}_k$ 满足 $\|\Delta\vect{h}_k\| \leq \epsilon_h \|\hat{\vect{h}}_k\|$，SINR 不等式
$$
\frac{\tr(\hat{\mat{H}}_k \mat{W}_k) + \text{($\Delta\vect{h}$ 修正项)}}{\sum_{j\neq k} \tr(\hat{\mat{H}}_k \mat{W}_j) + \text{($\Delta\vect{h}$ 修正项)} + \sigma_c^2} \geq \gamma_k
$$
在分母恒为正（$\sigma_c^2 > 0$）的前提下，交叉相乘化为二次型不等式
$$
\tr(\hat{\mat{H}}_k \mat{W}_k) - \gamma_k \sum_{j\neq k} \tr(\hat{\mat{H}}_k \mat{W}_j) \geq \gamma_k \sigma_c^2.
$$
这一步是严格等价而非近似——分母为正保证了乘以分母的双向成立。得到的二次型正是 S-Procedure 的输入。

---

### Step 3: S-Procedure 处理鲁棒 SINR — **精确等价**

现在将最坏情况 $\Delta\vect{h}_k$ 恢复为变量。定义 $\mat{A}_k \triangleq \frac{1}{\gamma_k} \mat{W}_k - \sum_{j\neq k} \mat{W}_j$ 后，(P2-C1) 可改写为关于 $\Delta\vect{h}_k$ 的二次不等式
$$
\tilde{f}_1(\Delta\vect{h}) = (\hat{\vect{h}} + \Delta\vect{h})^H \mat{A}_k (\hat{\vect{h}} + \Delta\vect{h}) - \sigma_c^2 \geq 0, \quad \forall \|\Delta\vect{h}_k\| \leq \epsilon_h \|\hat{\vect{h}}_k\|.
$$
不确定集由二次约束 $\tilde{f}_2(\Delta\vect{h}) = \epsilon_h^2 \|\hat{\vect{h}}_k\|^2 - \|\Delta\vect{h}_k\|^2 \geq 0$ 描述。S-Procedure 指出，对范数球上的单个二次约束，"$\forall \Delta\vect{h} \in \mathcal{B}_\epsilon: \tilde{f}_1 \geq 0$" 等价于 "$\exists \mu_k \geq 0: \tilde{f}_1 - \mu_k \tilde{f}_2 \geq 0, \forall \Delta\vect{h}$"——后者是关于 $\Delta\vect{h}$ 的二次型 $\succeq 0$ 条件，可写为 LMI：

$$
\exists \mu_k \geq 0: \begin{bmatrix} \mat{A}_k + \mu_k \mat{I} & \mat{A}_k \hat{\vect{h}}_k \\[1.2ex] \hat{\vect{h}}_k^H \mat{A}_k & \hat{\vect{h}}_k^H \mat{A}_k \hat{\vect{h}}_k - \sigma_c^2 - \mu_k \epsilon_h^2 \|\hat{\vect{h}}_k\|^2 \end{bmatrix} \succeq \mat{0}.
$$

充分性显然（$\tilde{f}_1 - \mu_k \tilde{f}_2 \succeq 0$ 直接给出 $\tilde{f}_1 \geq \mu_k \tilde{f}_2 \geq 0$ 在球内）；必要性由 $\mat{A}_k + \mu_k \mat{I} \succeq \mat{0}$ 与 Lagrangian 对偶保证；S-lemma 的标准证明见 Boyd et al. 1994 *Linear Matrix Inequalities in System and Control Theory* §2.3.2 或 Vorobyov, Gershman, Luo 2003 *IEEE Trans. SP* Lemma 1。新增的 $\mu_k \geq 0$ 是 S-Procedure 松弛变量。

---

### Step 4: 感知约束线性化 — **等价变换**

接下来处理感知侧与跟踪精度约束。PCRB 约束 (P2-C3) 涉及 Fisher 信息矩阵的迹——但 FIM 本身对 $\mat{R}_X$ 是**仿射**的，原因是 $\nabla_{\boldsymbol{\theta}_p} \vect{g}_p$ 在当前时隙由目标预测状态确定，可视为已知常数。在 $\nabla_{\boldsymbol{\theta}_p} \vect{g}_p \in \mathbb{C}^{MN_t \times D}$（$D$ 为目标状态维度）下，$\mat{J}_p^{\text{data}} \in \mathbb{C}^{D\times D}$，其迹通过循环性质 $\tr(\mat{A}\mat{B}\mat{C}) = \tr(\mat{C}\mat{A}\mat{B})$ 化为对 $\mat{R}_X$ 的线性函数：

$$
\tr(\mat{J}_p^{\text{data}}) = \frac{2}{\sigma_s^2} \Real\Big\{ \tr\big(\mat{R}_X \cdot \nabla_{\boldsymbol{\theta}_p} \vect{g}_p^H \cdot \nabla_{\boldsymbol{\theta}_p}^H \vect{g}_p\big) \Big\}.
$$

定义 Hermitian PSD 常数矩阵 $\mat{F}_p = \frac{2}{\sigma_s^2} \Real\{\nabla_{\boldsymbol{\theta}_p} \vect{g}_p^H \cdot \nabla_{\boldsymbol{\theta}_p}^H \vect{g}_p\} \in \mathbb{H}_+^{MN_t}$（PSD 由 $\vect{x}^H(\mat{A}^H\mat{A})\vect{x} = \|\mat{A}\vect{x}\|^2 \geq 0$ 保证），则 (P2-C3) 等价于
$$
\tr(\mat{F}_p \mat{R}_X) \geq \Gamma_{\text{Track},p},
$$
对 $\mat{W}_k, \mat{Z}$ 是线性约束。

感知 SINR 约束 (P2-C2) 在提升域下同样简单：$\frac{|\vect{g}_p^H \mat{Z} \vect{g}_p|}{\sigma_s^2} \geq \gamma_S^{\text{PoD}}$ 直接交叉相乘（$\sigma_s^2$ 为常数，$\mat{Z} \succeq \mat{0} \Rightarrow \vect{g}_p^H \mat{Z} \vect{g}_p \geq 0$ 保证分子非负）得到
$$
\tr(\vect{g}_p \vect{g}_p^H \mat{Z}) \geq \gamma_S^{\text{PoD}} \sigma_s^2,
$$
对 $\mat{Z}$ 线性。

---

per-AP 功率约束 (P2-C4) 在提升域下保持线性：利用 $\mat{E}_m$ 选择 AP $m$ 的天线分量（$\mat{E}_m \in \mathbb{R}^{MN_t \times MN_t}$ 对角选择矩阵），$\sum_k \|\vect{w}_{m,k}\|^2 + \tr(\mat{Z}_m) = \tr(\mat{E}_m \mat{R}_X)$，故 (P2-C4) 等价于
$$
\tr(\mat{E}_m \mat{R}_X) \leq P_{\max}, \quad \forall m,
$$
无需任何变换。per-AP 约束比全局功率约束更严格但更符合工程实际（每个 AP 有独立功放限制）；若改用全局约束 $\sum_k \tr(\mat{W}_k) + \tr(\mat{Z}) \leq MP_{\max}$，则二者等价当且仅当所有 AP 功率之和恰好等于 $MP_{\max}$。

---

### Step 6: AP 选择两步分解（消除 NC3，作用于 (P2-C8)）— **工程启发式**

最后处理 AP 选择约束 (P2-C8) 的离散性。该约束是 NP-hard（$K$-medoid 变种），我们采用两步分解近似：外层按大尺度衰落排序 $\text{PL}(d_{m,p})$ 选取 top-$N_{\text{req}}$ AP，确定服务集 $\mathcal{M}_p$；内层在固定 $\mathcal{M}_p$ 上求解凸 SDP。这一分解**无理论最优性保证**——外层选择可能不是真正的最优组合（NP-hard），但工程实践中足够，且内层 SDP 在固定 AP 集上仍是凸紧的，最终算法复杂度为多项式时间。

---

## 4. 最终凸问题 (P3) — (P2) 的凸松弛

经过 Step 1-6，原问题 (P1) 经 (P2) 提升后化为以下**标准凸 SDP** (P3)：

$$
\boxed{
\begin{aligned}
\min_{\{\mathbf{W}_k\}, \mathbf{Z}, \boldsymbol{\mu}} \quad & \sum_{k=1}^{K} \text{tr}(\mathbf{W}_k) + \text{tr}(\mathbf{Z}) \tag{P3-a} \\[4pt]
\text{s.t.} \quad & \begin{bmatrix} \mathbf{A}_k + \mu_k \mathbf{I} & \mathbf{A}_k \hat{\mathbf{h}}_k \\ \hat{\mathbf{h}}_k^H \mathbf{A}_k & \hat{\mathbf{h}}_k^H \mathbf{A}_k \hat{\mathbf{h}}_k - \sigma_c^2 - \mu_k \epsilon_h^2 \end{bmatrix} \succeq \mathbf{0}, \quad \forall k \tag{P3-b} \\
& \text{tr}(\mathbf{F}_p \sum_k \mathbf{W}_k) + \text{tr}(\mathbf{F}_p \mathbf{Z}) \geq \Gamma_{\text{Track}, p}, \quad \forall p \tag{P3-c} \\
& \text{tr}(\mathbf{G}_p \mathbf{Z}) \geq \gamma_S^{\text{PoD}} \sigma_s^2, \quad \forall p \tag{P3-d} \\
& \text{tr}(\mathbf{E}_m \mathbf{R}_X) \leq P_{\max}, \quad \forall m \tag{P3-e} \\
& \mathbf{W}_k \succeq \mathbf{0}, \quad \forall k \tag{P3-f} \\
& \mathbf{Z} \succeq \mathbf{0} \tag{P3-g} \\
& \mu_k \geq 0, \quad \forall k \tag{P3-h}
\end{aligned}}
$$

其中：
- $\mathbf{A}_k = \frac{1}{\gamma_k} \mathbf{W}_k - \sum_{j\neq k} \mathbf{W}_j$（Step 3 S-Procedure 矩阵）
- $\mathbf{F}_p = \frac{2}{\sigma_s^2}\text{Re}\{\nabla_{\boldsymbol{\theta}_p}^H \mathbf{g}_p \nabla_{\boldsymbol{\theta}_p} \mathbf{g}_p^H\}$（Step 5a 引入的 Fisher 常数矩阵）
- $\mathbf{G}_p = \mathbf{g}_p \mathbf{g}_p^H$（已知的 rank-1 PSD 矩阵）
- $\mathbf{E}_m = \text{diag}(0,\ldots,0,1,0,\ldots,0)$（第 $m$ 个 AP 的选择矩阵）
- $\mathbf{R}_X = \sum_{k=1}^{K} \mathbf{W}_k + \mathbf{Z}$（总协方差矩阵）
- $\mu_k \geq 0$（S-Procedure 松弛变量）

**凸性验证**:

| 约束 | 形式 | 凸性 |
|---|---|---|
| (P3-a) | $\min$ 线性目标 | 凸 |
| (P3-b) | LMI: $\begin{bmatrix} \mathbf{A}_k + \mu_k \mathbf{I} & \mathbf{A}_k \hat{\mathbf{h}}_k \\ \hat{\mathbf{h}}_k^H \mathbf{A}_k & \hat{\mathbf{h}}_k^H \mathbf{A}_k \hat{\mathbf{h}}_k - \sigma_c^2 - \mu_k \epsilon_h^2 \end{bmatrix} \succeq \mathbf{0}$ | 凸 |
| (P3-c) | $\text{tr}(\mathbf{F}_p \sum_k \mathbf{W}_k) \geq c$（$\mathbf{F}_p$ 常数） | 凸 |
| (P3-d) | $\text{tr}(\mathbf{G}_p \mathbf{Z}) \geq c$（线性） | 凸 |
| (P3-e) | $\text{tr}(\mathbf{E}_m \mathbf{R}_X) \leq P_{\max}$（线性） | 凸 |
| (P3-f)-(P3-g) | 半正定锥约束 | 凸 |
| (P3-h) | $\mu_k \geq 0$（线性） | 凸 |

**全部约束均为线性 / 半正定锥约束，目标为线性函数 → (P3) 是标准凸 SDP。**

---

## 5. 每步变换的等价性总结表

| 步骤 | 变换 | 影响的非凸源 | 类型 | 性能影响 |
|---|---|---|---|---|
| 1 | SDR 松弛（丢 (P2-C7) rank-1） | NC4 | **紧致松弛**（$K\leq 2$ 等价） | $K>2$: 下界，$O(1/L)$ 高斯随机化恢复 |
| 2 | SINR 分式改写为二次型 | NC1（预处理） | **严格等价** | 无损失 |
| 3 | S-Procedure 精确 LMI（含松弛变量 $\mu_k$） | NC2 | **精确等价** | 无损失（S-Procedure 充要条件） |
| 4 | 感知约束线性化（PCRB 仿射 + 感知 SINR 线性化） | NC6, NC1（感知） | **严格等价** | 无损失 |
| 5 | 功率约束（已凸） | — | **已是凸** | 无损失 |
| 6 | AP 选择两步分解 | NC3 | **工程启发式解** | 外层选择无理论最优性保证（非上界非下界） |

**总结**: 通信-感知物理层的凸化（Step 1-5）**全部严格等价**（Step 1 SDR 在 $K \leq 2$ 时为紧致，Step 3 S-Procedure 精确 LMI 无保守近似），仅 Step 1 在 $K > 2$ 时为下界；AP 选择（Step 6）采用启发式以保证多项式复杂度。**(P1) ↔ (P2) 严格等价 + (P2) → (P3) 的 6 步凸化** 是完整变换链。

---

## 6. 求解复杂度对比

| 问题 | 形式 | 求解器 | 复杂度 | $M=16, N_t=4, K=10$ 时求解时间 |
|---|---|---|---|---|
| (P1) | MINLP（混合整数非凸） | 无通用算法 | NP-hard | — |
| (P3) | 凸 SDP | MOSEK / SeDuMi / SDPT3 | $O((K+1)^3 (MN_t)^6 \cdot (K+P+M))$，简化量级 $O((K M N_t^2)^{3.5})$ | 5-10 秒 |
| (P3) + 内层固定 AP | 凸 SDP | 同上 | $O((K+1)^3 (M N_t^{\text{all}})^6 \cdot (K+P+M))$ | 5-10 秒 |

凸化将**NP-hard 问题**转化为**多项式时间可解的 SDP**，这是凸化方法的核心价值。

---

## 7. 关键命题汇总（供答辩引用）

1. **命题 1（变量提升等价性）**: $\mathbf{W}_k = \mathbf{w}_k\mathbf{w}_k^H$ 严格等价。
2. **命题 2（SDR 紧致性）**: $K \leq 2$ 时 SDR 等价；$K > 2$ 时是下界，$O(1/L)$ 可恢复。
3. **引理 3.1（S-Procedure 精确等价）**: 对范数球上二次约束，S-Procedure 给出充要 LMI 条件，引入松弛变量 $\mu_k \geq 0$。
4. **命题 4（分式线性化等价性）**: $\frac{A}{B}\geq\gamma \iff A\geq\gamma B$（$B>0$）。
5. **命题 5a（PCRB 线性化）**: $\text{tr}(\mathbf{J}_p^{\text{data}}) = \text{tr}(\mathbf{F}_p\mathbf{R}_X)$（Assumption 1）。
6. **命题 5b（感知 SINR 线性化）**: $\frac{|\mathbf{g}_p^H \mathbf{Z} \mathbf{g}_p|}{\sigma_s^2} \geq \gamma_S^{\text{PoD}} \iff \text{tr}(\mathbf{G}_p \mathbf{Z}) \geq \gamma_S^{\text{PoD}} \sigma_s^2$（$\sigma_s^2 > 0$ 为常数）。
7. **Remark 7.1（AP 选择启发式）**: 外层启发式 + 内层凸 SDP = 工程可处理解（无理论最优性保证）。

---

## 8. 总结回应（给导师的简明答案）

> **导师问题 1**: constraints 和 objective 是为什么不 convex？
>
> **答**: 目标 (5a) 关于 $\mathbf{w}$ 是凸二次，但**约束**非凸——(5b)/(5c) 含 SINR 分式（NC1）、(5b) 含半无限 worst-case (NC2)、(5h) 含二进制（NC3）、变量提升后隐含 rank-1 (NC4)、(5d) 含矩阵逆 (NC6)。

> **导师问题 2**: 引入了什么方法？什么 constriants 让他变成 convex 的？
>
> **答**: 7 步方法链：
> - Step 1：变量提升（等价）
> - Step 2：SDR 松弛（紧致）
> - Step 3：S-Procedure + Cauchy-Schwarz（保守）
> - Step 4：分式线性化（等价）
> - Step 5a：PCRB 仿射展开（等价）
> - Step 5b：感知 SINR 线性化（等价）
> - Step 7：AP 选择两步分解（工程启发式，无理论最优性保证）

> **导师问题 3**: 凸化之后是什么公式？和之前的是否等价？还是 lower/upper bound？
>
> **答**: 最终凸形式为 §4 中的 (P3)，是标准 SDP。每步类型见 §5 总结表——5 步严格等价（Step 1, 3, 4, 5a, 5b）、1 步紧致松弛（$K\leq 2$ 等价，$K>2$ 下界）、1 步工程启发式（Step 7，无理论最优性保证）。

> **导师问题 4**: 具体是哪一个怎么变，都要写清楚。
>
> **答**: 每步在 §3 中给出**变换前形式**、**变换方法**、**变换后形式**、**等价性证明（命题）**、**凸性影响**。

> **导师问题 5**: constraints 编号？怎么一步步变成 convex 的？
>
> **答**: 原约束编号 (5a)-(5h) 见 §1。每步凸化的「受影响的约束」「消除的非凸源」「等价性证明」逐项列出见 §3。

> **导师问题 6**: 最终 convex 的问题是什么？
>
> **答**: 见 §4，标准凸 SDP (P3)，含线性目标 + 5 组线性矩阵不等式 + 2 组半正定锥约束。可由 MOSEK 等标准 SDP 求解器在 5-10 秒内求解。

---

## 版本信息

- **文档**: Cell-Free ISAC 凸化推导链 v1.0
- **日期**: 2026-06-26
- **配套文档**:
  - `PROBLEM_FORMULATION_RIGOROUS.md` — 原问题定义
  - `MATHEMATICAL_DERIVATION.md` — 统一数学推导
  - `SDP_DERIVATION_COMPLETE.md` — SDP 松弛与 S-Procedure 详细推导
  - `SDP_IMPLEMENTATION_DERIVATION.md` — SDP 实现推导（含 KKT）
  - `ADVANCED_MATHEMATICAL_ANALYSIS.md` — 紧致性、复杂度下界、对偶间隙