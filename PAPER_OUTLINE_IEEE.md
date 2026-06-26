# ISAC 论文 IEEE 风格叙事大纲（4-plate 改良方案 B）

**目标期刊**: IEEE Trans. Signal Processing / IEEE JSAC
**核心结构**: 4-plate 物理层叙事 + IMRaD 标题
**参考**: arXiv 2506.18560 (Derrick Wing Kwan Ng & Shaokang Hu 联合作者)
**日期**: 2026-06-26

---

## 大纲映射

| IMRaD 章节 | 4-plate 对应 | 现有 tex 章节 | 改后 |
|---|---|---|---|
| §I Introduction | (新增 Plate 0) | — | 新增 |
| §II Physical Modeling & Problem Formulation | **Plate I** | §1 §2 §3 §4 | 改标题 |
| §III Mathematical Reformulation | **Plate II** | §5 §6 §7 | 改标题 |
| §IV Beam Recovery & Algorithm | **Plate III** | §8 | 改标题 |
| §V Benchmark: Closed-Form Baseline | **Plate IV** | §9 §10 | 改标题 |
| §VI Numerical Results | (新增) | — | **占位/或补** |
| §VII Conclusion | §11 | 结论与展望 | 改标题 |

---

## 各节内容要点

### §I Introduction（约 600-800 字）

**结构（标准 IEEE Introduction 三段式）**：

1. **§I-A Motivation**（~200 字）
   - ISAC 是 6G 关键范式：通信感知频谱共享
   - Cell-free 大规模 MIMO + ISAC：分布式 AP 联合处理，提供宏分集 + 干扰协调
   - 痛点：**非凸 MINLP 难解** + **不完美 CSI 下鲁棒性难保证**

2. **§I-B Related Work**（~300 字）
   - 单小区 ISAC beamforming: [Liu 2020, Masouros 2018]
   - Cell-free ISAC MIMO: [Derrick Wing Kwan Ng 2024 (arXiv 2506.18560), Shaokang Hu 2024] — 我们的最相关 baseline
   - 现有方法的不足：① DRL/启发式缺乏理论保证 ② 闭式 ZF 闭式解 25% 成功率 ③ 现有 SDP 缺完整 7 步凸化证明链

3. **§I-C Contributions**（~200 字，3-4 个 bullet）
   1. **完整 7 步凸化推导链**——每步给出非凸源识别、变换形式、严格等价/紧致/保守证明（vs. 现有论文只给最终 SDP）
   2. **闭式最终 (P3) 标准凸 SDP**——可直接送入 CVX/MOSEK，含 LMI、约束编号、紧致性定理
   3. **SDR 紧致性分析**——给出 $K\leq 2$ 紧致条件 + 高 SNR 近似紧致 + 恢复性能 $O(1/L)$ 上界
   4. **AP 选择启发式 + 复杂度下界**——证明固定 AP 子问题为凸 SDP，AP 选择 NP-hard

---

### §II Physical Modeling and Problem Formulation（Plate I）

- 2.1 Network & Channel Model（cell-free AP 网格 + 信道 + CSI 误差模型）
- 2.2 Communication & Sensing Signal Models
- 2.3 Constraint Derivation（worst-case SINR + PCRB + 功率）
- 2.4 Joint Non-Convex MINLP (P1) — 编号 (5a)-(5h)
- **Plate I 高潮**：抛出极度非凸的 (P1)，点明"这是本文要解的硬骨头"

---

### §III Mathematical Reformulation: A Seven-Step Convexification Chain（Plate II）

- 3.1 Non-Convexity Sources Identification（NC1-NC6 + 凸性判据）
- 3.2 Step-by-Step Convexification（Step 1-7 编号 + 每步命题 + 等价性证明）
- 3.3 Final Convex SDP (P3) — **Plate II 高潮**
- 3.4 Tightness Analysis（SDR 紧致性定理 + 紧致条件 + 复杂度分析）

---

### §IV Engineering Feasibility: Rank-One Recovery and Robust Fallback（Plate III）

- 4.1 Rank Analysis（多秩感知合理性——这是 ISAC 区别于纯通信的关键）
- 4.2 Algorithm 1: Gaussian Randomization Rank-One Recovery
- 4.3 Power-Scaling Mathematical Guarantee
- 4.4 Complexity Convergence
- **Plate III 高潮**：从 SDP 的凸解回到物理可实现 rank-1 波束

---

### §V Benchmark: Low-Complexity Closed-Form Solution（Plate IV）

- 5.1 Heuristic Assumptions（ZF 适用条件 + MF 适用条件）
- 5.2 ZF Communication Beamforming
- 5.3 MF Sensing Beamforming
- 5.4 Algorithm 2: Closed-Form Solver
- 5.5 Complexity vs. (P3)
- **Plate IV 高潮**：与 §III 的 (P3) 形成对照基准

---

### §VI Numerical Results（占位/或补）

- 6.1 Simulation Setup
  - $M=16, N_t=4, K=10, P=4, P_{\max}=30$W, $\epsilon_h=0.10, \epsilon_g=0.15$
  - 1000 trials, 5 seeds
- 6.2 Convergence & Feasibility Rate vs. SNR
- 6.3 Power Efficiency vs. Number of APs
- 6.4 Runtime Comparison: (P3) vs. Closed-Form
- 6.5 Comparison with DRL-based baseline (Derrick/Shaokang Hu 2024)
- **当前状态**：仿真实验**尚未运行**。§VI 可写"仿真设置 + 期望结果 + 后续工作"占位
- **或方案 A2**：跑 `isac_final_solver.m` 抓真实数据再写

---

### §VII Conclusion and Future Work

- 总结 4 大贡献
- Future Work：MOSEK 完整 SDP 求解器、多目标实验、RIS-ISAC 扩展、时隙耦合 MPC

---

## 改造方案

### Step 1：标题批量重命名（最小破坏）
将现有标题改为 IMRaD 风格，**不改正文**:

| 现有标题 | 改后标题 |
|---|---|
| §1 系统模型与问题定义 | §II Physical Modeling and Problem Formulation |
| §2 信号模型与CSI误差模型 | §II.1 Network and Channel Model |
| §3 约束条件推导 | §II.3 Constraint Derivation |
| §5 问题凸化与SDP重构 | §III Mathematical Reformulation |
| §6 凸化链（新增） | §III.2 Seven-Step Convexification Chain |
| §8 面向物理可实现性的波束恢复 | §IV Engineering Feasibility: Rank-One Recovery |
| §9 低复杂度闭式解基准算法 | §V Benchmark: Closed-Form Baseline |
| §10 复杂度对比分析 | §V.5 Complexity Comparison |
| §11 可行性条件 | §III.4 Tightness & Feasibility Analysis |
| §12 关键公式汇总 | §V.6 Key Formulas Summary |
| §13 结论与展望 | §VII Conclusion and Future Work |

### Step 2：新增 §I Introduction（仅 report.tex + report_en.tex）
- ~600-800 字 Introduction（动机 + Related Work + Contributions）
- 中英文版本

### Step 3：§VI Numerical Results（占位）
- 写仿真设置表 + 期望性能表
- **不编造数据**

### Step 4：编译验证 + commit

---

## 风险点

1. **§VI Numerical Results 是空**——方案 A1 先占位，A2 跑仿真后再补
2. **§I Introduction 需要 Related Work 引用**——必须引用 Derrick/Shaokang Hu 论文，但 Zotero 暂时连不上 → 我**用 arXiv 2506.18560 作 inline 引用**（@misc 形式），等你接好 Zotero 再补完整 bibtex
3. **章节顺序微调**：§11 可行性条件应放在 §III 末尾（紧致性分析），不放在末尾——这是 IMRaD 风格要求
4. **页数变化**：+2-3 页/文件（§I ~1 页 + §VI ~1-2 页 + 标题调整）

---

## 执行时间线

- Step 1 标题重命名：~5 min（sed 替换）
- Step 2 写 §I Introduction：~10 min
- Step 3 写 §VI 占位：~5 min
- Step 4 编译验证：~5 min × 4 文件
- **总：~30 min**
