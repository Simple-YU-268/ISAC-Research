# 第 1--4 章与实验包一致性审查

**审查日期：** 2026-08-03  
**范围：** `introduction.tex`、`literaturereview.tex`、`systemmodel.tex`、`solutionmethod.tex`、MATLAB 求解/恢复/验证代码和现有结果包。  
**统一语义：** `b(m,p)=1` 表示 AP `m` 获授权为目标 `p` 分配专用感知功率；它可以分配零功率。`b(m,p)=0` 强制该 AP--target 对的专用感知功率为零。此变量不门控通信波束。

\[
0\leq \operatorname{tr}(\mathbf E_m\mathbf S_p)
\leq P_{\max}b_{mp},\qquad \forall m,p.
\]

“Authorized APs” 是二元授权集合的计数，不能解释为实际拥有非零专用感知功率的 AP 数。

## 审查结论

当前求解器、固定拓扑重求解、验证器和论文第 1--4 章均采用上述单边授权门控；不需要加入 `P_min^sen` 下界，也不需要因关联语义而重跑现有实验。唯一需要修正的是实验输出与任何图表/文字中的 **Activated APs** 术语：它应统一为 **Authorized APs**，以避免把二元标签误读为实际发射。

## 代码与结果证据

| 证据类型 | 位置 | 结论 |
|---|---|---|
| 求解模型 | `sim/matlab/solve_p3_sca_t.m:117-122` | 仅实施 `tr(E_m S_p) <= Pmax*b(m,p)`；没有最小专用感知功率约束。 |
| 固定拓扑诊断 | `sim/matlab/check_fixed_b_pcrb.m:35-37` | 固定 `b` 后仍只使用同一单边门控。 |
| 物理验证 | `sim/matlab/validate_solution.m:128-134` | 只验证未授权 AP 的感知功率为零及上界，不要求获授权 AP 发射。 |
| 总硬件约束 | `sim/matlab/solve_p3_sca_t.m:111-115` | 通信和所有感知协方差共同受 `tr(E_m R_X) <= Pmax` 约束。 |
| 恢复流程 | `sim/matlab/baseline_alg2.m:73-99` | 每目标 Top-`N_req` 授权候选经固定 `b` 重求解和物理验证后再比较。 |
| 实验统计 | `sim/matlab/experiments_paper.m:232-234` | `b>0.5` 计数的是授权 AP；已改名为 `authorized_aps`。 |

现有 `.mat`、`.csv` 和图表结果仍可作为此单边授权模型的结果记录。它们可支持可行率、条件功率、PCRB、鲁棒性和授权拓扑比较；但不能支持“每个授权 AP 都实际发射感知波束”的主张。

## 分章一致性审查

| 章节 | 状态 | 结论与统一要求 |
|---|---|---|
| 第 1 章 Introduction | 一致 | `introduction.tex:42-48` 已将关联描述为 dedicated sensing-power authorization，且明确通信全局协作。不得把其改写为必须发射。 |
| 第 2 章 Literature Review | 一致，已澄清 | `literaturereview.tex:61-69` 已改为“authorize subsets of APs to allocate a sensing waveform”，避免将选中集合表述为必然非零发射集合。 |
| 第 3 章 System Model | 一致 | `systemmodel.tex:88-106` 的单边式 (3.7) 及 “selected AP is not forced to consume a minimum amount of power” 是规范定义；架构图中的橙色链路表示授权的专用感知资源，不保证非零功率。 |
| 第 4 章 Optimization Formulation and Solution | 一致 | P1/P2/P3 中的关联约束均应保持单边上界；`solutionmethod.tex:19-29` 的 “authorization rather than a minimum-power guarantee” 是规范解释。P1-C5' 的 `N_req` 是每目标获授权 AP 数。 |

## 术语与图表规范

| 推荐用语 | 禁止/避免用语 | 原因 |
|---|---|---|
| authorized AP / authorized sensing cluster | activated AP / transmitting AP | 二元 `b` 只给出授权，不保证正感知功率。 |
| authorized dedicated sensing power | mandatory sensing power | 单边 Big-M 没有最小功率。 |
| authorization cardinality `N_req` | number of actual sensing transmitters | `sum_m b_mp=N_req` 计数的是授权变量。 |
| actual nonzero sensing support (如需讨论) | 与 `b` 等同 | 仅可通过求解后的 `tr(E_m S_p)>tol` 另行计算，且不应代替模型的 `N_req`。 |

## 保留与禁止的论文主张

**可保留：**

- 通信波束全球协作，且不被 `b_mp` 直接门控。
- 专用感知协方差 `S_p` 产生 target-specific FIM/PCRB；通信协方差不获得 PCRB credit。
- 关联、总 AP 功率、鲁棒 SINR、sensing SINR、PCRB、基数及二元性均在固定候选后重求解并验证。
- 现有数值图表比较不同的授权拓扑及其可行性/条件性能。

**不得主张：**

- `b_mp=1` 必然意味着 AP `m` 对目标 `p` 发射非零专用感知波束。
- `N_req` 等于实际具有非零专用感知功率的 AP 数。
- “Authorized APs” 图表是实际发射 AP 数的证据。

## 后续维护要求

1. 所有新增论文文字、图注、表头与图例采用 **authorized**，不使用 **activated** 表示 `b`。
2. 新实验继续使用单边门控，除非研究问题明确改为实际参与约束；后者属于新模型，必须另建分支结果并完整重跑。
3. 若未来需要报告实际非零支持集，应另外以数值阈值定义和统计 `tr(E_mS_p)`，并与授权集合分开报告。
