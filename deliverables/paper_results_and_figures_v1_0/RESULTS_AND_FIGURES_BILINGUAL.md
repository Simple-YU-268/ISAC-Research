# Participation-Constrained Cell-Free ISAC: Results and Figures
# 参与约束无蜂窝 ISAC：结果与图像说明（中英文对照）

## 1. Purpose and authoritative scope / 目的与权威范围

This document is the bilingual guide to the evidence contained in this folder. It explains what each final figure measures, which result file supports it, the numerical conclusion that may be reported, and the limitations that must be retained in the paper, poster, and thesis.

本文是本文件夹中最终证据的中英文对照指南。它说明每张最终图测量什么、对应的数据文件、可在论文、海报和学位论文中报告的数值结论，以及必须保留的解释边界。

**Authoritative package rule.** This package is the final evidence source for the current paper. Do not mix its numerical claims with earlier exploratory or zero-sensing-floor runs elsewhere in the repository.

**权威包规则。** 本包是当前论文的最终证据来源。不得将其中数值结论与仓库内较早的探索性实验或零感知功率下限实验混用。

## 2. Model, configuration, and reporting rules / 模型、配置与报告规则

| Item / 项目 | Final setting / 最终设定 |
|---|---|
| Network / 网络 | Default: \(M=6\) APs, \(N_t=2\) transmit antennas per AP, \(K=3\) UEs, \(P=2\) targets, and \(N_\theta=2\) target-state dimensions. / 默认配置：\(M=6\) 个 AP、每 AP \(N_t=2\) 根发射天线、\(K=3\) 个 UE、\(P=2\) 个目标、\(N_\theta=2\) 维目标状态。 |
| Area and power / 区域与功率 | \(400\,\mathrm{m}\times400\,\mathrm{m}\), \(P_{\max}=0.1\) W (20 dBm). / \(400\,\mathrm{m}\times400\,\mathrm{m}\)，\(P_{\max}=0.1\) W（20 dBm）。 |
| Robustness / 鲁棒性 | CSI uncertainty radius \(\epsilon_h=0.05\) unless varied. / 除非专门扫描，否则 CSI 不确定半径为 \(\epsilon_h=0.05\)。 |
| QoS / 服务质量 | Communication and sensing SINR targets are 0 dB by default; PCRB allowance is automatically calibrated with \(\Gamma_{\rm alpha}=3\). / 默认通信和感知 SINR 门限均为 0 dB；PCRB 容许值由 \(\Gamma_{\rm alpha}=3\) 自动标定。 |
| Participation / 参与约束 | \(b_{mp}=1\) authorizes AP \(m\) to radiate nonzero dedicated sensing power for target \(p\). It does **not** gate globally cooperative communication. / \(b_{mp}=1\) 授权 AP \(m\) 为目标 \(p\) 发射非零专用感知功率；它**不**限制全局协同通信。 |
| Physical solution / 物理解 | A result is feasible only after binary-topology recovery, fixed-\(b\) continuous re-optimization, and explicit validation. / 只有经过二元拓扑恢复、固定 \(b\) 的连续重优化和显式物理验证后，结果才被计为可行。 |

### Reporting conventions / 报告口径

- **Feasibility is unconditional.** The denominator is all tested scenarios for that method and parameter point.  
  **可行率为无条件统计。** 分母是该方法和参数点的全部测试场景。
- **Power, sum rate, PCRB, and sensing SINR are conditional on feasible physical solutions**, unless a figure explicitly states a common-feasible set.  
  **功率、和速率、PCRB 和感知 SINR 仅在物理可行样本条件下统计**，除非图中特别说明采用共同可行样本集。
- **The SDR is a lower bound, not a physical baseline.** It must not be presented as a deployable solution.  
  **SDR 是下界，不是物理基线。** 不得将其描述为可部署方案。
- **Solver failure or time limit is not physical infeasibility.**  
  **求解器失败或超时不等于物理不可行。**

## 3. Figure index / 图像索引

| Figure / 图 | Main question / 核心问题 | Evidence file / 数据文件 |
|---|---|---|
| Fig. 1 | What does asymmetric sensing participation mean? / 非对称感知参与意味着什么？ | Deterministic schematic / 确定性示意图 |
| Fig. 2 | Why are two DC penalties needed? / 为什么需要双 DC 惩罚？ | `results/dual_dc_ablation_25seeds.mat` |
| Fig. 3 | How does \(N_{\rm req}\) affect feasibility, power, and runtime? / \(N_{\rm req}\) 如何影响可行率、功耗和时间？ | `results/nreq_qos_final.mat` |
| Fig. 4 | Are the QoS constraints actually met tightly? / QoS 约束是否被紧致满足？ | `results/nreq_qos_final.mat` |
| Fig. 5 | Does the DC relaxation statistically stabilize before recovery? / DC 松弛在恢复前是否统计稳定？ | `results/statistical_double_dc_convergence_final.mat` |
| Fig. S1 | What is the empirical distribution of the physical-recovery power gap to SDR? / 物理恢复相对 SDR 的功耗差距如何分布？ | `results/main_config_mc_100seeds_pilot_final.mat` |
| Fig. S2 | Which recovery components are needed for physical topology certification? / 物理拓扑认证需要哪些恢复环节？ | `results/recovery_ablation_30seeds_final.mat` |
| Fig. 7 | What is the cost of increasing transmit dimension? / 发射维度增大带来什么计算代价？ | `results/network_scaling_final.mat` |
| Fig. 8 | How sensitive is the required power to communication and tracking QoS targets? / 所需功耗对通信与追踪 QoS 门限有多敏感？ | `results/tradeoff_mc_final.mat` |
| Fig. 9 | Does the robust S-procedure improve outage protection? / 鲁棒 S-Procedure 是否改善中断保护？ | `results/csi_robustness_final.mat` |
| Fig. 10 | Does proposed recovery outperform association baselines? / 所提恢复是否优于关联基线？ | `results/main_m6_nreq_method_performance_30seeds.mat` |
| Fig. 11 | Does the result persist across physical settings? / 结果在不同物理设定下是否保持？ | `results/extended_physical_mc/` |
| Fig. 12 | What happens in deliberately difficult geometries? / 特意构造的困难几何下会怎样？ | `results/extended_physical_mc/stress_*.mat` |
| Fig. 11 (M12) | Can the method work at larger dimension? / 算法能否扩展至更高维？ | `results/m12_nreq3_8seeds.mat` |
| Fig. 12 (M12) | Does the \(N_{\rm req}\) effect remain at M12? / \(N_{\rm req}\) 效应在 M12 下是否仍成立？ | `results/m12_nreq2_4_5_5seeds.mat` |

> The `fig11` and `fig12` filename prefixes are historical. TeX assigns final figure numbers by document order, so these are not duplicate paper figures.  
> `fig11` 和 `fig12` 的文件名前缀是历史遗留；TeX 按文稿顺序决定最终编号，因此它们并不是论文中的重复图。

## 4. Detailed figure-by-figure interpretation / 逐图详细解读

### Fig. 1 — Architecture and participation semantics / 架构与参与变量语义

![Fig. 1 system architecture](figures/fig1_system_architecture.png)

**English.** The figure distinguishes globally cooperative communication covariances \(\{\mathbf W_k\}\) from target-specific dedicated sensing covariances \(\{\mathbf S_p\}\). The binary variable \(b_{mp}\) controls only whether AP \(m\) may allocate nonzero dedicated sensing power to target \(p\). Consequently, a selected AP–target pair is an actual sensing participant, whereas an AP may still transmit communication data even if it is not selected for that target.

**中文。** 该图区分了全局协同通信协方差 \(\{\mathbf W_k\}\) 与目标专属的感知协方差 \(\{\mathbf S_p\}\)。二元变量 \(b_{mp}\) 只控制 AP \(m\) 是否可向目标 \(p\) 分配非零专用感知功率。因此，被选择的 AP–目标对是真正的感知参与者；即使 AP 未被选作某个目标的感知节点，它仍可发射通信数据。

**Use in the paper / 论文用途.** This is the physical interpretation supporting the participation-constrained model; it is not a performance result.  
**论文用途。** 这是参与约束模型的物理解释图，不是性能结果图。

### Fig. 2 — Dual-DC penalty ablation / 双 DC 惩罚消融

![Fig. 2 dual DC ablation](figures/fig2_dual_dc_ablation.png)

**Experiment / 实验。** Twenty-five common scenarios compare SDR relaxation, rank-only, binary-only, and dual-DC variants, followed by the same fixed-topology physical validation procedure.

**Result / 结果。** The median binary residual falls from \(4.274\times10^{-1}\) without the binary penalty to \(5.940\times10^{-5}\) when binary DC is active. Rank residuals are below \(10^{-8}\) for all modes.

**Interpretation / 解读。** In this MU–MISO setting, SDR already produces nearly rank-one communication covariances. The empirical role of the dual mechanism is therefore primarily the recovery of the binary association, not an additional observed rank improvement. This supports retaining both terms in the general formulation while avoiding an overstated rank-penalty claim.  
**解读。** 在当前 MU–MISO 配置中，SDR 已产生近似秩一的通信协方差。因此，双 DC 的主要实证作用是恢复二元关联，而非额外显著改善秩残差。这支持在一般模型中保留两类惩罚，但不应夸大秩惩罚在本配置下的经验收益。

### Fig. 3 — Cluster-size trade-off / 感知集群规模权衡

![Fig. 3 cluster-size trade-off](figures/fig3_cluster_size_tradeoff.png)

**Experiment / 实验。** Thirty common seeds are tested for each \(N_{\rm req}\in\{2,3,4,5,6\}\). The figure reports physical feasibility, total transmit power, and end-to-end runtime; intervals are the 10th–90th percentiles.

**Result / 结果。** All tested points are physically feasible. Mean power is 37.65, 30.83, 29.50, 29.74, and 30.94 mW for \(N_{\rm req}=2,3,4,5,6\), respectively. The observed energy sweet spot is \(N_{\rm req}=4\).

**Interpretation / 解读。** Increasing \(N_{\rm req}\) initially provides FIM geometric diversity and lowers tracking power; eventually, the mandatory dedicated sensing-power floor and additional waveform resources offset that benefit. The relation is therefore non-monotone.  
**解读。** 增大 \(N_{\rm req}\) 最初会带来 FIM 几何分集并降低追踪功耗；但随后强制专用感知功率下限与额外波形资源消耗抵消收益，因此关系是非单调的。

### Fig. 4 — QoS tightness versus cluster size / 不同集群规模下的 QoS 紧致性

![Fig. 4 QoS versus cluster size](figures/fig4_qos_vs_cluster_size.png)

**English.** The maximum target PCRB ratio is essentially one at displayed precision, while the worst-user nominal communication and sensing-SINR margins are nonnegative. Recomputing PCRB from the returned covariance matrices gives the same conclusion.

**中文。** 最大目标 PCRB 比值在显示精度下基本为 1，最差用户的标称通信与感知 SINR 裕量均非负。由返回协方差重新计算 PCRB 后得到相同结论。

**Claim supported / 支持的结论。** The power reduction in Fig. 3 does not come from silently loosening tracking or SINR requirements.  
**支持的结论。** Fig. 3 中的功耗下降并非通过暗中放松追踪或 SINR 约束获得。

### Fig. 5 — Statistical double-DC stabilization / 双 DC 统计稳定过程

![Fig. 5 statistical double-DC convergence](figures/fig5_statistical_double_dc_convergence.png)

**Paper-ready bilingual caption / 论文可用中英文图注。**

> **Fig. 5. Statistical stabilization of the continuous double-DC SCA phase over 25 common realizations.** Iteration 0 denotes the unpenalized SDR relaxation. The binary-distance median and interquartile range decrease rapidly, whereas the Top-\(N_{\rm req}\) sensing-cluster support remains unchanged from the first DC iteration. The continuous-power trajectory is reported to characterize the effect of penalty continuation and is not interpreted as a monotone descent sequence.  
> **图 5. 基于 25 个共同随机场景的连续双 DC--SCA 阶段统计稳定过程。** 第 0 次迭代表示未施加惩罚的 SDR 松弛解。二元距离的中位数及四分位区间迅速降低，而 Top-\(N_{\rm req}\) 感知集群支持集从第一次 DC 迭代起保持不变。连续功率轨迹用于刻画惩罚延续的影响，不应解释为单调下降序列。

**Experimental setting / 实验设置。** For 25 common channel-and-geometry realizations, the continuous double-DC SCA subproblem is executed for six iterations after the unpenalized SDR initialization. Candidate construction, fixed-topology re-optimization, and physical-feasibility validation are subsequently performed as separate recovery steps; hence, this figure characterizes the stabilization of the continuous relaxation before discrete recovery.  
**实验设置。** 针对 25 个共同的信道与几何实现，在未施加惩罚的 SDR 初始化后执行六次连续双 DC--SCA 迭代。候选拓扑构造、固定拓扑重优化和物理可行性验证作为后续独立恢复步骤执行。因此，该图刻画的是离散恢复之前连续松弛问题的稳定过程。

**Results and discussion / 结果与讨论。** The median maximum binary distance decreases from \(4.336\times10^{-1}\) at the SDR point to \(1.139\times10^{-3}\) after one DC iteration and \(6.014\times10^{-5}\) after two iterations. For every realization, the Top-\(N_{\rm req}\) support is unchanged from the first DC iteration onward; the median rank residual is below \(5\times10^{-9}\) throughout. These results indicate that, under the considered configuration, the continuous relaxation identifies a stable sensing-cluster support within one to two DC iterations. The median continuous transmit power approaches 29.49 mW after the second iteration. Its small increase relative to the SDR point reflects the additional cost of enforcing near-binary association, rather than a loss of numerical stability.  
**结果与讨论。** 最大二元距离中位数由 SDR 点的 \(4.336\times10^{-1}\) 降至一次 DC 迭代后的 \(1.139\times10^{-3}\)，并在两次迭代后达到 \(6.014\times10^{-5}\)。对全部场景而言，Top-\(N_{\rm req}\) 支持集从第一次 DC 迭代起不再变化；秩残差中位数在全过程中低于 \(5\times10^{-9}\)。这表明，在所考虑的配置下，连续松弛可在一至两次 DC 迭代内识别稳定的感知集群支持集。连续发射功率中位数在第二次迭代后接近 29.49 mW；相对于 SDR 点的轻微增加反映了施加近似二元关联所需的额外代价，而非数值稳定性的损失。

**Reporting boundary / 报告边界。** The figure supports early termination of the continuous DC iterations after support stabilization, followed by the prescribed fixed-\(b\) recovery and validation. It does not, by itself, establish physical feasibility or global optimality of the original mixed-integer problem.  
**报告边界。** 该图支持在支持集稳定后提前终止连续 DC 迭代，并执行既定的固定 \(b\) 恢复与验证步骤。该图本身并不单独证明原始混合整数问题的物理可行性或全局最优性。

### Fig. S1 — Empirical power gap to the SDR lower bound / 相对 SDR 下界的经验功耗差距

![Fig. S1 empirical CDF of power gap to SDR](figures/figS1_power_gap_cdf.png)

**Experimental setting / 实验设置。** This supplementary plot is generated from the archived 100-realization M6 pilot output at \(N_{\rm req}=3\). For each physically recovered solution, the displayed quantity is \(100(P_{\rm physical}-P_{\rm SDR})/P_{\rm SDR}\), where the SDR solution is used only as a relaxation lower bound.  
**实验设置。** 该补充图基于归档的 M6、\(N_{\rm req}=3\) 的 100 个实现预实验输出。对于每个完成物理恢复的解，横轴表示 \(100(P_{\rm physical}-P_{\rm SDR})/P_{\rm SDR}\)；其中 SDR 解仅作为松弛问题的功耗下界。

**Results and discussion / 结果与讨论。** The empirical distribution is concentrated at moderate positive gaps, confirming the expected cost of enforcing a binary topology and physical validation relative to the continuous relaxation. This comparison should be interpreted as a lower-bound gap, rather than a comparison against a deployable SDR baseline.  
**结果与讨论。** 经验分布集中于中等的正功耗差距，表明相对于连续松弛，施加二元拓扑和物理验证会带来预期的额外代价。该比较应解读为相对下界的差距，而不应将 SDR 视为可部署的基线方案。

### Fig. S2 — Recovery ablation / 恢复机制消融

![Fig. S2 recovery ablation](figures/figS2_recovery_ablation.png)

**Experimental setting / 实验设置。** Thirty common M6 realizations at \(N_{\rm req}=3\) compare FIM-only topology construction, direct DC Top-\(N_{\rm req}\) rounding, and the full recovery procedure. The three panels report physical-feasibility rate, conditional median power penalty relative to SDR, and runtime; the error bars in the runtime panel extend to the 90th percentile.  
**实验设置。** 在 \(N_{\rm req}=3\) 下，30 个共同 M6 实现比较 FIM-only 拓扑构造、直接 DC Top-\(N_{\rm req}\) 取整和完整恢复流程。三个子图依次报告物理可行率、相对 SDR 的条件中位功耗差距和运行时间；运行时间子图中的误差条延伸至第 90 百分位数。

**Results and discussion / 结果与讨论。** FIM-only construction and full recovery are feasible in all 30 realizations, whereas direct DC Top-\(N_{\rm req}\) rounding is feasible in only 4 of 30 realizations. Direct rounding also exhibits a substantially larger conditional power penalty. The result demonstrates that a near-binary continuous relaxation alone is insufficient for physical certification: topology-aware candidate construction and fixed-topology continuous re-optimization are required.  
**结果与讨论。** FIM-only 构造和完整恢复在全部 30 个实现中均可行，而直接 DC Top-\(N_{\rm req}\) 取整仅在 30 个实现中的 4 个可行。直接取整还表现出显著更大的条件功耗差距。该结果表明，近二元连续松弛本身不足以完成物理认证；仍需结合拓扑感知的候选构造和固定拓扑连续重优化。

### Fig. 7 — Dimension sensitivity / 维度敏感性

![Fig. 7 dimension sensitivity](figures/fig7_dimension_sensitivity.png)

**Experiment / 实验。** The total transmit dimension is increased through \(M\in\{4,6,8\}\) with \(N_t=2\), corresponding to 8, 12, and 16 transmit dimensions, over ten seeds per point.

**Result / 结果。** All ten tested scenarios remain physically feasible at each point. Runtime increases with dimension, as expected for a robust SDP with per-target sensing covariances and DC recovery.

**Interpretation / 解读。** This is a computational-scaling result, not a real-time claim. Larger M12 validation is reported separately below.  
**解读。** 该图是计算规模结果，而非实时性能声明。更大规模的 M12 验证在后文单独报告。

### Fig. 8 — QoS-requirement sensitivity / QoS 门限敏感性

![Fig. 8 statistical trade-off](figures/fig8_statistical_tradeoff.png)

**Experimental setting / 实验设置。** The proposed method is evaluated on a \(5\times5\) grid of robust communication-SINR targets and PCRB allowance scales \(\alpha\), using five common realizations at every grid point. A smaller \(\alpha\) imposes a stricter tracking requirement. The left panel reports the conditional mean total transmit power, while the right panel reports the unconditional physical-feasibility rate.

**实验设置。** 在鲁棒通信 SINR 门限和 PCRB 容许尺度 \(\alpha\) 构成的 \(5\times5\) 网格上评估所提算法；每个网格点采用五个共同实现。较小的 \(\alpha\) 对应更严格的追踪要求。左图给出条件平均总发射功率，右图给出无条件物理可行率。

**Results and discussion / 结果与讨论。** The required transmit power increases when the communication-SINR target is raised and, for a fixed communication target, when the PCRB allowance is tightened by decreasing \(\alpha\). Thus, the figure quantifies the resource cost of increasingly stringent service requirements under the proposed architecture. All 25 examined operating points are physically feasible over the tested range. This result characterizes local QoS sensitivity; it neither traces a communication--sensing Pareto boundary nor determines the complete feasible region.

**结果与讨论。** 当通信 SINR 门限升高时，所需发射功率增加；在固定通信门限下，减小 \(\alpha\) 以收紧 PCRB 容许值同样会提高所需功耗。因此，该图量化了在所提架构下满足更严格服务要求的资源代价。全部 25 个测试工作点在所考察范围内均具有物理可行性。该结果刻画的是局部 QoS 敏感性，既不构成通信–感知帕累托边界，也不确定完整可行域。

### Fig. 9 — Robust CSI experiment / 鲁棒 CSI 实验

![Fig. 9 CSI robustness](figures/fig9_csi_robustness.png)

**Experiment / 实验。** The proposed robust design and a nominal-CSI baseline (designed with \(\epsilon_h=0\)) are tested under identical random complex perturbations in the prescribed uncertainty ball: 100 perturbations per seed, 30 seeds, and \(\epsilon_h\in\{0.02,0.05,0.08\}\).

**Result / 结果。** The robust design has zero sampled system outage. The nominal design has mean outage 89.0%, 89.6%, and 91.9% at the three uncertainty radii, respectively. At \(\epsilon_h=0.08\), the median robust power premium is only 5.12%.

**Interpretation / 解读。** The sampled experiment complements the worst-case S-procedure certificate; it does not replace that certificate.  
**解读。** 采样实验补充了最坏情况 S-Procedure 证书，但不能替代该证书。

### Fig. 10 — Association and recovery comparison / 关联与恢复方法对比

![Fig. 10 method comparison](figures/fig10_method_comparison_vs_nreq.png)

**Experiment / 实验。** Proposed binary-DC recovery, FIM-greedy, nearest-AP, and random cardinality-feasible association are compared on 30 common scenarios for each \(N_{\rm req}\). All physical methods use the same fixed-assignment robust beamforming and dedicated sensing-covariance re-optimization after topology selection. SDR is shown only as a power lower bound.

**Key results / 关键结果。**

- Proposed recovery is physically feasible in 30/30 trials at every tested \(N_{\rm req}\). / 所提恢复在每个测试 \(N_{\rm req}\) 下均为 30/30 可行。
- FIM-greedy is also feasible in 30/30 trials, but with slightly higher power. / FIM-greedy 同样为 30/30 可行，但平均功耗略高。
- At \(N_{\rm req}=3\), mean power is 30.83 mW (proposed), 30.93 mW (FIM-greedy), and 39.29 mW (nearest AP). / 当 \(N_{\rm req}=3\) 时，平均功耗分别为 30.83、30.93 和 39.29 mW。
- At \(N_{\rm req}=2\), nearest-AP is feasible in only 24/30 trials and random in 1/30 trials. / 当 \(N_{\rm req}=2\) 时，最近 AP 仅 24/30 可行，随机方法仅 1/30 可行。
- At \(N_{\rm req}=3\), random is feasible in only 10/30 trials and has conditional mean power 115.82 mW. / 当 \(N_{\rm req}=3\) 时，随机方法仅 10/30 可行，其条件平均功耗为 115.82 mW。

**Interpretation / 解读。** FIM geometry is a strong predictor of a good discrete sensing topology, explaining why FIM-greedy closely approaches the proposed method. The proposed algorithm remains necessary to optimize robust continuous covariances, meet all coupled constraints, and certify the recovered binary solution.  
**解读。** FIM 几何是优质离散感知拓扑的强预测因素，因此 FIM-greedy 接近所提方法；但所提算法仍是优化鲁棒连续协方差、满足耦合约束并认证恢复后二元解所必需的。

### Fig. 11 — Extended physical-factor study / 扩展物理因素实验

![Fig. 11 extended physical factors](figures/fig11_extended_physical_factors.png)

**Experiment / 实验。** One physical factor is varied at a time from \((M,N_t,K,P,N_{\rm req})=(6,2,3,2,3)\): AP count, AP antennas, UE load, target load, deployment side length, and AP power budget. There are 22 configurations and 30 common seeds per point, totalling 660 scenarios. Proposed, FIM-greedy, and nearest-AP methods use identical realizations and identical post-assignment continuous re-optimization.

**Results / 结果。** The \(N_t=1\) reference configuration is rejected before optimization because automatic reference-FIM calibration identifies it as unobservable; it is not counted as physical infeasibility. Of the remaining 630 observable scenarios, proposed and FIM-greedy are feasible in all cases, whereas nearest-AP is feasible in 600/630 cases. Across the AP-count sweep, proposed recovery reduces paired conditional power by 12.4%–18.2% relative to nearest AP; the paired reductions are 21.3% at \(K=4\) and 19.3% at \(P=3\).

**Interpretation / 解读。** The advantage is not limited to one nominal layout. Nearest distance does not encode the angular-information diversity required by PCRB, while FIM-aware association does.  
**解读。** 该优势并不局限于单一标称布局。最近距离不能编码 PCRB 所需的角度信息分集，而 FIM 感知的关联可以。

### Fig. 12 — Pressure geometries / 压力几何场景

![Fig. 12 pressure geometries](figures/fig12_pressure_geometries.png)

**Experiment / 实验。** Two controlled stress settings are used: edge UE–target co-location and crowded targets. Each has 30 common seeds.

**Results / 结果。** Proposed recovery and FIM-greedy are feasible in all 30 trials for both geometries. In the edge co-location case, nearest AP is feasible in 29/30 trials and needs 49.6% more paired power. With crowded targets, nearest AP is feasible in only 7/30 trials and needs 81.4% more power on the common feasible trials.

**Interpretation / 解读。** These are the clearest demonstrations that distance-only association can be geometrically wrong for position PCRB.  
**解读。** 这是最直观的证据：仅按距离关联在位置 PCRB 问题中可能在几何上就是错误的。

### Fig. 11 (M12) — Fixed-cardinality large-scale validation / 固定基数 M12 大规模验证

![Fig. 11 M12 scalability validation](figures/fig11_m12_scalability_validation.png)

**Experiment / 实验。** \(M=12\), \(N_t=2\), \(K=6\), \(P=3\), \(N_{\rm req}=3\), eight common realizations, four isolated MATLAB workers, and a 60-second MOSEK limit per convex subproblem.

**Results / 结果。** Proposed recovery is feasible in 8/8 cases; FIM-greedy and nearest AP are feasible in 6/8; random is feasible in 1/8. On six FIM-common-feasible realizations, proposed uses 37.10 mW versus 37.88 mW; on six nearest-common-feasible realizations, it uses 38.72 mW versus 94.89 mW. The proposed mean normalized PCRB is 0.999987. Mean end-to-end runtime is 496.3 seconds per scenario.

**Interpretation / 解读。** This validates the method beyond the small system, while explicitly revealing its high-dimensional computational cost. The runtime is a cost measurement, not a real-time claim.  
**解读。** 该实验验证方法可扩展到更大系统，同时明确揭示高维计算代价。该时间是成本度量，而非实时性能声明。

### Fig. 12 (M12) — Large-scale cardinality sweep / M12 大规模基数扫描

![Fig. 12 M12 Nreq scalability](figures/fig12_m12_nreq_scalability.png)

**Experiment / 实验。** The same M12, \(K=6\), \(P=3\) system is evaluated for \(N_{\rm req}\in\{2,3,4,5\}\) on five common realizations.

**Results / 结果。** Proposed recovery is feasible in 4/5 cases at \(N_{\rm req}=2\) and 5/5 for \(N_{\rm req}\geq3\). Its conditional mean power drops from 45.07 mW at \(N_{\rm req}=2\) to 35.72 mW at \(N_{\rm req}=4\), and is 35.76 mW at \(N_{\rm req}=5\). FIM-greedy reaches full feasibility only from \(N_{\rm req}=4\); nearest AP is feasible in only 2/5 cases at \(N_{\rm req}=2\) and needs 104.40 mW at \(N_{\rm req}=3\). PCRB ratios remain at their tracking boundaries.

**Interpretation / 解读。** The intermediate cluster-size optimum persists at a larger dimension. It is a system-level operating-point effect, not an artefact of the six-AP configuration.  
**解读。** 中间集群规模最优点在更高维度下仍存在，因此它是系统层面的工作点效应，而非六 AP 配置的偶然现象。

## 5. Consolidated findings / 汇总结论

1. **Physical semantics are enforced.** A selected \((m,p)\) pair transmits nonzero dedicated sensing power, while communication remains globally cooperative.  
   **物理语义得到落实。** 被选择的 \((m,p)\) 对发射非零专用感知功率，而通信保持全局协同。
2. **Binary recovery matters.** The binary DC penalty reduces the association residual by roughly four orders of magnitude and makes physically certifiable topology recovery possible.  
   **二元恢复至关重要。** 二元 DC 惩罚将关联残差降低约四个数量级，使可物理认证的拓扑恢复成为可能。
3. **Sensing-cluster size has an energy optimum.** In the principal M6 study, \(N_{\rm req}=4\) is the observed lowest-power operating point; the same pattern reappears in the M12 sweep.  
   **感知集群规模存在能耗最优点。** 在主 M6 实验中，\(N_{\rm req}=4\) 是观察到的最低功耗点；M12 扫描中同样出现该规律。
4. **Geometry is more important than distance for sensing association.** FIM-aware topology is consistently robust, while nearest-AP and random rules lose feasibility or require far more power in pressure geometries.  
   **几何信息比距离对感知关联更重要。** FIM 感知拓扑始终稳健；最近 AP 和随机规则在压力几何中会损失可行性或消耗显著更多功率。
5. **Robust CSI design is empirically necessary.** The S-procedure design eliminates sampled outage in the tested uncertainty range, whereas nominal design fails frequently.  
   **鲁棒 CSI 设计在经验上是必要的。** S-Procedure 设计在测试不确定范围内消除了采样中断，而标称设计频繁失效。
6. **The method scales, but not as a real-time solver.** M12 validation confirms physical feasibility and performance separation, while the observed hundreds-of-seconds runtime makes distributed or accelerated future work necessary.  
   **算法可扩展，但尚非实时求解器。** M12 验证确认了物理可行性和性能差异，但数百秒级运行时间说明仍需分布式或加速算法。

## 6. Safe wording for manuscript and presentation / 论文与答辩的安全表述

**Use / 可使用的表述**

- “The proposed recovery constructs a physically certified binary sensing topology and robust continuous covariances.”  
  “所提恢复构造了经物理认证的二元感知拓扑与鲁棒连续协方差。”
- “FIM-aware association is a strong topology baseline; the proposed recovery provides the final coupled robust covariance optimization and certification.”  
  “FIM 感知关联是强拓扑基线；所提恢复提供最终的耦合鲁棒协方差优化与认证。”
- “The observed minimum-power cluster size is intermediate under the tested configuration.”  
  “在所测试配置下，观察到的最低功耗集群规模处于中间值。”

**Do not use / 不应使用的表述**

- “Fig. 8 is the global Pareto boundary.” / “Fig. 8 是全局帕累托边界。”
- “The algorithm guarantees a globally optimal MINLP solution.” / “算法保证得到全局最优 MINLP 解。”
- “The M12 runtime demonstrates real-time feasibility.” / “M12 运行时间证明算法可实时运行。”
- “Solver time limit proves physical infeasibility.” / “求解器时间限制证明物理不可行。”

## 7. Reproduction pointers / 复现索引

- Final manuscript material: `paper/math_derivation_en.pdf` and `paper/numerical_results.tex`.  
  最终文稿材料：`paper/math_derivation_en.pdf` 与 `paper/numerical_results.tex`。
- MATLAB solvers, validators, experiments, plots, and audits: `scripts/`.  
  MATLAB 求解器、验证器、实验、绘图与审计脚本：`scripts/`。
- Final numerical data: `results/`; factor-level extended data: `results/extended_physical_mc/`.  
  最终数值数据：`results/`；分因素扩展数据：`results/extended_physical_mc/`。

## 8. Metadata note before submission / 投稿前元数据注意事项

The numerical-results TeX source records MATLAB R2024b, CVX, and MOSEK 11.2.2 for the main configuration, whereas the package README records MATLAB R2026a. Resolve this version-metadata discrepancy before final submission, and then freeze the software versions, machine specification, random seeds, and Git commit.

数值结果 TeX 源文件对主配置记录的是 MATLAB R2024b、CVX 和 MOSEK 11.2.2，而本包 README 记录 MATLAB R2026a。最终投稿前应解决这一版本元数据不一致，并冻结软件版本、机器配置、随机种子和 Git 提交号。
