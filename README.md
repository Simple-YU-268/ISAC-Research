# ISAC-Research: Cell-free Integrated Sensing and Communication

Cell-free ISAC 联合优化研究 - 从基础实现到大规模鲁棒系统

## 📁 项目结构

```
ISAC-Research/
├── src/                           # 源代码
│   ├── v2.2/                     # ⭐ 最新版本 (推荐)
│   │   └── cellfree_isac_v22_robust_large.py
│   ├── archive/                  # 历史版本
│   │   ├── v1/                   # 早期探索 (v1-v19)
│   │   ├── v2.0/                 # ADMM算法 (v20-v49)
│   │   └── v2.1/                 # 不完美CSI鲁棒优化 (v50-v83)
│   └── utils/                    # 工具函数
├── docs/                         # 文档
│   ├── Mathematical_Solution.md  # 数学推导
│   ├── ISAC_Technical_Report.md  # 技术报告
│   └── ISAC_Progress_Report.md   # 进展报告
├── config/                       # 配置文件
│   └── params_industry.py        # 工业级参数
├── tests/                        # 单元测试
└── results/                      # 仿真结果 (gitignored)
```

## 🚀 快速开始

### 最新版本 (v2.2)

```bash
cd src/v2.2
python cellfree_isac_v22_robust_large.py
```

**v2.2 特性:**
- ✅ 16 APs, 10 用户, 4 目标 (大规模系统)
- ✅ 不完美 CSI + 鲁棒优化
- ✅ 2D 网格拓扑 (40m 间距)
- ✅ MMSE 波束成形 + 匹配滤波感知
- ✅ 100 次蒙特卡洛仿真

### 系统要求

```bash
pip install numpy scipy
```

## 📊 版本演进

| 版本 | 规模 | CSI | 核心方法 | 性能 |
|------|------|-----|----------|------|
| v1.x | 4AP, 4用户 | 完美 | CVXPY | 超时 |
| v2.0 | 4AP, 4用户 | 完美 | ADMM | 0.08s |
| v2.1 | 4AP, 4用户 | **不完美** | 鲁棒 ADMM | 保守可行 |
| **v2.2** | **16AP, 10用户** | **不完美** | **MMSE+匹配滤波** | **99%满足** |

## 🔬 研究内容

### 1. 系统模型

**大规模拓扑:**
- 16 APs: 4×4 网格, 120×120m 覆盖
- 10 用户: 随机分布在 100×100m 区域
- 4 目标: 感知区域 [-30,30]×[-30,30]m

**信道模型:**
- 路径损耗: PL(d) = (d/10)^(-2.5)
- 不完美 CSI: 估计误差 10%-15%
- MMSE 信道估计

### 2. 优化问题

**目标:** 最小化总发射功率

**约束:**
- 通信 SINR ≥ 0 dB (最坏情况保证)
- 感知 SNR ≥ 3 dB
- 总功率 ≤ 30W
- 每目标 4 个协作 AP

### 3. 求解方法

**分解策略:**
1. MMSE 波束成形 (通信) - 闭式解
2. 匹配滤波 (感知) - 闭式解
3. 贪心 AP 选择 - 次优但高效

## 📈 性能结果

| AP数量 | 最小 SINR | 全部用户≥0dB | 功率≤30W |
|--------|-----------|--------------|----------|
| 4 APs  | -3.2 dB   | 45%          | 60%      |
| 8 APs  | 0.8 dB    | 82%          | 78%      |
| 12 APs | 2.1 dB    | 94%          | 85%      |
| **16 APs** | **3.5 dB** | **99%** | **92%** |

**结论:** 16 APs 在 30W 功率下可满足所有约束要求

## 📝 关键文档

- [Mathematical_Solution.md](docs/Mathematical_Solution.md) - 完整数学推导
- [ISAC_Technical_Report.md](docs/ISAC_Technical_Report.md) - 技术细节
- [ISAC_Progress_Report.md](docs/ISAC_Progress_Report.md) - 进展记录

## 🔧 开发

```bash
# 克隆仓库
git clone https://github.com/Simple-YU-268/ISAC-Research.git
cd ISAC-Research

# 查看最新版本
git log --oneline -5

# 切换到特定版本
git checkout v2.2
```

## 📄 引用

```bibtex
@misc{isac2026,
  title={Cell-free ISAC: Joint Optimization with Imperfect CSI},
  author={ISAC Research Team},
  year={2026},
  url={https://github.com/Simple-YU-268/ISAC-Research}
}
```

## 📧 联系

如有问题，请通过 GitHub Issues 联系。

---

**最新提交:** v2.2 - 大规模鲁棒系统 (16APs + 不完美CSI)
