# 成功判定标准详解

## 单时隙成功判定

一个时隙成功需要**同时满足**以下4个条件：

```python
success = (
    np.all(sinrs >= self.sinr_req) and   # 所有用户SINR达标
    snr >= self.snr_req and               # 感知SNR达标  
    np.all(crbs <= self.crb_req) and      # 所有目标CRB达标
    power <= self.Pmax                    # 总功率不超
)
```

### 详细指标

| 指标 | 要求 | 公式 | 说明 |
|------|------|------|------|
| **SINR** | ≥ 10 dB | $\frac{|\mathbf{h}_k^H \mathbf{w}_k|^2}{\sum_{j \neq k} |\mathbf{h}_k^H \mathbf{w}_j|^2 + \sigma^2}$ | 信干噪比 |
| **SNR** | ≥ 10 dB | $\frac{\sum |\mathbf{g}^H \mathbf{Z} \mathbf{g}|^2}{\sigma^2 \sum \text{tr}(\mathbf{Z})}$ | 感知信噪比 |
| **CRB** | ≤ 1 m | $\frac{\sigma^2}{\sum |\mathbf{g}^H \mathbf{w}|^2}$ | Cramér-Rao下界 |
| **Power** | ≤ 3.2 W | $\|\mathbf{W}\|_F^2 + \sum \text{tr}(\mathbf{Z})$ | 总发射功率 |

---

## 时帧成功判定

对于T个时隙的时帧：

```python
frame_success = all([slot_success_t for t in range(T)])
```

**关键**: 时帧成功 = **所有T个时隙都成功**

这就是为什么：
- 单时隙成功率 70% → 时帧成功率 0.7⁵ = 17% (T=5)
- 单时隙成功率 70% → 时帧成功率 0.7³ = 34% (T=3)

---

## 优化过程中的成功判定

在算法内部，使用`compute_violation`来指导优化：

```python
def compute_violation(H, G, a, W, Z, margin_db=0):
    # 计算约束违反程度
    violation = 0
    violation += Σ max(0, SINR_required - SINR_actual)  # SINR缺口
    violation += max(0, SNR_required - SNR_actual)      # SNR缺口
    violation += Σ max(0, CRB_actual - CRB_required)    # CRB缺口
    
    if violation < 0.1:  # 所有约束满足
        return True  # 成功
```

---

## 参数设置

```python
M=64, K=10, P=4, Nt=4     # 系统参数
Pmax=3.2W                 # 总功率上限
sigma2=0.001              # 噪声功率
sinr_req=10dB             # SINR要求
snr_req=10dB              # SNR要求  
crb_req=1m                # CRB要求
error_var=0.001           # 信道估计误差
```

---

## 总结

| 判定级别 | 条件 | 当前表现 |
|---------|------|---------|
| 单时隙 | 4个约束同时满足 | 70-98% |
| 时帧(T=5) | 5个时隙全成功 | 5-15% |
| 时帧(T=3) | 3个时隙全成功 | 40% |

**关键洞察**: 时帧成功率随长度指数下降！
