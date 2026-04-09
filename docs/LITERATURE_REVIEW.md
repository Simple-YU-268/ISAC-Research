# ISAC目标跟踪文献综述

## 搜索日期: 2026-04-09

---

## 1. 核心发现

### 1.1 EKF (扩展卡尔曼滤波) - 最主流方法

**论文**: 
- "Cooperative Sensing-Assisted Predictive Beam Tracking for MIMO-OFDM Networked ISAC Systems" (arXiv 2025)
- "Predictive Beamforming with Distributed MIMO" (arXiv 2025)
- "Integrated Sensing, Communication, and Control for UAV-Assisted Mobile Target Tracking" (arXiv 2026)

**方法**:
```
状态: [x, y, vx, vy] (位置+速度)
观测: 目标回波信号
EKF: 预测-更新循环
```

**关键创新**:
- EKF融合多个BS的测量结果
- 线性化非线性观测模型
- 预测目标状态用于下一时刻波束设计

---

### 1.2 预测波束成形 (Predictive Beamforming)

**论文**: 
- "Cooperative Sensing-Assisted Predictive Beam Tracking" (arXiv 2025)

**方法**:
```
1. 当前时隙: 估计目标状态
2. EKF预测: 下一时刻目标位置
3. 计算PC-CRLB (预测条件CRLB)
4. 优化波束: 满足PC-CRLB约束
5. 下一时隙: 使用预测波束
```

**目标函数**:
- 最大化通信速率
- 约束: PC-CRLB ≤ 要求值

---

### 1.3 自适应波束方案

**论文**: 
- "Extended Target Adaptive Beamforming for ISAC: A Perspective of Predictive Error Ellipse" (arXiv 2026)

**方法**:
- MEE (Minimum Error Ellipse) 问题建模
- 自适应调整波束覆盖范围
- 减少EKF持续跟踪的计算开销

---

### 1.4 WMMSE-SDR算法

**论文**: 
- "Information and sensing beamforming optimization for multi-user multi-target MIMO ISAC systems" (Springer 2023)

**方法**:
- WMMSE (Weighted MMSE) 处理多用户干扰
- SDR (Semi-Definite Relaxation) 解决非凸问题
- 联合优化CRB和通信速率

---

## 2. 关键算法对比

| 方法 | 预测精度 | 复杂度 | 适用场景 |
|------|---------|--------|---------|
| EKF | 中 | 低 | 非线性运动 |
| UKF | 高 | 中 | 强非线性 |
| 粒子滤波 | 很高 | 高 | 多模态 |
| 深度学习 | 高 | 训练高/推理低 | 大数据 |

---

## 3. 对我们的启示

### 3.1 当前v38问题

**现状**:
- 实现了基本卡尔曼预测
- 但未用于指导波束设计
- 时帧成功率40% (T=3)

### 3.2 改进方向

**方向1: PC-CRLB指导波束设计**
```python
# 文献方法
1. EKF预测下一位置
2. 计算预测CRB (PC-CRLB)
3. 优化波束满足 PC-CRLB ≤ threshold
4. 同时最大化通信速率
```

**方向2: 自适应帧长**
```python
# 根据预测误差调整
if prediction_error > threshold:
    T = 3  # 短帧，高精度
else:
    T = 5  # 长帧，高效率
```

**方向3: 多AP协作感知**
```python
# 文献方法: 融合多个AP的测量
for each AP:
    measurement = sense_target()
    
# EKF融合所有测量
state = EKF_update(all_measurements)
```

---

## 4. 推荐实现

### 4.1 v39: PC-CRLB指导优化

基于文献方法改进v38:

```python
def optimize_with_pc_crlb(H_est, G_est, H_true, G_true, predicted_pos):
    """
    PC-CRLB指导的波束优化
    
    1. 基于预测位置计算PC-CRLB
    2. 优化波束满足 PC-CRLB ≤ crb_req
    3. 同时最大化通信SINR
    """
    # 计算预测信道
    G_pred = generate_channel_at(predicted_pos)
    
    # 计算PC-CRLB
    pc_crlb = compute_crlb(G_pred, W)
    
    # 优化: 最大化SINR, 约束 PC-CRLB ≤ 1m
    # 使用SCA或SDR求解
```

### 4.2 预期提升

| 方法 | 预期时帧成功率 |
|------|--------------|
| v38 (当前) | 40% |
| v39 (PC-CRLB) | 60-70% |
| v40 (自适应帧) | 75-80% |

---

## 5. 参考文献

1. arXiv:2508.12723 - Cooperative Sensing-Assisted Predictive Beam Tracking
2. arXiv:2501.17746 - Predictive Beamforming with Distributed MIMO  
3. arXiv:2602.05209 - Integrated Sensing, Communication, and Control
4. arXiv:2601.06125 - Extended Target Adaptive Beamforming
5. Springer 2023 - WMMSE-SDR for Multi-user Multi-target ISAC
