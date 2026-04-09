# ISAC系统参数配置
# 行业标准参数 (基于3GPP和文献调研)

## 系统参数
M = 64          # AP数量 (随机分布)
Nt = 16          # 每AP天线数
K = 10           # 通信用户数
P = 4            # 感知目标数
sigma2 = 0.001   # 噪声功率
Pmax = 3.2       # 总功率 (W)

## 行业标准约束 (基于3GPP ISAC)
# 参考: 3GPP TR 22.837, "CRB-Rate Tradeoff in ISAC" (V27DK4RL)
sinr_req = 10    # 通信SINR要求 (dB) - 行业标准
snr_req = 10     # 感知SNR要求 (dB) - 行业标准
crb_req = 1      # CRB要求 (<1m位置精度)

## 功率分配
# 通信:感知 = 0.6:0.4 (可调)
P_comm_ratio = 0.6
P_sens_ratio = 0.4

## 信道估计
error_var = 0.05  # 信道估计误差方差

## AP选择
n_ap_default = 7  # 默认选择AP数
alpha = 0.5        # 通信/感知权重 (0.5=均衡)
