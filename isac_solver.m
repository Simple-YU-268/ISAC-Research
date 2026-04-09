% ISAC系统 - 三步式AP选择 + 鲁棒波束成形
% MATLAB版本
%
% 系统参数:
%   M = 64 APs (随机分布)
%   Nt = 16 天线/AP
%   K = 10 通信用户
%   P = 4 感知目标
%   Pmax = 3.2W 总功率

clear; clc; close all;

%% 参数设置
M = 64;           % AP数量
Nt = 16;          % 每AP天线数
K = 10;           % 通信用户数
P = 4;            % 感知目标数
sigma2 = 0.001;   % 噪声功率
Pmax = 3.2;       % 总功率 (W)
P_comm = 0.6 * Pmax;  % 通信功率
P_sens = 0.4 * Pmax;  % 感知功率
n_ap = 7;         % 激活AP数
error_var = 0.05; % 信道估计误差方差

fprintf('=== ISAC系统 MATLAB求解器 ===\n');
fprintf('参数: M=%d, Nt=%d, K=%d, P=%d, Pmax=%.1fW\n', M, Nt, K, P, Pmax);

%% 生成随机AP分布
rng(42); % 固定随机种子
ap_x = rand(M, 1) * 100 - 50;
ap_y = rand(M, 1) * 100 - 50;
ap = [ap_x, ap_y];

% 用户和目标位置
user_pos = rand(K, 2) * 40 - 20;  % [-20, 20]
target_pos = rand(P, 2) * 80 - 40; % [-40, 40]

%% 生成信道
H_u = zeros(M, K, Nt);
H_t = zeros(M, P, Nt);

for m = 1:M
    for k = 1:K
        d = max(norm(ap(m,:) - user_pos(k,:)), 3);
        pl = (d / 10)^(-2.5);
        H_u(m, k, :) = sqrt(pl/2) * (randn(Nt,1) + 1j*randn(Nt,1));
    end
    for p = 1:P
        d = max(norm(ap(m,:) - target_pos(p,:)), 3);
        pl = (d / 10)^(-2.5);
        H_t(m, p, :) = sqrt(pl/2) * (randn(Nt,1) + 1j*randn(Nt,1));
    end
end

%% 添加信道估计误差
H_u_est = H_u + sqrt(error_var/2) * (randn(M, K, Nt) + 1j*randn(M, K, Nt));
H_t_est = H_t + sqrt(error_var/2) * (randn(M, P, Nt) + 1j*randn(M, P, Nt));

%% Step 1: 目标探测
detection_score = zeros(M, 1);
for p = 1:P
    h_stack = reshape(H_t_est(p, :, :), [M, Nt]);
    detection_score = detection_score + sum(abs(h_stack).^2, 2);
end

%% Step 2: AP选择
s_t = detection_score / max(detection_score);
s_u = squeeze(sum(abs(H_u_est).^2, [2,3]));
s_u = s_u / max(s_u);
combined = 0.5 * s_u + 0.5 * s_t;

[~, selected_idx] = sort(combined, 'descend');
selected_idx = selected_idx(1:n_ap);
ap_mask = false(M, 1);
ap_mask(selected_idx) = true;

fprintf('\n选择AP索引: ');
fprintf('%d ', selected_idx);

%% Step 3: 鲁棒MMSE通信波束
H_u_sel = H_u_est(ap_mask, :, :);
N_total = n_ap * Nt;

Hs = reshape(H_u_sel, [N_total, K]);
A = Hs * Hs' + sigma2 * (1 + 10*error_var) * eye(N_total);
W = A \ Hs;  % MMSE解

% 功率归一化
p_w = sum(abs(W(:)).^2);
if p_w > P_comm
    W = W * sqrt(P_comm / p_w);
end
W = reshape(W, [n_ap, Nt, K]);

%% Step 4: 感知波束 (匹配滤波)
Z = zeros(n_ap, Nt, P);
for p = 1:P
    h_stack = reshape(H_t_est(ap_mask, p, :), [n_ap * Nt, 1]);
    norm_h = norm(h_stack);
    if norm_h > 0
        z_p = conj(h_stack) / norm_h * sqrt(P_sens / P);
        Z(:, :, p) = reshape(z_p, [n_ap, Nt]);
    end
end

%% 验证约束
fprintf('\n\n=== 约束验证 ===\n');

% 通信SINR
H_u_true_sel = H_u(ap_mask, :, :);
Hs_true = reshape(H_u_true_sel, [N_total, K]);
Wf = reshape(W, [N_total, K]);

comm_pass = 0;
for k = 1:K
    sig = abs(Wf(:, k)' * Hs_true(:, k))^2;
    inter = 0;
    for j = 1:K
        if j ~= k
            inter = inter + abs(Wf(:, j)' * Hs_true(:, k))^2;
        end
    end
    sinr_db = 10*log10(sig / (inter + sigma2 + 1e-10));
    fprintf('用户%d SINR: %.2f dB %s\n', k, sinr_db, iif(sinr_db >= 0, '✓', '✗'));
    if sinr_db >= 0
        comm_pass = comm_pass + 1;
    end
end
fprintf('通信: %d/%d 用户达标\n', comm_pass, K);

% 感知SNR
sensing_pass = 0;
for p = 1:P
    signal = 0;
    for m = 1:n_ap
        signal = signal + abs(Z(m,:,p) * conj(H_t(ap_mask(m), p, :))')^2;
    end
    noise = sigma2 * sum(abs(Z(:)).^2);
    snr_db = 10*log10(signal / (noise + 1e-10));
    fprintf('目标%d SNR: %.2f dB %s\n', p, snr_db, iif(snr_db >= 0, '✓', '✗'));
    if snr_db >= 0
        sensing_pass = sensing_pass + 1;
    end
end
fprintf('感知: %d/%d 目标达标\n', sensing_pass, P);

% 功率
total_power = sum(abs(W(:)).^2) + sum(abs(Z(:)).^2);
fprintf('功率: %.4fW / %.1fW %s\n', total_power, Pmax, iif(total_power <= Pmax, '✓', '✗'));

fprintf('\n=== 结果 ===\n');
if comm_pass == K && sensing_pass == P && total_power <= Pmax
    fprintf('✅ 所有约束满足!\n');
else
    fprintf('❌ 部分约束未满足\n');
end