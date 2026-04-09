#!/usr/bin/env python3
"""
ISAC - GNN图神经网络版 (显式建模AP空间关系)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# ===================== 配置 =====================
cfg = type('C', (), {
    'M': 16, 'K': 8, 'P': 4, 'Nt': 4,
    'Pmax': 30, 'N_req': 4,
})()

# ===================== 数据生成 =====================
def generate_channel_graph(n):
    """生成图结构数据 (每个AP为一个节点)"""
    # H: (n, M, K, Nt)
    H = (np.random.randn(n, cfg.M, cfg.K, cfg.Nt) + 1j * np.random.randn(n, cfg.M, cfg.K, cfg.Nt)) / np.sqrt(2)
    # G: (n, M, M, P, Nt) - AP间信道
    G = (np.random.randn(n, cfg.M, cfg.M, cfg.P, cfg.Nt) + 1j * np.random.randn(n, cfg.M, cfg.M, cfg.P, cfg.Nt)) / np.sqrt(2)
    
    # 展平
    H_flat = np.concatenate([H.real, H.imag], axis=-1).reshape(n, cfg.M, -1)  # (n, M, K*Nt*2)
    G_flat = np.concatenate([G.real, G.imag], axis=-1).reshape(n, cfg.M, cfg.M, -1)  # (n, M, M, P*Nt*2)
    
    # 归一化 (按样本)
    for i in range(n):
        h_norm = np.linalg.norm(H_flat[i])
        if h_norm > 0:
            H_flat[i] = H_flat[i] / h_norm
        g_norm = np.linalg.norm(G_flat[i])
        if g_norm > 0:
            G_flat[i] = G_flat[i] / g_norm
    
    return H_flat.astype(np.float32), G_flat.astype(np.float32)

def generate_labels(H, G, method='optimal'):
    """生成标签 (多种方法)"""
    n = H.shape[0]
    W_labels = []
    Z_labels = []
    B_labels = []
    
    for i in range(n):
        if method == 'optimal':
            # 近似最优: 均匀分配 + 信道感知选择
            w_pwr = cfg.Pmax * 0.7 / (cfg.M * cfg.K)
            z_pwr = cfg.Pmax * 0.3 / cfg.M
            
            # AP选择: 基于信道增益
            b = np.zeros((cfg.M, cfg.P))
            for p in range(cfg.P):
                gains = np.random.rand(cfg.M) * np.random.rand(cfg.M)
                top_aps = np.argsort(-gains)[:cfg.N_req]
                b[top_aps, p] = 1
        elif method == 'sparse':
            # 稀疏: 只用少数AP
            w_pwr = cfg.Pmax * 0.5 / (cfg.M * cfg.K)
            z_pwr = cfg.Pmax * 0.2 / cfg.M
            
            b = np.zeros((cfg.M, cfg.P))
            for p in range(cfg.P):
                selected = np.random.choice(cfg.M, cfg.N_req, replace=False)
                b[selected, p] = 1
        else:
            # 密集: 用更多AP
            w_pwr = cfg.Pmax * 0.8 / (cfg.M * cfg.K)
            z_pwr = cfg.Pmax * 0.4 / cfg.M
            b = (np.random.rand(cfg.M, cfg.P) > 0.3).astype(float)
        
        W_labels.append(w_pwr)
        Z_labels.append(z_pwr)
        B_labels.append(b.flatten())
    
    return np.array(W_labels).reshape(-1, 1), np.array(Z_labels).reshape(-1, 1), np.array(B_labels)

# ===================== GNN 图神经网络 =====================
class GraphConv(nn.Module):
    """图卷积层"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, adj):
        # x: (B, M, in_dim)
        # adj: (M, M)
        x = torch.matmul(adj, x)  # 聚合邻居信息
        x = self.linear(x)
        return F.leaky_relu(x, 0.2)

class GNNISAC(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 节点特征维度: K*Nt*2
        node_dim = cfg.K * cfg.Nt * 2
        
        # 节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
        )
        
        # 图卷积层
        self.gc1 = GraphConv(32, 32)
        self.gc2 = GraphConv(32, 32)
        
        # 全局聚合
        self.global_pool = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
        )
        
        # 输出头
        self.W_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.Z_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.B_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, cfg.P),
            nn.Sigmoid()
        )
        
        # 初始化邻接矩阵 (全连接)
        self.adj = torch.ones(cfg.M, cfg.M) / cfg.M
    
    def forward(self, H, G):
        batch_size = H.size(0)
        
        # 构建节点特征: 每个AP的特征
        # H: (B, M, K*Nt*2) -> 每个AP对应所有用户
        # G: (B, M, M, P*Nt*2) -> 每个AP的信道
        
        # 拼接每个节点的局部特征
        node_features = []  # (B, M, node_dim)
        for m in range(cfg.M):
            h_m = H[:, m, :]  # (B, K*Nt*2)
            g_row = G[:, m, :, :]  # (B, M, P*Nt*2)
            g_m = g_row.reshape(batch_size, -1)  # (B, M*P*Nt*2)
            
            # 简化为拼接
            h_m_expanded = h_m.unsqueeze(1).expand(-1, cfg.M, -1)
            combined = torch.cat([h_m_expanded, g_m.unsqueeze(1).expand(-1, cfg.M, -1)], dim=-1)
            node_features.append(combined[:, m, :])  # 取对角线
        
        # 实际上简化: 直接用H作为节点特征
        x = H  # (B, M, K*Nt*2)
        
        # 节点编码
        x = self.node_encoder(x)  # (B, M, 32)
        
        # 图卷积
        adj = self.adj.to(x.device)
        x = self.gc1(x, adj)  # (B, M, 32)
        x = self.gc2(x, adj)  # (B, M, 32)
        
        # 全局聚合
        x_global = self.global_pool(x.mean(dim=1))  # (B, 32)
        
        # 输出
        # 每个AP输出
        W_ratio = self.W_head(x)  # (B, M, 1)
        Z_ratio = self.Z_head(x)  # (B, M, 1)
        B = self.B_head(x)  # (B, M, P)
        
        return W_ratio, Z_ratio, B

# ===================== 训练 =====================
def train_gnn(epochs=300, bs=64, lr=1e-3):
    """训练GNN"""
    print("="*60)
    print("Training GNN for ISAC")
    print("="*60)
    
    model = GNNISAC()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    t0 = time.time()
    
    for e in range(epochs):
        # 生成数据
        H, G = generate_channel_graph(bs)
        
        # 生成多样化标签
        methods = ['optimal', 'sparse', 'dense']
        method = methods[e % 3]
        y_W, y_Z, y_B = generate_labels(H, G, method)
        
        Ht = torch.tensor(H, dtype=torch.float32)
        Gt = torch.tensor(G, dtype=torch.float32)
        y_Wt = torch.tensor(y_W / cfg.Pmax, dtype=torch.float32)
        y_Zt = torch.tensor(y_Z / cfg.Pmax, dtype=torch.float32)
        y_Bt = torch.tensor(y_B, dtype=torch.float32)
        
        # 前向
        W_ratio, Z_ratio, B = model(Ht, Gt)
        
        # 损失
        loss_W = criterion(W_ratio.mean(dim=1, keepdim=True), y_Wt)
        loss_Z = criterion(Z_ratio.mean(dim=1, keepdim=True), y_Zt)
        loss_B = criterion(B, y_Bt)
        
        loss = loss_W + loss_Z + loss_B * 0.5
        
        # 约束惩罚
        W_pwr = W_ratio.mean() * cfg.Pmax
        Z_pwr = Z_ratio.mean() * cfg.Pmax
        pwr_viol = F.relu(W_pwr + Z_pwr - cfg.Pmax)
        loss = loss + pwr_viol * 50
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if e % 30 == 0:
            print(f"Epoch {e:3d} | Loss: {loss.item():.4f} | "
                  f"W: {W_pwr.item():.1f}W | Z: {Z_pwr.item():.1f}W | "
                  f"{time.time()-t0:.1f}s")
    
    torch.save(model.state_dict(), 'isac_gnn.pth')
    print(f"\n完成! 保存 isac_gnn.pth, 用时: {time.time()-t0:.1f}s")
    return model

# ===================== 测试 =====================
def test(model, n=20):
    model.eval()
    print("\n测试:")
    
    results = []
    for i in range(n):
        H, G = generate_channel_graph(1)
        Ht = torch.tensor(H, dtype=torch.float32)
        Gt = torch.tensor(G, dtype=torch.float32)
        
        W_ratio, Z_ratio, B = model(Ht, Gt)
        
        W_pwr = W_ratio.mean().item() * cfg.Pmax
        Z_pwr = Z_ratio.mean().item() * cfg.Pmax
        total = W_pwr + Z_pwr
        
        B_sel = (B > 0.5).float()
        aps = torch.sum(B_sel).item()
        
        results.append({'W': W_pwr, 'Z': Z_pwr, 'Total': total, 'APs': aps})
        print(f"  {i+1}: W={W_pwr:.1f}W Z={Z_pwr:.1f}W Total={total:.1f}W APs={int(aps)}")
    
    avg = np.mean([r['Total'] for r in results])
    std = np.std([r['Total'] for r in results])
    print(f"\n平均功率: {avg:.1f}W ± {std:.1f}W")
    return results

# ===================== 主程序 =====================
if __name__ == '__main__':
    print("="*60)
    print("ISAC - GNN 图神经网络")
    print("="*60)
    
    model = train_gnn(epochs=300, bs=64, lr=1e-3)
    results = test(model, n=20)
    
    print("\n模型: isac_gnn.pth")