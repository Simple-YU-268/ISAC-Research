#!/usr/bin/env python3
"""
ISAC - v12 (连续AP选择 + 软决策)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

cfg = type('C', (), {
    'M': 16, 'K': 8, 'P': 4, 'Nt': 4,
    'Pmax': 30, 'N_req': 4,
})()

def generate_data(n):
    H = np.random.randn(n, cfg.M, cfg.K, cfg.Nt*2).astype(np.float32)
    H = H / (np.linalg.norm(H, axis=(-1,-2), keepdims=True) + 1e-8)
    return H

def generate_labels_v12(n):
    """使用连续值表示AP选择强度"""
    W_labels, Z_labels, B_labels = [], [], []
    
    for i in range(n):
        r = np.random.rand()
        if r < 0.33:
            w_ratio, z_ratio = 0.7, 0.3
            b = np.zeros((cfg.M, cfg.P))
            for p in range(cfg.P):
                selected = np.random.choice(cfg.M, cfg.N_req, replace=False)
                b[selected, p] = 1.0  # 明确选择
        elif r < 0.66:
            w_ratio, z_ratio = 0.5, 0.2
            b = np.zeros((cfg.M, cfg.P))
            for p in range(cfg.P):
                selected = np.random.choice(cfg.M, 2, replace=False)
                b[selected, p] = 0.8
        else:
            w_ratio, z_ratio = 0.85, 0.4
            # 软选择
            b = np.random.rand(cfg.M, cfg.P)
        
        W_labels.append(w_ratio)
        Z_labels.append(z_ratio)
        B_labels.append(b.flatten())
    
    return np.array(W_labels).reshape(-1,1), np.array(Z_labels).reshape(-1,1), np.array(B_labels)

class ISAC_v12(nn.Module):
    def __init__(self):
        super().__init__()
        
        hd = cfg.M * cfg.K * cfg.Nt * 2
        
        # 更强的编码器
        self.encoder = nn.Sequential(
            nn.Linear(hd, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
        )
        
        self.W_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.Z_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        
        # B头 - 直接用Sigmoid输出概率
        self.B_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, cfg.M * cfg.P),
            # 不加Sigmoid，用BCEWithLogitsLoss
        )
    
    def forward(self, x):
        emb = self.encoder(x)
        
        W_ratio = torch.sigmoid(self.W_head(emb))
        Z_ratio = torch.sigmoid(self.Z_head(emb))
        
        # B用logits，然后用sigmoid
        B_logits = self.B_head(emb)
        
        return W_ratio, Z_ratio, B_logits

def train_v12(epochs=500, bs=64, lr=1e-3):
    model = ISAC_v12()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2)
    
    print("="*60)
    print("Training ISAC v12")
    print("="*60)
    
    t0 = time.time()
    
    for e in range(epochs):
        H = generate_data(bs)
        y_W, y_Z, y_B = generate_labels_v12(bs)
        
        Xt = torch.tensor(H.reshape(bs, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B_logits = model(Xt)
        
        W_target = torch.tensor(y_W, dtype=torch.float32).squeeze()
        Z_target = torch.tensor(y_Z, dtype=torch.float32).squeeze()
        B_target = torch.tensor(y_B, dtype=torch.float32)
        
        # 损失
        loss_W = F.mse_loss(W_ratio.squeeze(), W_target)
        loss_Z = F.mse_loss(Z_ratio.squeeze(), Z_target)
        
        # BCE with logits
        loss_B = F.binary_cross_entropy_with_logits(B_logits, B_target)
        
        loss = loss_W + loss_Z + loss_B * 0.3
        
        # 功率约束
        W_pwr = W_ratio.squeeze() * cfg.Pmax
        Z_pwr = Z_ratio.squeeze() * cfg.Pmax
        total_pwr = W_pwr + Z_pwr
        
        overuse = F.relu(total_pwr - cfg.Pmax).mean() * 50
        underuse = F.relu(cfg.Pmax * 0.2 - total_pwr).mean() * 20
        
        loss = loss + overuse + underuse
        
        # 软AP选择奖励 - 鼓励选择
        B_prob = torch.sigmoid(B_logits)
        avg_sel = B_prob.mean()
        loss = loss - avg_sel * 0.1  # 奖励选择更多AP
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if e % 50 == 0:
            w_pwr = W_ratio.mean().item() * cfg.Pmax
            z_pwr = Z_ratio.mean().item() * cfg.Pmax
            total = w_pwr + z_pwr
            aps = (torch.sigmoid(B_logits) > 0.5).float().sum(dim=1).mean().item()
            print(f"Epoch {e:3d} | Loss: {loss.item():.4f} | "
                  f"W: {w_pwr:.1f}W | Z: {z_pwr:.1f}W | Total: {total:.1f}W | APs: {aps:.1f}")
    
    torch.save(model.state_dict(), 'isac_v12.pth')
    print(f"\n完成!")
    return model

def test_v12(model, n=20):
    model.eval()
    print("\n测试:")
    
    results = []
    for i in range(n):
        H = generate_data(1)
        Xt = torch.tensor(H.reshape(1, -1), dtype=torch.float32)
        
        W_ratio, Z_ratio, B_logits = model(Xt)
        
        W_pwr = W_ratio.item() * cfg.Pmax
        Z_pwr = Z_ratio.item() * cfg.Pmax
        total = W_pwr + Z_pwr
        
        B_prob = torch.sigmoid(B_logits)
        
        # 使用软阈值
        B_sel = (B_prob > 0.3).float()  # 更低的阈值
        aps = B_sel.sum().item()
        
        B_reshaped = B_sel.view(cfg.M, cfg.P)
        aps_per_target = B_reshaped.sum(dim=0).tolist()
        
        results.append({'W': W_pwr, 'Z': Z_pwr, 'Total': total, 'APs': aps, 'per_target': aps_per_target})
        print(f"  {i+1}: W={W_pwr:.1f}W Z={Z_pwr:.1f}W Total={total:.1f}W APs={int(aps)} | 每目标: {[int(x) for x in aps_per_target]}")
    
    avg = np.mean([r['Total'] for r in results])
    std = np.std([r['Total'] for r in results])
    aps_avg = np.mean([r['APs'] for r in results])
    print(f"\n平均功率: {avg:.1f}W ± {std:.1f}W")
    print(f"平均AP数: {aps_avg:.1f}")
    return results

if __name__ == '__main__':
    print("="*60)
    print("ISAC v12")
    print("="*60)
    
    model = train_v12(epochs=500, bs=64, lr=1e-3)
    results = test_v12(model, n=20)