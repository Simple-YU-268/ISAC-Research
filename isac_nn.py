#!/usr/bin/env python3
"""
ISAC Beamforming - v5 (更严格的功率约束)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

# ===================== 配置 =====================
cfg = type('C', (), {
    'M': 16, 'K': 8, 'P': 4, 'Nt': 4,
    'Pmax': 30, 'N_req': 4
})()

def generate_data(n):
    h = np.random.randn(n, cfg.M*cfg.K*cfg.Nt*2).astype(np.float32)
    g = np.random.randn(n, cfg.M*cfg.M*cfg.P*cfg.Nt*2).astype(np.float32)
    x = np.concatenate([h, g], 1)
    return x / (np.linalg.norm(x, 1, keepdims=True) + 1e-8)

class ISACNet(nn.Module):
    def __init__(self):
        super().__init__()
        d = cfg.M*cfg.K*cfg.Nt*2 + cfg.M*cfg.M*cfg.P*cfg.Nt*2
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
        )
        # W: 每个(M,K)分配功率比例
        self.w_fc = nn.Linear(32, cfg.M*cfg.K)
        # 相位
        self.phase_fc = nn.Linear(32, cfg.M*cfg.K*cfg.Nt)
        # Z: 每个AP感知功率
        self.z_fc = nn.Linear(32, cfg.M)
        # b: AP选择
        self.b_fc = nn.Linear(32, cfg.M*cfg.P)
    
    def forward(self, x):
        h = self.net(x)
        
        # W: 功率 + 相位
        # 输出比例，然后缩放到 Pmax
        w_ratio = torch.abs(self.w_fc(h)).sigmoid()  # (B, M*K)
        w_ratio = w_ratio / (w_ratio.sum(dim=-1, keepdim=True) + 1e-8)  # 归一化
        w_pwr = w_ratio * cfg.Pmax * 0.8  # 留一些给Z
        
        phase = self.phase_fc(h).tanh().view(-1, cfg.M, cfg.K, cfg.Nt) * np.pi
        w_mag = (w_pwr.view(-1, cfg.M, cfg.K, 1) / cfg.Nt).sqrt()
        W = w_mag * torch.complex(phase.cos(), phase.sin())
        
        # Z: 感知功率 (对角矩阵)
        z_ratio = torch.abs(self.z_fc(h)).sigmoid()
        z_ratio = z_ratio / (z_ratio.sum(dim=-1, keepdims=True) + 1e-8)
        z_pwr = z_ratio * cfg.Pmax * 0.2  # 20%给感知
        
        Z = torch.zeros(cfg.M, cfg.Nt, cfg.Nt, dtype=torch.complex64, device=x.device)
        for i in range(cfg.M):
            if z_pwr[0, i] > 0.01:
                Z[i] = torch.eye(cfg.Nt, device=x.device) * z_pwr[0, i] / cfg.Nt
        
        # b: AP选择
        b = self.b_fc(h).sigmoid().view(-1, cfg.M, cfg.P)
        
        return W, Z, b

def loss_fn(W, Z, b):
    # 严格功率约束
    pwr_W = torch.sum(torch.abs(W)**2)
    pwr_Z = torch.sum(torch.real(Z))
    total = pwr_W + pwr_Z
    pwr_viol = torch.relu(total - cfg.Pmax)
    
    # AP选择: 每目标N_req个
    b_sel = (b > 0.5).float()
    sel_cnt = torch.sum(b_sel, dim=1)  # (B, P)
    sel_viol = torch.mean(torch.abs(sel_cnt - cfg.N_req))
    
    # 目标: 最小化功率
    loss = total + pwr_viol * 500 + sel_viol * 50
    
    return loss, total.item(), pwr_viol.item(), sel_viol.item()

def train(epochs=300, bs=64, lr=5e-4):
    model = ISACNet()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    print("Training v5...")
    start = time.time()
    
    for e in range(epochs):
        X = torch.tensor(generate_data(bs), dtype=torch.float32)
        W, Z, b = model(X)
        loss, pwr, pv, sv = loss_fn(W, Z, b)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        
        if e % 30 == 0:
            print(f"Epoch {e:3d} | Loss: {loss.item():.4f} | "
                  f"Total: {pwr:.1f}W | Viol: {pv:.3f} | Sel: {sv:.2f} | {time.time()-start:.1f}s")
    
    # 保存
    torch.save(model.state_dict(), 'isac_v5.pth')
    print(f"完成! 保存isac_v5.pth, 总时间: {time.time()-start:.1f}s")
    return model

def test(model, n=10):
    model.eval()
    print("\n测试结果:")
    results = []
    for i in range(n):
        X = torch.tensor(generate_data(1), dtype=torch.float32)
        W, Z, b = model(X)
        pw = torch.sum(torch.abs(W)**2).item()
        pz = torch.sum(torch.real(Z)).item()
        ap = torch.sum((b > 0.5).float()).item()
        results.append({'W': pw, 'Z': pz, 'Total': pw+pz, 'APs': ap})
        print(f"  {i+1}: W={pw:.1f}W Z={pz:.1f}W Total={pw+pz:.1f}W APs={int(ap)}")
    
    avg = np.mean([r['Total'] for r in results])
    print(f"\n平均总功率: {avg:.1f}W")
    return results

if __name__ == '__main__':
    print("="*50)
    print("ISAC NN v5 - 严格功率约束")
    print("="*50)
    m = train(epochs=300, bs=64, lr=5e-4)
    test(m, n=10)