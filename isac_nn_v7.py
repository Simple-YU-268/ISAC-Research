#!/usr/bin/env python3
"""
ISAC Beamforming - Final v7 (修复版)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

cfg = type('C', (), {
    'M': 16, 'K': 8, 'P': 4, 'Nt': 4,
    'Pmax': 30, 'N_req': 4,
    'gamma_k_dB': 5, 'sigma2': 1e-5
})()

# ===================== 数据 =====================
def generate_realistic(n):
    H = (np.random.randn(n, cfg.M, cfg.K, cfg.Nt) + 1j * np.random.randn(n, cfg.M, cfg.K, cfg.Nt)) / np.sqrt(2)
    G = (np.random.randn(n, cfg.M, cfg.M, cfg.P, cfg.Nt) + 1j * np.random.randn(n, cfg.M, cfg.M, cfg.P, cfg.Nt)) / np.sqrt(2)
    H = H.astype(np.complex64)
    G = G.astype(np.complex64)
    H_flat = np.concatenate([H.real, H.imag], axis=-1).reshape(n, -1)
    G_flat = np.concatenate([G.real, G.imag], axis=-1).reshape(n, -1)
    X = np.concatenate([H_flat, G_flat], axis=1).astype(np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

# ===================== 网络 =====================
class TransformerISAC(nn.Module):
    def __init__(self):
        super().__init__()
        
        hd = cfg.M * cfg.K * cfg.Nt * 2
        gd = cfg.M * cfg.M * cfg.P * cfg.Nt * 2
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(hd + gd, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 输出头
        self.w_pwr = nn.Linear(64, cfg.M * cfg.K)
        self.w_phase = nn.Linear(64, cfg.M * cfg.K * cfg.Nt)
        self.z_pwr = nn.Linear(64, cfg.M)
        self.b_logit = nn.Linear(64, cfg.M * cfg.P)
    
    def forward(self, x):
        # 输入投影
        h = self.input_proj(x)  # (B, 64)
        
        # 扩展为序列 (B, M, 64) - 模拟16个AP
        h = h.unsqueeze(1).expand(-1, cfg.M, -1)
        
        # Transformer
        h_trans = self.transformer(h)  # (B, M, 64)
        
        # 聚合
        h_fused = h_trans.mean(dim=1)  # (B, 64)
        
        # W
        w_pwr = torch.abs(self.w_pwr(h_fused)).sigmoid()
        w_pwr = w_pwr / (w_pwr.sum(dim=-1, keepdims=True) + 1e-8) * cfg.Pmax * 0.7
        phase = torch.tanh(self.w_phase(h_fused)).view(-1, cfg.M, cfg.K, cfg.Nt) * np.pi
        w_mag = (w_pwr.view(-1, cfg.M, cfg.K, 1) / cfg.Nt).sqrt()
        W = w_mag * torch.complex(phase.cos(), phase.sin())
        
        # Z
        z_pwr = torch.abs(self.z_pwr(h_fused)).sigmoid()
        z_pwr = z_pwr / (z_pwr.sum(dim=-1, keepdims=True) + 1e-8) * cfg.Pmax * 0.3
        Z = torch.zeros(cfg.M, cfg.Nt, cfg.Nt, dtype=torch.complex64, device=x.device)
        for i in range(cfg.M):
            Z[i] = torch.eye(cfg.Nt, device=x.device) * z_pwr[0, i].item() / cfg.Nt
        
        # b
        b = torch.sigmoid(self.b_logit(h_fused)).view(-1, cfg.M, cfg.P)
        
        return W, Z, b

# ===================== 损失 =====================
def loss_fn(W, Z, b):
    pwr_W = torch.sum(torch.abs(W)**2)
    pwr_Z = torch.sum(torch.real(Z))
    total = pwr_W + pwr_Z
    pwr_viol = torch.relu(total - cfg.Pmax)
    
    b_sel = (b > 0.5).float()
    sel_viol = torch.mean(torch.abs(torch.sum(b_sel, dim=1) - cfg.N_req))
    
    loss = total + pwr_viol * 100 + sel_viol * 20
    return loss, total.item(), pwr_viol.item(), sel_viol.item()

# ===================== 训练 =====================
def train(epochs=500, bs=64, lr=1e-3):
    model = TransformerISAC()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    print("="*55)
    print("Training Transformer ISAC v7")
    print("="*55)
    
    t0 = time.time()
    best = float('inf')
    
    for e in range(epochs):
        X = generate_realistic(bs)
        Xt = torch.tensor(X, dtype=torch.float32)
        
        W, Z, b = model(Xt)
        loss, pwr, pv, sv = loss_fn(W, Z, b)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if loss.item() < best:
            best = loss.item()
            torch.save(model.state_dict(), 'isac_v7.pth')
        
        if e % 50 == 0:
            print(f"Epoch {e:3d} | Loss: {loss.item():.4f} | "
                  f"Pwr: {pwr:.1f}W | Viol: {pv:.3f} | "
                  f"Sel: {sv:.2f} | {time.time()-t0:.1f}s")
    
    print(f"\n完成! 最佳: {best:.4f}, 用时: {time.time()-t0:.1f}s")
    model.load_state_dict(torch.load('isac_v7.pth'))
    return model

# ===================== 测试 =====================
def test(model, n=20):
    model.eval()
    print("\n测试:")
    
    results = []
    for i in range(n):
        X = generate_realistic(1)
        Xt = torch.tensor(X, dtype=torch.float32)
        
        W, Z, b = model(Xt)
        
        pw = torch.sum(torch.abs(W)**2).item()
        pz = torch.sum(torch.real(Z)).item()
        ap = torch.sum((b > 0.5).float()).item()
        
        results.append({'W': pw, 'Z': pz, 'Total': pw+pz, 'APs': ap})
        print(f"  {i+1}: W={pw:.1f}W Z={pz:.1f}W Total={pw+pz:.1f}W APs={int(ap)}")
    
    avg = np.mean([r['Total'] for r in results])
    std = np.std([r['Total'] for r in results])
    print(f"\n平均: {avg:.1f}W ± {std:.1f}W")
    return results

if __name__ == '__main__':
    print("="*55)
    print("ISAC Transformer v7 - 改进版")
    print("="*55)
    
    model = train(epochs=500, bs=64, lr=1e-3)
    results = test(model, n=20)
    
    print("\n模型: isac_v7.pth")