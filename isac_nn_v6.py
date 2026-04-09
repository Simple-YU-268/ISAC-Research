#!/usr/bin/env python3
"""
ISAC Beamforming - Final v6
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

cfg = type('C', (), {
    'M': 16, 'K': 8, 'P': 4, 'Nt': 4,
    'Pmax': 30, 'N_req': 4,
})()

def generate_batch(n):
    # H: (B, M*K*Nt*2) 展平
    H = np.random.randn(n, cfg.M*cfg.K*cfg.Nt*2).astype(np.float32)
    # G: (B, M*M*P*Nt*2) 展平
    G = np.random.randn(n, cfg.M*cfg.M*cfg.P*cfg.Nt*2).astype(np.float32)
    # 归一化
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
    return np.concatenate([H, G], axis=-1)

class ISACNet(nn.Module):
    def __init__(self):
        super().__init__()
        hd = cfg.M*cfg.K*cfg.Nt*2
        gd = cfg.M*cfg.M*cfg.P*cfg.Nt*2
        
        self.net = nn.Sequential(
            nn.Linear(hd + gd, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
        )
        
        self.w_pwr = nn.Linear(32, cfg.M * cfg.K)
        self.w_phase = nn.Linear(32, cfg.M * cfg.K * cfg.Nt)
        self.z_pwr = nn.Linear(32, cfg.M)
        self.b_logit = nn.Linear(32, cfg.M * cfg.P)
    
    def forward(self, x):
        h = self.net(x)
        
        # W
        w_pwr = torch.abs(self.w_pwr(h)).sigmoid()
        w_pwr = w_pwr / (w_pwr.sum(dim=-1, keepdims=True) + 1e-8) * cfg.Pmax * 0.7
        phase = torch.tanh(self.w_phase(h)).view(-1, cfg.M, cfg.K, cfg.Nt) * np.pi
        w_mag = (w_pwr.view(-1, cfg.M, cfg.K, 1) / cfg.Nt).sqrt()
        W = w_mag * torch.complex(phase.cos(), phase.sin())
        
        # Z
        z_pwr = torch.abs(self.z_pwr(h)).sigmoid()
        z_pwr = z_pwr / (z_pwr.sum(dim=-1, keepdims=True) + 1e-8) * cfg.Pmax * 0.3
        Z = torch.zeros(cfg.M, cfg.Nt, cfg.Nt, dtype=torch.complex64, device=x.device)
        for i in range(cfg.M):
            Z[i] = torch.eye(cfg.Nt, device=x.device) * z_pwr[0,i].item() / cfg.Nt
        
        # b
        b = torch.sigmoid(self.b_logit(h)).view(-1, cfg.M, cfg.P)
        
        return W, Z, b

def loss_fn(W, Z, b):
    pwr_W = torch.sum(torch.abs(W)**2)
    pwr_Z = torch.sum(torch.real(Z))
    total = pwr_W + pwr_Z
    pwr_viol = torch.relu(total - cfg.Pmax)
    
    b_sel = (b > 0.5).float()
    sel_viol = torch.mean(torch.abs(torch.sum(b_sel, dim=1) - cfg.N_req))
    
    loss = total + pwr_viol * 100 + sel_viol * 20
    return loss, total.item(), pwr_viol.item(), sel_viol.item()

def train(epochs=300, bs=64, lr=1e-3):
    model = ISACNet()
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    print("Training v6...")
    t0 = time.time()
    
    for e in range(epochs):
        X = generate_batch(bs)
        Xt = torch.tensor(X, dtype=torch.float32)
        
        W, Z, b = model(Xt)
        loss, pwr, pv, sv = loss_fn(W, Z, b)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        
        if e % 30 == 0:
            print(f"Epoch {e:3d} | Loss: {loss.item():.4f} | "
                  f"Pwr: {pwr:.1f}W | Viol: {pv:.3f} | "
                  f"Sel: {sv:.2f} | {time.time()-t0:.1f}s")
    
    torch.save(model.state_dict(), 'isac_v6.pth')
    print(f"Done! Saved isac_v6.pth")
    return model

def test(model, n=10):
    model.eval()
    print("\n测试:")
    for i in range(n):
        X = generate_batch(1)
        Xt = torch.tensor(X, dtype=torch.float32)
        
        W, Z, b = model(Xt)
        pw = torch.sum(torch.abs(W)**2).item()
        pz = torch.sum(torch.real(Z)).item()
        ap = torch.sum((b > 0.5).float()).item()
        print(f"  {i+1}: W={pw:.1f}W Z={pz:.1f}W Total={pw+pz:.1f}W APs={int(ap)}")

if __name__ == '__main__':
    print("="*50)
    print("ISAC NN v6")
    print("="*50)
    m = train(epochs=300)
    test(m, n=10)