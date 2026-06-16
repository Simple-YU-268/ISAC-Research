#!/usr/bin/env python3
"""
Cell-Free ISAC Rigorous Solver v2.0 (Python)
Based on standard form with per-target AP selection, sensing SINR, and PCRB
Mathematical derivation: MATHEMATICAL_DERIVATION.md
"""

import numpy as np
import numpy.linalg as la

def default_config():
    return {
        'seed': 42,
        'M': 16, 'Nt': 4, 'K': 10, 'P': 1,
        'apGridSide': 4, 'apMin': -60, 'apMax': 60,
        'userMin': -50, 'userMax': 50,
        'targetMin': -50, 'targetMax': 50,
        'd0': 10, 'pathLossExp': 2.5, 'minDistance': 5,
        'epsilonH': 0.10, 'epsilonG': 0.15,
        'sigmaC2': 0.5, 'sigmaS2': 0.5,
        'Pmax': 30.0,
        'gammaK': 1.0,  # 0 dB linear
        'gammaS': 1.0,  # 0 dB linear
        'gammaTrack': 1.0,  # PCRB threshold
        'Nreq': 4,
        'nTrials': 20
    }

def generate_scenario(cfg):
    np.random.seed(cfg['seed'])
    
    # AP positions (grid)
    x = np.linspace(cfg['apMin'], cfg['apMax'], cfg['apGridSide'])
    y = np.linspace(cfg['apMin'], cfg['apMax'], cfg['apGridSide'])
    apPos = np.array([[xi, yi] for xi in x for yi in y])[:cfg['M']]
    
    # User positions
    userPos = cfg['userMin'] + (cfg['userMax'] - cfg['userMin']) * np.random.rand(cfg['K'], 2)
    
    # Target positions
    targetPos = cfg['targetMin'] + (cfg['targetMax'] - cfg['targetMin']) * np.random.rand(cfg['P'], 2)
    
    M, Nt, K, P = cfg['M'], cfg['Nt'], cfg['K'], cfg['P']
    
    # Communication channels H: MNt x K
    H = np.zeros((M*Nt, K), dtype=complex)
    for k in range(K):
        for m in range(M):
            d = max(np.linalg.norm(apPos[m] - userPos[k]), cfg['minDistance'])
            pl = (cfg['d0']/d)**cfg['pathLossExp']
            idx = slice(m*Nt, (m+1)*Nt)
            H[idx, k] = np.sqrt(pl) * (np.random.randn(Nt) + 1j*np.random.randn(Nt)) / np.sqrt(2)
    
    # Sensing channels G: MNt x P
    G = np.zeros((M*Nt, P), dtype=complex)
    for p in range(P):
        for m in range(M):
            d = max(np.linalg.norm(apPos[m] - targetPos[p]), cfg['minDistance'])
            pl = (cfg['d0']/d)**cfg['pathLossExp']
            idx = slice(m*Nt, (m+1)*Nt)
            G[idx, p] = np.sqrt(pl) * (np.random.randn(Nt) + 1j*np.random.randn(Nt)) / np.sqrt(2)
    
    return {'apPos': apPos, 'userPos': userPos, 'targetPos': targetPos, 'H': H, 'G': G}

def solve_one_scenario(cfg, scenario):
    result = {
        'success': False,
        'nActiveAPs': 0,
        'minSinrWcDb': -np.inf,
        'minSnrWcDb': -np.inf,
        'totalPower': np.inf,
        'violation': np.inf
    }
    
    # Robustness factors
    etaH = ((1 - cfg['epsilonH']) / (1 + cfg['epsilonH']))**2
    etaG = ((1 - cfg['epsilonG']) / (1 + cfg['epsilonG']))**2
    
    # Robust thresholds
    gammaK_robust = cfg['gammaK'] / etaH
    gammaS_robust = cfg['gammaS'] / etaG
    
    M, Nt, K, P = cfg['M'], cfg['Nt'], cfg['K'], cfg['P']
    H = scenario['H']
    G = scenario['G']
    
    # Step 1: AP Selection (per target)
    b_mp = np.zeros((M, P), dtype=int)
    for p in range(P):
        g_p = G[:, p]
        apStrength = np.zeros(M)
        for m in range(M):
            idx = slice(m*Nt, (m+1)*Nt)
            apStrength[m] = np.linalg.norm(g_p[idx])**2
        
        selected = np.argsort(apStrength)[-cfg['Nreq']:]
        b_mp[selected, p] = 1
    
    activeAPs = np.where(np.sum(b_mp, axis=1) > 0)[0]
    nActive = len(activeAPs)
    result['nActiveAPs'] = nActive
    
    # Step 2: Extract subchannels
    activeIdx = []
    for i, m in enumerate(activeAPs):
        activeIdx.extend(range(m*Nt, (m+1)*Nt))
    activeIdx = np.array(activeIdx)
    
    H_all = H[activeIdx, :]  # nActive*Nt x K
    G_all = G[activeIdx, :]    # nActive*Nt x P
    
    # Step 3: Communication beamforming (ZF)
    Wcomm = np.zeros((nActive*Nt, K), dtype=complex)
    Pcomm_per_k = np.zeros(K)
    useZF = False
    
    if np.linalg.matrix_rank(H_all) >= K:
        # ZF solution
        HHH = H_all.conj().T @ H_all
        try:
            Wzf = H_all @ la.inv(HHH)
            for k in range(K):
                wzf_k = Wzf[:, k]
                norm_wzf = np.linalg.norm(wzf_k)
                if norm_wzf > 1e-10:
                    w_k = wzf_k / norm_wzf
                    Pcomm_per_k[k] = gammaK_robust * cfg['sigmaC2'] * norm_wzf**2
                    Wcomm[:, k] = np.sqrt(Pcomm_per_k[k]) * w_k
            useZF = True
        except la.LinAlgError:
            pass
    
    if not useZF:
        # MRT fallback
        for k in range(K):
            h_k = H_all[:, k]
            Wcomm[:, k] = np.sqrt(cfg['sigmaC2']) * h_k / np.linalg.norm(h_k)
    
    Pcomm_total = np.sum(Pcomm_per_k)
    
    # Step 4: Sensing beamforming (Matched Filter)
    Wsens = np.zeros((nActive*Nt, P), dtype=complex)
    Psens_per_p = np.zeros(P)
    
    for p in range(P):
        g_p = G_all[:, p]
        norm_g = np.linalg.norm(g_p)
        if norm_g > 1e-10:
            Psens_per_p[p] = gammaS_robust * cfg['sigmaS2'] / norm_g**2
            Wsens[:, p] = np.sqrt(Psens_per_p[p]) * g_p / norm_g
    
    Psens_total = np.sum(Psens_per_p)
    
    # Step 5: Per-AP power check
    P_per_ap = np.zeros(nActive)
    for i, m in enumerate(activeAPs):
        idx_local = slice(i*Nt, (i+1)*Nt)
        
        Pcomm_ap = np.sum([np.linalg.norm(Wcomm[idx_local, k])**2 for k in range(K)])
        Psens_ap = np.sum([np.linalg.norm(Wsens[idx_local, p])**2 for p in range(P)])
        P_per_ap[i] = Pcomm_ap + Psens_ap
    
    max_ap_power = np.max(P_per_ap)
    total_power = np.sum(P_per_ap)
    
    # Step 6: Verification
    # Communication SINR
    minSinrWc = np.inf
    for k in range(K):
        hk = H_all[:, k]
        desired = np.abs(hk.conj() @ Wcomm[:, k])**2
        interf = cfg['sigmaC2']
        for j in range(K):
            if j != k:
                interf += np.abs(hk.conj() @ Wcomm[:, j])**2
        sinr_nom = desired / interf
        sinr_wc = sinr_nom * etaH
        minSinrWc = min(minSinrWc, sinr_wc)
    
    # Sensing SINR
    minSnrWc = np.inf
    for p in range(P):
        gp = G_all[:, p]
        desired = np.abs(gp.conj() @ Wsens[:, p])**2
        snr_nom = desired / cfg['sigmaS2']
        snr_wc = snr_nom * etaG
        minSnrWc = min(minSnrWc, snr_wc)
    
    # PCRB (simplified)
    minTraceJ = np.inf
    for p in range(P):
        traceJ = 0
        for k in range(K):
            for i, m in enumerate(activeAPs):
                idx_local = slice(i*Nt, (i+1)*Nt)
                g_mp = G[m*Nt:(m+1)*Nt, p]
                w_mk = Wcomm[idx_local, k]
                traceJ += np.abs(g_mp.conj() @ w_mk)**2
        minTraceJ = min(minTraceJ, traceJ)
    
    # Violation calculation
    vSinr = max(0, (cfg['gammaK'] - minSinrWc) / cfg['gammaK'])
    vSnr = max(0, (cfg['gammaS'] - minSnrWc) / cfg['gammaS'])
    vPower = max(0, (max_ap_power - cfg['Pmax']) / cfg['Pmax'])
    vPcrb = max(0, (cfg['gammaTrack'] - minTraceJ) / cfg['gammaTrack'])
    
    violation = max(vSinr, vSnr, vPower, vPcrb)
    
    # Result
    result['minSinrWcDb'] = 10*np.log10(minSinrWc) if minSinrWc > 0 else -np.inf
    result['minSnrWcDb'] = 10*np.log10(minSnrWc) if minSnrWc > 0 else -np.inf
    result['totalPower'] = total_power
    result['maxApPower'] = max_ap_power
    result['violation'] = violation
    result['success'] = (violation <= 1e-6)
    result['useZF'] = useZF
    result['Pcomm'] = Pcomm_total
    result['Psens'] = Psens_total
    result['b_mp'] = b_mp
    result['activeAPs'] = activeAPs
    
    return result

def main():
    cfg = default_config()
    
    nSuccess = 0
    results = []
    
    for trial in range(cfg['nTrials']):
        cfg['seed'] = 42 + trial  # Different seed per trial
        scenario = generate_scenario(cfg)
        result = solve_one_scenario(cfg, scenario)
        results.append(result)
        
        if result['success']:
            nSuccess += 1
            status = 'OK'
        else:
            status = 'FAIL'
        
        print(f"Trial {trial+1:02d}: {status} | Map={result['nActiveAPs']:2d} | "
              f"SNRwc={result['minSnrWcDb']:5.1f}dB | SINRwc={result['minSinrWcDb']:5.1f}dB | "
              f"P={result['totalPower']:5.2f}W | v={result['violation']:.4f}")
    
    print(f"\n=== RESULT: {nSuccess}/{cfg['nTrials']} = {nSuccess/cfg['nTrials']*100:.0f}% success ===")
    
    if nSuccess > 0:
        succ = [r for r in results if r['success']]
        avg_power = np.mean([r['totalPower'] for r in succ])
        avg_snr = np.mean([r['minSnrWcDb'] for r in succ])
        avg_sinr = np.mean([r['minSinrWcDb'] for r in succ])
        avg_aps = np.mean([r['nActiveAPs'] for r in succ])
        print(f"Avg: Power={avg_power:.2f}W, SNRwc={avg_snr:.1f}dB, SINRwc={avg_sinr:.1f}dB, ActiveAPs={avg_aps:.1f}")
    
    # Show detailed breakdown for first successful trial
    succ_trials = [r for r in results if r['success']]
    if succ_trials:
        r = succ_trials[0]
        print(f"\n=== Detailed Breakdown (Trial 1) ===")
        print(f"Active APs: {r['activeAPs']}")
        print(f"AP-Target association (b_mp):")
        print(r['b_mp'])
        print(f"Communication power: {r['Pcomm']:.2f}W")
        print(f"Sensing power: {r['Psens']:.2f}W")
        print(f"Max AP power: {r['maxApPower']:.2f}W (limit: {cfg['Pmax']}W)")
        print(f"ZF used: {r['useZF']}")
    else:
        # Show first trial details for debugging
        r = results[0]
        print(f"\n=== Debug: Trial 1 Details ===")
        print(f"Active APs: {r['activeAPs']}")
        print(f"AP-Target association (b_mp):")
        print(r['b_mp'])
        print(f"Communication power: {r['Pcomm']:.2f}W")
        print(f"Sensing power: {r['Psens']:.2f}W")
        print(f"Max AP power: {r['maxApPower']:.2f}W (limit: {cfg['Pmax']}W)")
        print(f"ZF used: {r['useZF']}")
        print(f"Violation: {r['violation']:.4f}")
        
        # Check per-AP power distribution
        print(f"\nPer-AP power distribution:")
        for i, ap in enumerate(r['activeAPs']):
            # Recalculate per-AP power
            pass  # Would need to store per-AP power in result

if __name__ == '__main__':
    main()
