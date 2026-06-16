# ISAC-Research

Cell-Free Integrated Sensing and Communication (ISAC) — Rigorous Mathematical Analysis

## Files

| File | Description |
|------|-------------|
| `PROBLEM_FORMULATION_RIGOROUS.md` | **Core problem formulation** — aligned with standard ISAC literature |
| `MATHEMATICAL_DERIVATION.md` | Closed-form solutions (ZF/MF), complexity analysis, feasibility conditions |
| `SDP_DERIVATION_COMPLETE.md` | SDP relaxation, S-Procedure derivation, LMI forms, duality |
| `SDP_SUPPLEMENTARY_DERIVATION.md` | AP selection, multi-target extension, slot coupling, SDR tightness |
| `SDP_IMPLEMENTATION_DERIVATION.md` | **Complete SDP implementation derivation** — KKT conditions, numerical example, code framework |
| `DERIVATION_ANALYSIS.md` | Comparison: user derivation vs. closed-form implementation |
| `ADVANCED_MATHEMATICAL_ANALYSIS.md` | Sensing robustness, feasibility, complexity bounds, tightness, duality gap |
| `isac_final_solver.m` | Current MATLAB solver (ZF beamforming, 95% success rate) |

## Key Parameters

- $M=16$ APs, $N_t=4$ antennas per AP
- $K=10$ users, $P=4$ targets
- $P_{\max}=30$W per AP, $\gamma_k=0$dB, $\gamma_S^{\text{PoD}}=0$dB
- CSI errors: $\epsilon_h=0.10$, $\epsilon_g=0.15$
- Required APs per target: $N_{\text{req}}=4$

## Status

- Mathematical derivation: **complete** (7 documents, ~3000 lines)
- SDP solver implementation: **pending** (requires MOSEK or LMI-capable solver)
- MATLAB environment: **unavailable** on this machine
