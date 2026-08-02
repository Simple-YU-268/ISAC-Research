# Poster Copy

## Title

**Robust Beamforming for Participation-Constrained Cell-Free ISAC**

*Geometry-aware AP-target sensing authorization with globally cooperative communication*

## Why this matters

- Each target needs exactly `Nreq` authorized APs for its dedicated sensing waveform.
- AP-target geometry determines whether the PCRB tracking constraint can be met efficiently.
- Communication beams should remain globally cooperative; only dedicated sensing waveforms are gated.

## Proposed design

\[
\mathbf R_X=\sum_k\mathbf W_k+\sum_p\mathbf S_p.
\]

1. **Relax:** solve the robust SDR with AP power, PCRB, and SINR constraints.
2. **Recover:** use rank and binary DC penalties, then recover an integer association.
3. **Certify:** re-optimize with fixed association and accept only physically validated solutions.

## Evidence

- Dual DC reduces the median binary distance to `5.94e-5`; rank-only relaxation remains near `0.43`.
- In the 30-seed method study, proposed recovery is physically feasible at every tested cluster cardinality; distance-only and random associations fail in challenging regimes.
- Under sampled CSI perturbations, the robust design has zero observed outage; the nominal design has 89--92% mean outage.
- In the M12 study, the lowest observed conditional mean power is 35.72 mW at `Nreq=4`, while PCRB remains tight.

## Takeaway

Dedicated sensing should be geometry-aware and sparsely authorized.  Dual-DC
recovery finds physically feasible binary clusters, while robust beamforming
uses global communication degrees of freedom to meet QoS constraints.

## Footer

Metrics: feasibility is unconditional; power and QoS values are conditioned on
physical feasibility unless stated otherwise.
