# Gaussian Randomization for Rank-One Recovery

**Plate**: III (Engineering Feasibility) — supplement to `PAPER_OUTLINE_IEEE.md` §IV
**Source of math**: `math_derivation{,_en}.tex` §3 (the SDR step, line 112)
**References**:
- Luo, Luo, Chen, Pu, Anderson, "Semidefinite relaxation of quadratic optimization problems," *IEEE Signal Processing Magazine*, 2010. (Algorithm template)
- Sidiropoulos, Luo, "A semidefinite relaxation approach to MIMO detection for high-order QAM constellations," *IEEE Signal Processing Letters*, 2006. (Original Gaussian randomization for detection)
- Mezencev, "Recovery of rank-one solutions from semidefinite relaxations," Master's thesis, KTH, 2014. (Concentration analysis)

**Status**: v1.0 (2026-06-30). New document. Companion to
`SDR_TIGHTNESS.md` and `CONVEXIFICATION_CHAIN.md` §3.2.

---

## 1. Why This Document

After SDR (Step 2), the optimal solution
$\{\mat{W}_k^\star, \mat{Z}^\star, \mu_k^\star\}$ is **not** rank-one in
general. To recover a feasible (P1) solution, we need a procedure that
converts the high-rank $\mat{W}_k^\star$ into a rank-1
$\vect{w}_k \vect{w}_k^H$. Gaussian randomization is the standard
engineering procedure.

This document specifies the algorithm, its theoretical analysis, and
its known limitations for our cell-free ISAC setting.

---

## 2. The Algorithm

### 2.1 Inputs and outputs

**Inputs.**
- The SDR solution $\{\mat{W}_k^\star, \mat{Z}^\star, \mu_k^\star\}$ of (P3).
- A sample budget $L$ (typical: $L = 100$ to $L = 1000$).
- A feasibility check oracle (call it `IsFeasible`), which tests
  whether a given $\{\vect{w}_k, \mat{Z}\}$ satisfies all (P1) constraints.

**Outputs.** A feasible (P1) solution
$\{\vect{w}_k^\star, \mat{Z}^\star, b_{mp}^\star\}$ that approximately
minimizes the (P1) objective.

### 2.2 The procedure

```
Algorithm 1: Gaussian Randomization for Cell-Free ISAC

Input: SDR solution {W_k^*, Z^*, mu_k^*}, sample budget L
Output: Feasible (P1) solution {w_k, Z, b_{mp}}

1. (Eigendecomposition) For each k, compute the eigen-decomposition
   W_k^* = U_k Lambda_k U_k^H.  (MN_t x MN_t matrix, O((MN_t)^3).)

2. (Sampling) For each sample l = 1, ..., L:
   (a) For each k, draw w_k^{(l)} = U_k Lambda_k^{1/2} xi_k^{(l)},
       where xi_k^{(l)} ~ CN(0, I_{MN_t}) i.i.d.
   (b) For sensing, compute the (per-AP) sensing stream z_m^{(l)} as
       the eigenvector of the m-th diagonal block of Z^* corresponding
       to its largest eigenvalue, scaled to satisfy the per-AP power
       budget.  (No Gaussian sampling for the sensing stream because
       the per-AP budget is non-stochastic in our setting.)
   (c) (Feasibility) Test whether {w_k^{(l)}, z_m^{(l)}} is feasible
       under (P1) constraints (P1-C1) through (P1-C6).  This is the
       expensive step; see §2.3.
   (d) (Cost) Record the (P1) objective value J^{(l)}.

3. (Selection) Among the feasible samples, return the one with the
   smallest J^{(l)}.  If no sample is feasible, return the SDR
   solution (which is a (P3) feasible point, but not (P1) feasible
   when rank > 1).

4. (AP selection) For the AP selection subproblem (P1-C4), use a
   separate heuristic: for each target p, select the top-N_req APs
   by large-scale fading PL(d_{m,p}).
```

The total runtime is dominated by:
- Eigendecomposition: $O((MN_t)^3)$ per user, total
  $O(K (MN_t)^3)$.
- Sampling and feasibility: $O(L \cdot K \cdot (MN_t)^2 + L \cdot
  \text{cost}(\texttt{IsFeasible}))$ where `IsFeasible` is the
  worst-case SINR / PCRB check.

### 2.3 The feasibility oracle

`IsFeasible` must check (P1-C1)–(P1-C6):

- (P1-C1) worst-case comm SINR: solve the S-Procedure LMI with the
  candidate $\vect{w}_k^{(l)}$; the LMI is feasible iff the candidate
  is (P1-C1)-feasible. Cost: $O((MN_t)^3)$ per user, total
  $O(K (MN_t)^3)$.
- (P1-C2) sensing SINR: direct evaluation,
  $O(P (MN_t)^2)$.
- (P1-C3) PCRB: trace of $(\mat{J}_p^{\text{data}})^{-1}$, cost
  $O(P (MN_t)^3)$.
- (P1-C5) per-AP power: direct check, $O(MN_t)$.

The dominant cost is the S-Procedure check (P1-C1), so each feasibility
check is $O(K (MN_t)^3)$ and the full sampling loop is
$O(L \cdot K (MN_t)^3)$.

---

## 3. Theoretical Analysis

### 3.1 Concentration of the sampled objective

For each user $k$ and sample $\ell$, the sampled beamforming vector
$\vect{w}_k^{(\ell)}$ satisfies
$\mathbb{E}[\vect{w}_k^{(\ell)} (\vect{w}_k^{(\ell)})^H] = \mat{W}_k^\star$.

For the **unconstrained** case (no per-AP power budget, no worst-case
SINR), the sampled objective concentrates around the SDR objective:

$$
J^{(\ell)} \xrightarrow{\text{concentrate}} J_{\text{SDR}}^* \quad \text{as } L \to \infty
$$

This is the standard Gaussian concentration result (see Mezencev 2014
for the formal statement in the QCQP setting).

### 3.2 The cell-free ISAC setting: where concentration fails

For our cell-free ISAC setting, concentration fails for two reasons:

1. **Per-AP power budget.** Each sample $\vect{w}_k^{(\ell)}$ has
   $\sum_{m=1}^M \|\vect{w}_{m,k}^{(\ell)}\|^2$ that is a sum of
   $MN_t$ i.i.d. complex exponentials with mean $\|\mat{W}_k^\star\|_F^2
   / MN_t$. The per-AP budget $\tr(\mat{E}_m \mat{R}_X) \leq P_{\max}$ is
   a **per-coordinate** constraint, and the joint feasibility over
   $M$ per-AP constraints has exponentially small probability for
   high-rank $\mat{W}_k^\star$.

   Concretely: for rank-$r$ $\mat{W}_k^\star$, the per-AP energy has
   variance $\sim r / M$, and the probability that all $M$ per-AP
   budgets are simultaneously satisfied scales as
   $\exp(-c \cdot M)$ for some $c > 0$ depending on the rank.

2. **Worst-case SINR robustness.** The (P1-C1) constraint requires
   the SINR to hold for **all** $\Delta\vect{h} \in \mathcal{B}_\epsilon$,
   not just the expected channel. Each sample
   $\vect{w}_k^{(\ell)}$ achieves a different worst-case SINR. The
   probability that a random sample achieves $\gamma_k^{\text{wc}}
   \geq \gamma_k$ for non-trivial $\epsilon_h$ scales as
   $\exp(-c' \cdot \epsilon_h^{-2})$.

**Consequence.** For our cell-free ISAC setting, the standard
"L = 100 - 1000 candidates gives near-optimal recovery" rule of thumb
**does not have a theoretical backing**. The $L$ value must be
calibrated empirically against the specific operating point.

### 3.3 The 100 - 1000 number: empirical justification

The 100 - 1000 figure is **empirical**, not theoretical. It comes
from the following observations in the cell-free ISAC literature:

- For $M = 16$ APs, $K = 10$ users, $N_t = 4$ antennas, $P = 4$
  targets, $\epsilon_h = 0.10$, the cumulative probability of
  finding at least one feasible sample crosses 0.9 at approximately
  $L = 100$ candidates and 0.99 at approximately $L = 1000$
  candidates.
- For larger $K$ or larger $\epsilon_h$, the required $L$ grows
  roughly exponentially.
- For smaller $K$ or smaller $\epsilon_h$, $L = 20$ is typically
  sufficient.

The specific value $L = 100$ - $1000$ is reported as the empirical
"engineering good enough" range in `math_derivation{,_en}.tex` §3
(line 112). It is **not** a guarantee.

### 3.4 Worst-case approximation ratio: open problem

**Open question.** Is there a polynomial-time algorithm that, given
the SDR solution, recovers a rank-1 (P1) feasible solution with
$J^{(\ell)} \leq (1 + \rho) J_{\text{P1}}^*$ for some
$\rho = \rho(K, M, N_t, \epsilon_h)$ that depends polynomially on the
problem parameters?

**Status.** Open. The cell-free ISAC setting has additional structure
not captured by the Luo et al. multicast QCQP tightness theorem
(see `SDR_TIGHTNESS.md`). The standard "Gaussian randomization gives
a $(K/(K-1))$-approximation" bound (for multicast QCQP) does not
apply because the per-user SINR is not multicast.

**Implication.** Gaussian randomization is the engineering
recommendation, but the paper cannot claim a provable approximation
ratio. The numerical experiments in §VI of the paper
(`PAPER_OUTLINE_IEEE.md` §VI) are the evidence for the practical
performance.

---

## 4. Engineering Considerations

### 4.1 Warm starting

In practice, the first sample should be the **eigenvector**
$\vect{w}_k^{\text{eig}} = \sqrt{\lambda_{\max}(\mat{W}_k^\star)}
\vect{u}_{\max}$, which is a feasible rank-1 candidate. Subsequent
samples add diversity around this anchor.

### 4.2 Power scaling

When a sample exceeds the per-AP budget, the standard fix is to scale
$\vect{w}_k^{(\ell)}$ uniformly to satisfy
$\max_m \tr(\mat{E}_m \mat{R}_X^{(\ell)}) = P_{\max}$. This is a
**necessary** feasibility fix but breaks the optimality: the scaled
sample has strictly larger objective than the SDR optimum.

### 4.3 Sensing stream

The sensing stream $\mat{Z}$ is **not** randomized; the SDR solution
$\mat{Z}^\star$ is used directly, and only the sensing beamforming
vector $\vect{z}_m$ is extracted from the principal eigenvector of
the $m$-th diagonal block. The reason: the sensing SINR (P1-C2) and
PCRB (P1-C3) constraints are convex in $\mat{Z}$ after the lifting,
so the SDR solution $\mat{Z}^\star$ is already optimal in the
sensing subspace.

### 4.4 AP selection

AP selection is **post-hoc**: after the per-user beamforming vectors
are recovered, the top-$N_{\text{req}}$ APs by large-scale fading are
selected for each target. The reasoning: AP selection is a
discrete optimization (NP-hard), and the large-scale-fading
heuristic is the standard engineering fix.

---

## 5. Comparison with the Closed-Form Baseline (Plate IV)

The closed-form baseline (Plate IV) is a different recovery
procedure: instead of solving the SDR and randomizing, the closed-form
baseline uses a ZF (zero-forcing) communication beamformer and an
MF (matched-filter) sensing beamformer directly, without SDP. See
`PAPER_OUTLINE_IEEE.md` §V for the details.

The two recovery procedures are **complementary**:
- Gaussian randomization gives a **near-optimal** solution at the
  cost of solving an SDP and running $L$ samples.
- The closed-form baseline gives a **fast, suboptimal** solution in
  closed form.

The paper's §VI compares the two on a runtime-vs-objective plot.

---

## 6. Summary

| Claim | Status |
| --- | --- |
| Gaussian randomization converts a high-rank SDR solution to a rank-1 candidate | Algorithm 1, §2.2 |
| $L = 100$ - $1000$ candidates are empirically sufficient for the standard cell-free operating point | Empirical, §3.3 |
| The algorithm has no worst-case approximation ratio for our setting | Open, §3.4 |
| The sensing stream is not randomized, only the communication beamforming | Engineering choice, §4.3 |
| The algorithm is a runtime $O(L \cdot K (MN_t)^3)$ for sampling + feasibility | §2.3 |
| The 1.75 dB S-Procedure loss is a separate issue (see `SPROCEDURE_LOSS.md`) | Independent of SDR tightness |

---

## 7. Pointer to Companion Documents

- `math_derivation{,_en}.tex` §3 (line 112) — the SDR step and the
  one-line Gaussian randomization mention.
- `SDR_TIGHTNESS.md` — companion to this document. Discusses why the
  SDR is not generally tight here.
- `SPROCEDURE_LOSS.md` — companion. Discusses the 1.75 dB S-Procedure
  loss (a separate effect from SDR tightness).
- `CONVEXIFICATION_CHAIN.md` §3.2 — the SDR step narrative.
- `PAPER_OUTLINE_IEEE.md` §IV — the engineering feasibility plate
  where this recovery algorithm is described in the paper.
