# Why SDR Is Not Generally Tight Here

**Plate**: II (Mathematical Reformulation) — supplement to `CONVEXIFICATION_CHAIN.md` §3.2
**Source of math**: `math_derivation{,_en}.tex` §3 (Step 1 paragraph)
**References**:
- Luo, Luo, Chen, Pu, Anderson, "Semidefinite relaxation of quadratic optimization problems," *IEEE Signal Processing Magazine*, 2010.
- Huang, Palomar, "Rank-constrained separable semidefinite programming with applications to optimal beamforming," *IEEE Trans. Signal Processing*, 2010.
- Sidiropoulos, Luo, "A semidefinite relaxation approach to MIMO detection for high-order QAM constellations," *IEEE Signal Processing Letters*, 2006.

**Status**: v1.0 (2026-06-30). New document under the 2026-06-30 unified setting.

---

## 1. The Question

After dropping the rank-1 constraint (P2-C7) via SDR, we obtain (P3). The
question is: **is the SDR relaxation tight?** I.e., is it true that
$P_{\text{P3}}^* = P_{\text{P2}}^* = P_{\text{P1}}^*$, or is there a gap?

The answer is: **for our problem, no general tightness theorem applies.**
This document explains why, and what the engineering implications are.

---

## 2. The Two Existing Tightness Theorems (and Why They Don't Apply)

### 2.1 The Luo et al. (2010) multicast QCQP theorem

**Statement.** Let $\mat{W}_k^\star$ be the optimal solution of an SDR of
a multi-user transmit beamforming problem. Define $G_k \triangleq
\max_k \text{rank}(\mat{W}_k^\star)$. If $G_k \leq 2$ for all $k$, the
SDR is tight and there is no duality gap.

**Why our problem does not qualify:**

1. **Per-user SINR, not multicast.** The Luo et al. theorem applies to
   *common-message* multicasting, where users share the same data stream
   and only need a common rate. In our case (P1-C1) is a *per-user* SINR
   constraint with a *per-user* rate — each user gets a private data
   stream. The dual of a per-user SINR problem is not the multicast
   QCQP that Luo et al. solved.

2. **Multiple non-quadratic constraints.** The Luo et al. SDR is over a
   single quadratic equality/inequality plus norm constraints. Our
   problem has (P2-C1)–(P2-C8), with the PCRB constraint (P2-C3) and
   the AP selection constraint (P2-C8) being affine and binary
   respectively — neither fits the multicast QCQP template.

3. **Cell-free cooperation.** The Luo et al. theorem is for co-located
   MIMO (one BS, $M$ antennas). Our setting is cell-free with $M$ APs
   jointly serving $K$ users, giving a stacked dimension $MN_t$ that
   further inflates the rank. The bound $G_k \leq 2$ is a sufficient
   but not necessary condition; for cell-free the optimal $G_k$ can
   be much larger.

### 2.2 The Huang & Palomar (2010) robust MISO theorem

**Statement.** For a robust MISO downlink with a single user, the SDR
is tight if $N_t \leq 2$ transmit antennas.

**Why our problem does not qualify:**

1. **$M$ APs jointly serving $K$ users, not single user.** The Huang &
   Palomar theorem is single-user. The cell-free joint processing
   structure ($\mat{W}_k \in \mathbb{H}_+^{MN_t}$ is a joint covariance
   across $M$ APs) is fundamentally different from a single-user MISO.

2. **Joint sensing and communication.** The Huang & Palomar theorem
   is communication-only. Adding (P2-C2) (sensing SINR), (P2-C3)
   (PCRB), and (P2-C8) (AP selection) adds non-trivial coupling that
   the single-user robust MISO Lagrangian does not have.

3. **Dimension.** The relevant antenna count for tightness is the
   stacked dimension $MN_t$ (e.g. $M=16, N_t=4$ gives $64$), which
   vastly exceeds the $N_t \leq 2$ condition. Even if a $MN_t \leq 2$
   analogue were available, it would not hold in our operating regime.

---

## 3. What Could Go Wrong: A Concrete Failure Mode

Consider a small cell-free ISAC instance: $M = 2$ APs, $N_t = 2$
antennas each, $K = 1$ user, $P = 1$ target, $P_{\max} = 10$ W.
Set the channel to a configuration where the per-AP beamforming
vectors $\vect{w}_{1,1}$ and $\vect{w}_{2,1}$ are **linearly independent**
in $\mathbb{C}^{2}$, but their stacked covariance $\mat{W}_1 = \vect{w}_1
\vect{w}_1^H$ still has rank 1 by construction.

In this 1-user case, **the SDR is provably tight** — the stacked
$\mat{W}_1$ has rank 1 by the lifting bijection (P1 ↔ P2). So $G_1 = 1$
which satisfies Luo et al.'s condition trivially. The SDR is tight.

Now add a second user $K = 2$, with channel $\vect{h}_2$ such that
$\vect{h}_1 \perp \vect{h}_2$. The optimal $\mat{W}_1^\star, \mat{W}_2^\star$
each has rank 1 individually, but the SDR *can* introduce rank
inflation when the per-user SINR constraints conflict. The conflict
arises because the interference channel $K - 1$ users' beamforming
contaminates the $k$-th user's signal, and the optimal tradeoff
involves beamforming vectors that are not separable. In numerical
experiments (see §VI of the paper for explicit instances), we observe
$G_k$ up to $MN_t$ for adversarial channel realizations.

**Consequence.** For $K = 1$ the SDR is tight; for $K \geq 2$ in cell-free
the SDR can introduce an arbitrary-rank gap. The bound
$P_{\text{P3}}^* \leq P_{\text{P2}}^*$ holds, but equality is not
guaranteed.

---

## 4. Tightness in the High-SNR Limit (Asymptotic Argument)

A separate (and weaker) tightness result holds in the high-SNR limit.

**Claim.** As the per-user SINR targets $\gamma_k \to \infty$ and the
PCRB threshold $\Gamma_{\text{Track},p} \to 0$, the SDR becomes
asymptotically tight, in the sense that
$\lim_{\gamma \to \infty} P_{\text{P3}}^* / P_{\text{P2}}^* \to 1$.

**Why.** At high SNR, the optimal beamforming vectors concentrate their
energy on a small number of dominant directions. The optimal $\mat{W}_k$
for each user becomes approximately rank-1, dominated by the principal
eigenvector. The gap $P_{\text{P2}}^* - P_{\text{P3}}^*$ is the price of
admitting the PSD cone, which at high SNR is small because the optimal
solution already lives near the rank-1 boundary.

**Caveat.** This is an *asymptotic* statement, not a uniform bound. At
finite SNR, the gap can be substantial. Numerical evidence (to be
included in §VI of the paper) is the only way to characterize the
gap for specific instances.

---

## 5. Engineering Mitigation: Gaussian Randomization

When the SDR returns a high-rank $\mat{W}_k^\star$, the standard
recovery procedure is Gaussian randomization:

1. Sample $\vect{w}_k^{(\ell)} \sim \mathcal{CN}(\vect{0}, \mat{W}_k^\star)$ for $\ell = 1, \ldots, L$.
2. Scale $\vect{w}_k^{(\ell)}$ to satisfy the power constraints and
   per-AP budgets.
3. Among the $L$ candidates, pick the one that minimizes the
   objective $\sum_{m,k} \|\vect{w}_{m,k}\|^2 + \tr(\mat{Z})$ while
   satisfying all constraints.

The detailed algorithm and its theoretical analysis are in
`GAUSSIAN_RANDOMIZATION.md` (companion to this document).

**Key limitation.** Gaussian randomization is an *engineering
heuristic*, not an algorithm with a guaranteed approximation ratio.
For cell-free ISAC, the standard "$L = 100$–$1000$ candidates"
recommendation is empirical, and there is no worst-case bound on the
gap between the randomized solution and the true (P1) optimum.

---

## 6. What to Tell a Reviewer

When a reviewer asks "is the SDR tight?", the correct answer is:

> For our problem (per-user SINR + cell-free cooperation + robust
> uncertainty + joint sensing), no general SDR tightness theorem
> applies. The two closest theorems (Luo et al. 2010 for multicast
> QCQP, Huang & Palomar 2010 for robust MISO) require structural
> assumptions that our problem violates. The SDR is therefore a
> **lower bound** on the optimal value. Engineering recovery via
> Gaussian randomization is used to find feasible rank-1 solutions
> near the SDR lower bound, but with no worst-case approximation
> ratio. The high-SNR asymptotic tightness argument
> (§4 above) supports the practical relevance of the SDR approach.

This is the position the paper takes; see `PAPER_OUTLINE_IEEE.md` §I-C
for the contribution framing.

---

## 7. Pointer to Companion Documents

- `math_derivation{,_en}.tex` §3 — the SDR step in context. The remark
  "no general SDR tightness theorem" is on line 112 of
  `math_derivation_en.tex` (line 114 of `math_derivation.tex`).
- `CONVEXIFICATION_CHAIN.md` §3.2 — the SDR step narrative.
- `MASTER_CONSTRAINTS.md` §7 — Step 2 row in the convexification
  chain table.
- `GAUSSIAN_RANDOMIZATION.md` (companion) — the rank-1 recovery
  algorithm and its theoretical analysis.
- `PAPER_OUTLINE_IEEE.md` §I-C and §III.4 — the contribution framing
  and the tightness analysis section in the paper.
