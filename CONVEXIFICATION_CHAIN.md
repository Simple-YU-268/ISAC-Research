# Convexification Chain

**Plate**: II (Mathematical Reformulation) — Steps 1 through 7
**Source of math**: `math_derivation{,_en}.tex` §3 (逐步凸化) and §6 (最终凸 SDP 问题)
**Master reference**: `MASTER_CONSTRAINTS.md`
**Status**: v3.0 (2026-06-30). Rewritten under the unified mathematical setting
established by the 2026-06-30 audit. Supersedes the previous
`CONVEXIFICATION_CHAIN.md` (deleted in 19ff970).

---

## 1. Why This Document Exists

The original (P1) is a non-convex mixed-integer nonlinear program. The
seven convexification steps transform it into a convex SDP (P3) that can
be solved by an interior-point method. This document is the prose
companion to `math_derivation{,_en}.tex` §3.

**Per-step guarantees.** Every step states:

1. **What non-convex source it removes** (NC1–NC6, see §2).
2. **What transformation it applies** (lifting, SDR, S-Procedure, etc.).
3. **Whether the transformation is strict equivalence, tight relaxation,
   or conservative approximation** — and what that means for the optimal
   value of the problem.
4. **Where the result lives in `math_derivation{,_en}.tex`** so the
   reader can find the full derivation.

**Original-constraint → convex-constraint correspondence.** The final
table at §4 maps each (P1-Cx) to its (P3-Cx) form.

---

## 2. The Six Non-Convexity Sources (NC1–NC6)

These are the six sources of non-convexity in the original problem. Each
step in the chain targets one or more of them.

| ID | Source | Affected constraints | Why it is non-convex |
| --- | --- | --- | --- |
| **NC1** | SINR fractional structure | (P1-C1) comm, (P1-C2) sensing | The set $\{(\vect{w}, t) : |\vect{x}^H \vect{w}|^2 \geq t \cdot (\sum_{j \neq k} |\vect{x}^H \vect{w}_j|^2 + \sigma^2)\}$ is not closed under convex combinations. |
| **NC2** | Semi-infinite robust constraints | (P1-C1) (worst-case SINR over $\|\Delta\vect{h}\| \leq \epsilon_h$) | The universal quantifier $\forall \Delta\vect{h} \in \mathcal{B}_\epsilon$ is an infinite family of convex constraints; the resulting set is not convex. |
| **NC3** | Binary AP selection | (P1-C4) | $\{0, 1\}$ is a discrete subset of $\mathbb{R}$, not an affine subspace. |
| **NC4** | Implicit rank-1 | (P2-C7) (after lifting to (P2)) | The rank-1 set $\{(\mat{W}, \vect{w}) : \mat{W} = \vect{w}\vect{w}^H\}$ is a non-affine algebraic variety. |
| **NC5** | Beam-sensing bilinear coupling | (P1-C5) (per-AP power: $\|\vect{w}_{m,k}\|^2$ and $\tr(\mat{Z}_m)$ in the same constraint) | The product of two PSD variables is non-convex. |
| **NC6** | Matrix inverse in PCRB | (P1-C3) | $\tr(\mat{J}_p^{-1})$ is a convex function of $\mat{J}_p^{-1}$ but $\mat{J}_p^{-1}$ is the inverse of an affine function of $\mat{R}_X$, which is not convex in $\mat{R}_X$. |

The mathematical criterion for convexity used throughout the chain is
recalled from `math_derivation{,_en}.tex` §3:

> A set $\mathcal{C}$ is convex iff $\forall \vect{x}, \vect{y} \in \mathcal{C}, \forall \theta \in [0, 1] : \theta \vect{x} + (1 - \theta)\vect{y} \in \mathcal{C}$.

The six sources above each violate this criterion in a specific way. Each
convexification step restores a portion of the criterion.

---

## 3. The Seven Convexification Steps

### 3.1 Step 1 — Lifting: (P1) → (P2)

**Removes.** No non-convexity directly. Lays the ground for SDR (Step 2).

**Transformation.** Replace per-AP sensing covariance $\{\mat{Z}_m\}$ with
joint block-diagonal $\mat{Z} \in \mathbb{H}_+^{MN_t}$, and per-user
beamforming vectors $\{\vect{w}_{m,k}\}$ with joint covariance
$\mat{W}_k = \vect{w}_k \vect{w}_k^H \in \mathbb{H}_+^{MN_t}$. Add the
rank-1 constraint (P2-C7).

**Equivalence.** **Strictly equivalent** to (P1) by the bijection
$\vect{w}_k \leftrightarrow \mat{W}_k = \vect{w}_k \vect{w}_k^H$. No
relaxation.

**Source.** `math_derivation{,_en}.tex` §2 (协方差提升形式).

### 3.2 Step 2 — Semidefinite Relaxation (SDR): drop (P2-C7)

**Removes.** NC4 (rank-1).

**Transformation.** Drop the rank-1 constraint (P2-C7); keep only PSD
$\mat{W}_k \succeq \mat{0}$. Feasible set grows from the rank-1 manifold
$\mathcal{F}_{\text{rank-1}}$ to the semidefinite cone
$\mathcal{F}_{\text{SDR}} = \{\mat{W} \succeq \mat{0}\}$.

**Equivalence.** **Tight relaxation** for $K \leq 2$ (per Luo et al.
2010 multicast QCQP), but **no general tightness theorem** applies for
our case:

- The multicast $G_k \leq 2$ condition (Luo et al.) does not transfer to
  per-user SINR with cell-free cooperation and robust uncertainty.
- The robust MISO $N_t \leq 2$ condition (Huang & Palomar 2010) does not
  apply: our dimension is $MN_t$, and the problem has both
  communication (per-user) and sensing (per-target) constraints.

**Practical implication.** $P_{\text{SDR}}^* \leq P_{\text{P2}}^*
= P_{\text{P1}}^*$ in general. When $\rank(\mat{W}_k^*) > 1$ at the SDR
solution, **Gaussian randomization** with $L = 100$–$1000$ candidates is
the engineering fix for rank-1 recovery. See
`PAPER_OUTLINE_IEEE.md` §IV (Plate III) for the algorithm.

**Source.** `math_derivation{,_en}.tex` §3 first paragraph (Step 1 注释).

### 3.3 Step 3 — S-Procedure for worst-case comm SINR: (P2-C1) → LMI

**Removes.** NC2 (semi-infinite) and NC1 (SINR fraction) — both at once
in the comm-side constraint.

**Transformation.** The worst-case SINR is rewritten as a quadratic
inequality in $\Delta\vect{h}_k$ that must hold on a ball. The
S-Procedure (Frisk 1995, Beck 1997) replaces the infinite family with a
single LMI at the cost of a non-negative slack variable $\mu_k$:

$$
\exists \mu_k \geq 0 : \begin{bmatrix}
\mat{A}_k + \mu_k \mat{I} & \mat{A}_k \hat{\vect{h}}_k \\
\hat{\vect{h}}_k^H \mat{A}_k & \hat{\vect{h}}_k^H \mat{A}_k \hat{\vect{h}}_k - \sigma_c^2 - \mu_k \epsilon_h^2 \|\hat{\vect{h}}_k\|^2
\end{bmatrix} \succeq \mat{0}
$$

**Equivalence.** **Strictly equivalent** (the LMI holds if and only if
the original worst-case quadratic inequality holds, by Lemma 1 in the
source — the 2-quadratic S-Procedure on the ball $\{\Delta\vect{h} :
\|\Delta\vect{h}\|_2 \leq \epsilon_h \|\hat{\vect{h}}_k\|_2\}$ requires
no Slater-type additional assumption; the (S3.1) form above uses the
ball-outside convention ($\|\Delta\vect{h}\|_2 \geq \epsilon_h \|\hat{\vect{h}}_k\|
$ as the $f_1$ description), giving if-and-only-if via Boyd and
Vandenberghe, 2004, §4.3, Theorem 4.1).

**Source.** `math_derivation.tex` §2-C, the (S3.1) block, Lemma 1,
and the surrounding discussion of ball-outside vs. ball-inside
convention.

### 3.4 Step 4 — Fraction linearization: cross-multiply the comm-SINR

**Removes.** NC1 on the comm side (already partially addressed by
Step 3's S-Procedure; this step formalizes the cross-multiplication).

**Transformation.** Provided the denominator $\sum_{j \neq k}
|\vect{h}_k^H \vect{w}_j|^2 + \sigma_c^2$ is positive (guaranteed by
$\sigma_c^2 > 0$), the SINR inequality is equivalent to

$$
\tr(\hat{\mat{H}}_k \mat{W}_k) - \gamma_k \sum_{j \neq k} \tr(\hat{\mat{H}}_k \mat{W}_j) \geq \gamma_k \sigma_c^2
$$

**Equivalence.** **Strictly equivalent** (denominator positivity
guaranteed).

**Source.** `math_derivation{,_en}.tex` §3 (the (S2.1) derivation, just
above (S3.1)).

### 3.5 Step 5a — PCRB affine expansion: (P2-C3) → linear

**Removes.** NC6 (matrix inverse).

**Transformation.** The data Fisher information matrix $\mat{J}_p^{\text{data}}$
is affine in $\mat{R}_X$ because $\nabla_{\boldsymbol{\theta}_p} \vect{g}_p$ is
treated as known at the current time slot (Assumption 1 in
`math_derivation{,_en}.tex`). By the cyclic property of trace,

$$
\tr(\mat{J}_p^{\text{data}}) = \tr(\mat{F}_p \mat{R}_X), \quad
\mat{F}_p = \frac{2}{\sigma_s^2} \Real\{\nabla_{\boldsymbol{\theta}_p} \vect{g}_p^H \cdot \nabla_{\boldsymbol{\theta}_p}^H \vect{g}_p\} \in \mathbb{H}_+^{MN_t}
$$

**Equivalence.** **Strictly equivalent** under Assumption 1
(current-slot $\nabla \vect{g}_p$ is a known constant).

**Source.** `math_derivation{,_en}.tex` §4 (Step 5a, equation (S4.1a)
and (S4.1b) and the surrounding proposition).

### 3.6 Step 5b — Sensing SINR linearization: (P2-C2) → linear

**Removes.** NC1 on the sensing side.

**Transformation.** Because $\mat{Z} \succeq \mat{0}$ implies
$\vect{g}_p^H \mat{Z} \vect{g}_p \geq 0$ (the numerator is non-negative),
the cross-multiplication $\sigma_s^2 > 0$ gives the linear form

$$
\tr(\vect{g}_p \vect{g}_p^H \mat{Z}) \geq \gamma_S^{\text{PoD}} \sigma_s^2
$$

**Equivalence.** **Strictly equivalent.**

**Source.** `math_derivation{,_en}.tex` §4 (Step 5b, equation (S4.2)).

### 3.7 Step 6 — Per-AP power is already linear: (P2-C4) → linear

**Removes.** NC5 (bilinear coupling).

**Transformation.** The per-AP power constraint couples
$\|\vect{w}_{m,k}\|^2$ and $\tr(\mat{Z}_m)$ in (P1-C5). In the lifted
form (P2-C4), the same coupling is expressed as
$\tr(\mat{E}_m \mat{R}_X) \leq P_{\max}$, where $\mat{E}_m$ is the
per-AP antenna selector and $\mat{R}_X = \sum_k \mat{W}_k + \mat{Z}$.

**Equivalence.** **Strictly equivalent** (the lifting replaces the
bilinear product by a linear expression in $\mat{R}_X$).

**Per-AP vs sum-power.** The repo's strict preference is **per-AP peak
power** $P_{\max}$ (constraint (P3-C4)), not sum-power. This is
documented in `MASTER_CONSTRAINTS.md` §6 (USER PROFILE preferences).

**Source.** `math_derivation{,_en}.tex` §4 (Step 5b's last paragraph and
equation (S5.1)).

### 3.8 Step 7 — AP selection heuristic: (P2-C8) → fixed-set subproblem

**Removes.** NC3 (binary AP selection).

**Transformation.** Two-step decomposition:

1. **Outer (discrete).** Sort APs by large-scale fading $\text{PL}(d_{m,p})$
   to target $p$ and select the top $N_{\text{req}}$ APs. This determines
   the service set $\mathcal{M}_p$.
2. **Inner (continuous).** Solve the convex SDP (P3) on the fixed
   service set.

**Equivalence.** **Heuristic** — no optimality guarantee. The outer
selection may not be the true optimum (the AP selection subproblem is
NP-hard, a $K$-medoid variant). The inner SDP on a fixed set is
convex and tight. The two-step combination can produce either a
lower bound or an incomparable objective value depending on the actual
AP set chosen.

**Source.** `math_derivation{,_en}.tex` §4 (Step 7 paragraph) and §6
proof (Step 7 无理论最优性保证).

---

## 4. The Final Convex SDP (P3)

The seven steps together produce (P3). For the full statement of (P3),
see `math_derivation{,_en}.tex` §6 and `MASTER_CONSTRAINTS.md` §2. The
constraint-by-constraint correspondence to (P1) is in
`MASTER_CONSTRAINTS.md` §4.

**Summary.** (P3) is a standard convex SDP. All constraints are LMIs
or linear inequalities, the objective is linear, and the problem can be
solved by an interior-point method in polynomial time. SDR (Step 2)
makes the value a lower bound on (P1). S-Procedure (Step 3) makes the
robust feasibility set an upper bound on the true worst-case set.
AP selection (Step 7) is a heuristic. Everything else is strict
equivalence.

**Plate II is closed at (P3).** Plate III (`PAPER_OUTLINE_IEEE.md` §IV)
handles rank-one recovery; Plate IV (`PAPER_OUTLINE_IEEE.md` §V) provides
the closed-form baseline.

---

## 5. Tightness and Complexity

### 5.1 SDR tightness (Step 2)

For our setting (per-user SINR + cell-free cooperation + robust
uncertainty), **no general SDR tightness theorem applies**. The
multicast $G_k \leq 2$ condition and the robust MISO $N_t \leq 2$
condition do not transfer. The SDR solution may have
$\rank(\mat{W}_k^*) > 1$, in which case Gaussian randomization with
$L = 100$–$1000$ candidates is the engineering fix. No worst-case
tightness bound exists.

### 5.2 S-Procedure loss (Step 3)

The S-Procedure is a sufficient condition, not necessary. The loss is
approximately 1.75 dB of power margin for $\epsilon_h = 0.10$
($\eta_h \approx 0.669$).

### 5.3 AP selection (Step 7)

The two-step decomposition has no theoretical optimality guarantee. The
solved objective can be either a lower bound on (P1) (when the
large-scale-fading-sort picks a service set that includes a true-optimum
AP) or an incomparable value (when it does not).

### 5.4 Computational complexity

The worst-case interior-point complexity is
$O\big((K+1)^3 (MN_t)^6 \cdot (K + P + M)\big)$, exact. This comes
from the SDP variable dimension $n = O(K (MN_t)^2)$ and the LMI
constraint maximum size $MN_t + 1$. The simplified order is
$O((K M N_t^2)^{3.5})$.

**Source.** `math_derivation{,_en}.tex` §6 (theorem and proof).

---

## 6. Pointer to Companion Documents

- `math_derivation{,_en}.tex` — the LaTeX-compilable source of record. The 7-step chain is in §3. The (P3) form is in §6. The theorem + proof of convexity and complexity is in §6.
- `MASTER_CONSTRAINTS.md` — the master formula reference. The per-constraint reference card is in §2. The (P1)→(P3) correspondence table is in §4. The symbol table is in §5. The strict-preference notes (per-AP power, S-Procedure exact form, channel normalization) are in §6.
- `PROBLEM_FORMULATION.md` — the companion to this file. Covers Plate I (system model, signal model, (P1), (P2)) and points to (P3) for the convexification steps.
- `PAPER_OUTLINE_IEEE.md` — the 4-plate IMRaD outline. Drives the structure of this file.

If a conflict arises between this file and `MASTER_CONSTRAINTS.md` or
`math_derivation{,_en}.tex`, the latter two win. This file is a narrative
companion, not the source of record for the math.
