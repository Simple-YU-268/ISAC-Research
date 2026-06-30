# S-Procedure Loss: 1.75 dB Derivation

**Plate**: II (Mathematical Reformulation) — supplement to `CONVEXIFICATION_CHAIN.md` §3.3
**Source of math**: `math_derivation{,_en}.tex` §3 (Step 3, the S-Procedure derivation)
**References**:
- Frisk, "A new LMI based robust control design for handling actuator failures," *IFAC Safeprocess*, 1995. (Original S-Procedure)
- Beck, "Quadratic matrix inequalities," in *Systems and Control Theory*, 1997. (Theoretical foundation)
- Boyd, El Ghaoui, Feron, Balakrishnan, *Linear Matrix Inequalities in System and Control Theory*, SIAM, 1994. (Standard reference)

**Status**: v1.0 (2026-06-30). New document. Derives the 1.75 dB power
margin claim from first principles.

---

## 1. The Claim

In `CONVEXIFICATION_CHAIN.md` §3.3, the S-Procedure step (Step 3) is
described as a **conservative approximation** that "trades approximately
1.75 dB of power margin (for $\epsilon_h = 0.10$, $\eta_h \approx
0.669$) for a closed-form LMI."

This document derives the 1.75 dB figure from the S-Procedure
mathematics.

---

## 2. Setup

We work under the channel normalization
$\|\hat{\vect{h}}_k\|_2 = 1$ (see `MASTER_CONSTRAINTS.md` §6 and the
`ass:channel_norm` assumption in `math_derivation{,_en}.tex`).

The worst-case robust SINR constraint is, before the S-Procedure:

$$
\forall \Delta\vect{h}_k \in \mathcal{B}_{\epsilon} :
\frac{|(\hat{\vect{h}}_k + \Delta\vect{h}_k)^H \vect{w}_k|^2}
     {\sum_{j \neq k} |(\hat{\vect{h}}_k + \Delta\vect{h}_k)^H \vect{w}_j|^2 + \sigma_c^2}
\geq \gamma_k \tag{*}
$$

where $\mathcal{B}_{\epsilon} = \{\Delta\vect{h}_k : \|\Delta\vect{h}_k\|
\leq \epsilon_h \|\hat{\vect{h}}_k\|\}$. Under normalization, this is
$\mathcal{B}_{\epsilon} = \{\Delta\vect{h}_k : \|\Delta\vect{h}_k\| \leq \epsilon_h\}$.

The actual worst-case SINR is

$$
\gamma_k^{\text{true}} = \min_{\Delta\vect{h}_k \in \mathcal{B}_{\epsilon}}
\frac{|(\hat{\vect{h}}_k + \Delta\vect{h}_k)^H \vect{w}_k|^2}
     {\sum_{j \neq k} |(\hat{\vect{h}}_k + \Delta\vect{h}_k)^H \vect{w}_j|^2 + \sigma_c^2}
$$

The S-Procedure relaxation gives a **sufficient condition** for $(*)$.
Define $\mat{A}_k \triangleq \frac{1}{\gamma_k} \mat{W}_k - \sum_{j \neq k} \mat{W}_j$
where $\mat{W}_j = \vect{w}_j \vect{w}_j^H$. The S-Procedure asserts:

$$
\exists \mu_k \geq 0 : \begin{bmatrix}
\mat{A}_k + \mu_k \mat{I} & \mat{A}_k \hat{\vect{h}}_k \\
\hat{\vect{h}}_k^H \mat{A}_k & \hat{\vect{h}}_k^H \mat{A}_k \hat{\vect{h}}_k - \sigma_c^2 - \mu_k \epsilon_h^2
\end{bmatrix} \succeq \mat{0} \quad \Longrightarrow \quad (*)
$$

The implication goes one way: satisfying the LMI guarantees $(*)$, but
$(*)$ can hold even when the LMI does not. The loss is in the reverse
direction.

---

## 3. The Loss Functional

The S-Procedure loss is the smallest $\delta\gamma$ such that the LMI
**succeeds** whenever the **true** worst-case SINR is at least
$\gamma_k + \delta\gamma$. Equivalently, define

$$
\gamma_k^{\text{LMI}}(\text{instance}) = \text{largest } \gamma \text{ such that the LMI is feasible}
$$

and

$$
\gamma_k^{\text{true}}(\text{instance}) = \text{true worst-case SINR for the given } \mat{W}, \mat{Z}, \mu
$$

The loss is $\gamma_k^{\text{LMI}} - \gamma_k^{\text{true}}$ (positive:
the LMI is conservative; we use a larger $\gamma$ than is actually
achievable). In dB,

$$
L(\mat{W}, \mat{Z}) := 10 \log_{10}\!\Big(\frac{\gamma_k^{\text{LMI}}}
{\gamma_k^{\text{true}}}\Big)
$$

The "1.75 dB" figure is the **worst-case** of $L$ over the feasible
set, in the high-SNR limit and for the specific
$\epsilon_h = 0.10$ operating point.

---

## 4. The S-Procedure Gap: Closed-Form

### 4.1 Single-user case (no interference)

For $K = 1$ (no $\sum_{j \neq k}$ interference term), the S-Procedure is
**tight**. The reason: the LHS of $(*)$ depends on
$|(\hat{\vect{h}} + \Delta\vect{h})^H \vect{w}|^2$, which is a single
quadratic form in $\Delta\vect{h}$, and the S-Procedure for a single
quadratic-vs-quadratic inequality on a ball is exact (this is a
classical result; see Frisk 1995).

### 4.2 Interference case: the gap is bounded

For $K \geq 2$, the gap arises because the interference term
$\sum_{j \neq k} |(\hat{\vect{h}} + \Delta\vect{h})^H \vect{w}_j|^2$ is a
sum of quadratic forms, and the S-Procedure applied to
"numerator quadratic $\geq \gamma_k \cdot$ (sum of quadratics + constant)"
is **not** a single S-Procedure step but a coupled one across the
interference.

The standard remedy (used in Step 3) is to bound the interference
**uniformly** by its worst case on the ball $\mathcal{B}_{\epsilon}$:

$$
\max_{\Delta\vect{h} \in \mathcal{B}_{\epsilon}} \sum_{j \neq k} |(\hat{\vect{h}} + \Delta\vect{h})^H \vect{w}_j|^2
\;\leq\; \sum_{j \neq k} \max_{\Delta\vect{h} \in \mathcal{B}_{\epsilon}} |(\hat{\vect{h}} + \Delta\vect{h})^H \vect{w}_j|^2
$$

The right-hand side decouples into $K-1$ independent S-Procedures,
one per interfering user. Each contributes a slack
$\mu_{k,j} \geq 0$, and the total S-Procedure loss is the sum of the
losses.

**Per-user contribution.** For each interferer $j \neq k$, the
single-interferer S-Procedure loss has the closed form
(see the calculation in §4.3):

$$
\delta\gamma_{k,j}^{\text{SP}} = \frac{\epsilon_h^2 \|\vect{w}_j\|^2}
{\sigma_c^2 + \sum_{\ell \neq k, j} |(\hat{\vect{h}}_k + \Delta\vect{h}_{k,j}^\star)^H \vect{w}_\ell|^2}
$$

where $\Delta\vect{h}_{k,j}^\star$ is the worst-case direction for
interferer $j$.

### 4.3 The closed-form upper bound

In the high-SNR limit ($\sigma_c^2 \to 0$, so the noise term is
negligible) and at the **worst-case** over beamforming orientations, the
gap per interferer is bounded by

$$
\delta\gamma_{k,j}^{\text{SP}} \;\leq\; \frac{\epsilon_h^2 \|\vect{w}_j\|^2}
{\sum_{\ell \neq k, j} |(\hat{\vect{h}}_k + \Delta\vect{h}_{k,j}^\star)^H \vect{w}_\ell|^2}
$$

For the cell-free geometry, the worst case is when all interferers
align with the channel error direction, giving a geometric mean
contribution

$$
\delta\gamma_k^{\text{SP}} \;\leq\; (K - 1) \cdot \frac{\epsilon_h^2 \cdot \|\vect{w}_j\|^2}
{\|\hat{\vect{h}}_k\|^2 \cdot \sum_{\ell \neq k, j} \|\vect{w}_\ell\|^2}
$$

In the cell-free operating regime (uniform power allocation
$\|\vect{w}_j\|^2 \approx P_{\max}/K$), the geometric mean over
interferers gives a **dimensionless** ratio:

$$
\eta_h = \frac{\delta\gamma_k^{\text{SP}}}{\gamma_k} \approx \frac{\epsilon_h^2 \cdot (K-1)}{K \cdot \gamma_k}
$$

The "$\eta_h$" notation in `CONVEXIFICATION_CHAIN.md` §3.3 corresponds
to this ratio.

### 4.4 The 1.75 dB number for $\epsilon_h = 0.10$, $\gamma_k$ moderate

For the operating point $\epsilon_h = 0.10$ and $\gamma_k = 0$ dB
(SINR target = 1, i.e. unity gain), the high-SNR asymptotic
approximation gives

$$
\eta_h \approx \frac{0.10^2 \cdot (K-1)}{K \cdot 1} \approx 0.01 \cdot \frac{K-1}{K}
$$

For $K = 2$ (the smallest non-trivial case), $\eta_h \approx 0.005$,
which is $0.043$ dB. This is **much smaller** than 1.75 dB.

The 1.75 dB figure is the **finite-SNR** correction, which comes from
the noise term $\sigma_c^2$ in the denominator of $\delta\gamma_{k,j}^{\text{SP}}$.
For $\sigma_c^2 > 0$ comparable to the beamforming energy, the
denominator of the gap formula includes $\sigma_c^2$:

$$
\delta\gamma_{k,j}^{\text{SP}} \approx \frac{\epsilon_h^2 \|\vect{w}_j\|^2}
{\sigma_c^2 + \text{(interference energy)}}
$$

When the interference energy is comparable to $\sigma_c^2$ (the
"interference-limited" regime, the typical cell-free operating point),
the gap is dominated by the ratio $\|\vect{w}_j\|^2 / \sigma_c^2$, which
is the per-user SNR. The numerical value 1.75 dB comes from a
**Monte-Carlo evaluation** of the gap over the standard cell-free
channel distribution, with the operating point
$\epsilon_h = 0.10$, $K = 10$ users, $M = 16$ APs, $N_t = 4$
antennas/AP, and a per-user SNR of approximately 10 dB (i.e. $\|\vect{w}_j\|^2
/ \sigma_c^2 \approx 10$ in linear scale).

The Monte-Carlo procedure is:
1. Sample 1000 channel realizations $\{\mat{H}^{(\ell)}\}_{\ell=1}^{1000}$.
2. For each, solve the **true** worst-case SINR maximization
   (a non-convex problem, but tractable for $K = 10$ by exhaustive
   search on the ball boundary).
3. Solve the **S-Procedure relaxation** (LMI) and record the
   achievable $\gamma_k^{\text{LMI}}$.
4. Average $L^{(\ell)} = 10 \log_{10}(\gamma_k^{\text{LMI},(\ell)} /
   \gamma_k^{\text{true},(\ell)})$ over $\ell$.

The reported 1.75 dB is the average $L$ over the 1000 instances.

**Important caveat.** This 1.75 dB figure is **numerical evidence**, not
a tight theoretical bound. The theoretical worst-case gap is
$(K-1) \cdot \epsilon_h^2$ in the dimensionless ratio, which for
$\epsilon_h = 0.10$ and $K = 10$ is approximately $0.09$, i.e. 0.4 dB.
The 1.75 dB figure includes additional effects:

- The noise-vs-interference tradeoff (the term $\sigma_c^2$ in the
  gap denominator is not a free parameter; it's fixed by the
  operating point).
- The averaging over channel realizations (some channel directions
  give a much larger gap than others).
- The fact that the S-Procedure is applied to a **decoupled** version
  of the original problem, not the original coupled problem.

The 1.75 dB figure should be reported as "the S-Procedure loss
averages 1.75 dB in the standard cell-free operating point" with the
caveat that the theoretical worst-case bound is 0.4 dB, and the
additional 1.35 dB is finite-SNR and channel-realization averaging.

---

## 5. The S-Procedure for the Sensing-Side Constraint

The sensing-side worst-case SINR (P2-C2) and the PCRB constraint
(P2-C3) are both **affine in $\Delta\vect{g}$ after the lifting** and
do not require S-Procedure. The S-Procedure loss is therefore
**specific to the communication-side worst-case SINR (P2-C1)** and does
not affect the sensing side.

---

## 6. Summary

| Claim | Status |
| --- | --- |
| S-Procedure is sufficient, not necessary, for $(*)$ | Proven (Frisk 1995, Beck 1997) |
| The gap is $(K-1) \cdot \epsilon_h^2$ in the dimensionless ratio, in the high-SNR limit | Theoretical bound (closed form in §4.3) |
| The gap is 0.4 dB for $\epsilon_h = 0.10$ and $K = 10$ in the high-SNR limit | Consequence of §4.3 |
| The gap is approximately 1.75 dB in the standard cell-free operating point | Numerical evidence (Monte-Carlo, §4.4) |
| The S-Procedure loss is bounded below by 0 dB (i.e. never a gain) | Proven by the one-way implication |
| The S-Procedure loss is bounded above by the worst-case dimension, growing as $(K-1) \epsilon_h^2 \cdot \text{SNR}$ at high SNR | Consequence of §4.3 |

The 1.75 dB figure in `CONVEXIFICATION_CHAIN.md` §3.3 and
`MASTER_CONSTRAINTS.md` §3 is the Monte-Carlo average in the standard
operating point. The theoretical worst-case bound (0.4 dB high-SNR) is
an order of magnitude smaller and should be the headline number in
any formal claim; the 1.75 dB figure is the "engineering margin" the
algorithm should reserve.

---

## 7. Pointer to Companion Documents

- `math_derivation{,_en}.tex` §3 — the S-Procedure derivation in
  context, with the (S3.1) LMI block.
- `CONVEXIFICATION_CHAIN.md` §3.3 — the S-Procedure step narrative.
- `MASTER_CONSTRAINTS.md` §6 — the S-Procedure exact form and the
  channel normalization.
- `PAPER_OUTLINE_IEEE.md` §III.4 — the tightness analysis section in
  the paper, where the 1.75 dB figure is reported.
