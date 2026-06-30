# Problem Formulation

**Plate**: I (Physical Modeling) + Plate II opening (Mathematical Reformulation start)
**Source of math**: `math_derivation{,_en}.tex` §§1–3
**Master reference**: `MASTER_CONSTRAINTS.md`
**Status**: v3.0 (2026-06-30). Rewritten under the unified mathematical setting
established by the 2026-06-30 audit. Supersedes the previous
`PROBLEM_FORMULATION_RIGOROUS.md` (deleted in 19ff970).

---

## 1. Network and Channel Model

### 1.1 Cell-free AP layout

A cell-free integrated sensing and communication (ISAC) system with $M$ access
points (APs) jointly serving $K$ single-antenna communication users and
illuminating $P$ point targets.

| Symbol | Set | Size | Description |
| --- | --- | --- | --- |
| $\mathcal{M}$ | $\{1, \ldots, M\}$ | $M$ | APs, each with $N_t$ antennas |
| $\mathcal{K}$ | $\{1, \ldots, K\}$ | $K$ | Communication users, single antenna each |
| $\mathcal{P}$ | $\{1, \ldots, P\}$ | $P$ | Sensing targets |

Default operating point (per `PAPER_OUTLINE_IEEE.md` §6.1): $M = 16$, $N_t = 4$,
$K = 10$, $P = 4$.

### 1.2 Channel model with bounded error

For user $k \in \mathcal{K}$ and AP $m \in \mathcal{M}$:

$$
\vect{h}_{m,k} = \hat{\vect{h}}_{m,k} + \Delta\vect{h}_{m,k}, \quad \|\Delta\vect{h}_{m,k}\|_2 \leq \epsilon_{h,m}
$$

For target $p \in \mathcal{P}$ and AP $m \in \mathcal{M}$:

$$
\vect{g}_{m,p} = \hat{\vect{g}}_{m,p} + \Delta\vect{g}_{m,p}, \quad \|\Delta\vect{g}_{m,p}\|_2 \leq \epsilon_{g,m}
$$

**Stacking convention.** The joint AP signal dimension is $MN_t$. The stacked
channels and beamforming vectors are:

$$
\hat{\vect{h}}_k = \big[\hat{\vect{h}}_{1,k}^T, \ldots, \hat{\vect{h}}_{M,k}^T\big]^T \in \mathbb{C}^{MN_t}
\quad\quad
\hat{\vect{g}}_p = \big[\hat{\vect{g}}_{1,p}^T, \ldots, \hat{\vect{g}}_{M,p}^T\big]^T \in \mathbb{C}^{MN_t}
$$

**Channel normalization** (see `math_derivation{,_en}.tex` Assumption
`ass:channel_norm`):

$$
\|\hat{\vect{h}}_k\|_2 = 1, \quad \forall k \in \mathcal{K}
$$

This normalization is what allows the final (P3-C1) LMI to drop the
$\|\hat{\vect{h}}_k\|^2$ factor that appears in the intermediate S-Procedure
form (S3.1). See `MASTER_CONSTRAINTS.md` §6 for the equivalence with the
un-normalized form.

For the full S-Procedure mathematical statement see `math_derivation{,_en}.tex`
§3 (逐步凸化 Step 3).

### 1.3 Decision variables

| Symbol | Domain | Description |
| --- | --- | --- |
| $\vect{w}_{m,k} \in \mathbb{C}^{N_t}$ | — | Per-AP beamforming vector at AP $m$ for user $k$ |
| $\mat{Z}_m \in \mathbb{H}_+^{N_t}$ | PSD cone | Per-AP sensing covariance at AP $m$ |
| $b_{mp} \in \{0, 1\}$ | binary | AP selection indicator: $b_{mp} = 1$ if AP $m$ serves target $p$ |

The lifted variables used downstream:

$$
\vect{w}_k = \big[\vect{w}_{1,k}^T, \ldots, \vect{w}_{M,k}^T\big]^T \in \mathbb{C}^{MN_t},
\quad
\mat{W}_k = \vect{w}_k \vect{w}_k^H \in \mathbb{H}_+^{MN_t},
\quad
\mat{Z} = \text{blkdiag}(\mat{Z}_1, \ldots, \mat{Z}_M) \in \mathbb{H}_+^{MN_t}
$$

The total transmit covariance is $\mat{R}_X \triangleq \sum_{k=1}^K \mat{W}_k + \mat{Z}$.

---

## 2. Signal Model

### 2.1 Communication signal

AP $m$ transmits $\sum_{k=1}^K \vect{w}_{m,k} s_k + \mat{Z}_m^{1/2} \vect{x}_m$ where
$s_k \sim \mathcal{CN}(0, 1)$ is user $k$'s data symbol and
$\vect{x}_m \sim \mathcal{CN}(\vect{0}, \mat{I})$ is the dedicated sensing stream.
The received signal at user $k$ is

$$
y_k = \sum_{m=1}^M \vect{h}_{m,k}^H \Big(\sum_{j=1}^K \vect{w}_{m,j} s_j + \mat{Z}_m^{1/2} \vect{x}_m\Big) + n_k,
\quad n_k \sim \mathcal{CN}(0, \sigma_c^2)
$$

### 2.2 Sensing signal

The echo from target $p$ at AP $m$ is

$$
\vect{r}_{m,p} = \vect{g}_{m,p}^H \mat{Z}_m \vect{g}_{m,p} \, s_p + \vect{n}_{m,p}^{(\text{sens})}
$$

The sensing signal-to-noise ratio is

$$
\text{SINR}_{S,p} = \frac{|\vect{g}_p^H \mat{Z} \vect{g}_p|}{\sigma_s^2}
$$

### 2.3 PCRB on the target state

The data Fisher information matrix for target $p$ is

$$
\mat{J}_p^{\text{data}} = \frac{2}{\sigma_s^2} \Real\Big\{\nabla_{\boldsymbol{\theta}_p}^H \vect{g}_p \cdot \mat{R}_X \cdot \nabla_{\boldsymbol{\theta}_p} \vect{g}_p^H\Big\}
$$

where $\boldsymbol{\theta}_p$ is the target state. The corresponding
position-PCRB is $\text{tr}\big((\mat{J}_p^{\text{data}})^{-1}\big)$.

For the affine expansion of the PCRB used downstream, see
`math_derivation{,_en}.tex` §4 (逐步凸化 Step 5a) and the proposition that
defines the constant Hermitian PSD matrix $\mat{F}_p$.

---

## 3. The Worst-Case Communication SINR (Constraint (5b) / (P1-C1))

The single most consequential non-convex constraint in the original problem.
For user $k$, the instantaneous receive SINR under the actual channel
$\vect{h}_k = \hat{\vect{h}}_k + \Delta\vect{h}_k$ is

$$
\text{SINR}_k(\Delta\vect{h}_k) = \frac{|\vect{h}_k^H \vect{w}_k|^2}{\sum_{j \neq k} |\vect{h}_k^H \vect{w}_j|^2 + \sigma_c^2}
$$

The **worst-case** SINR over the uncertainty ball
$\|\Delta\vect{h}_k\| \leq \epsilon_h \|\hat{\vect{h}}_k\|$ is

$$
\text{SINR}_k^{\text{wc}} := \min_{\|\Delta\vect{h}_k\| \leq \epsilon_h \|\hat{\vect{h}}_k\|}
\text{SINR}_k(\Delta\vect{h}_k) \geq \gamma_k, \quad \forall k \in \mathcal{K} \tag{P1-C1 / (5b)}
$$

This constraint is doubly non-convex:
1. **Fractional quadratic form** in $\vect{w}_k$ (NC1).
2. **Semi-infinite** worst-case quantifier over the ball (NC2).

The full handling of both non-convexities via the 7-step chain is in
`CONVEXIFICATION_CHAIN.md` §3.1 (Step 3) and §3.4 (Step 4).

---

## 4. The Original Problem (P1)

Combining all constraints from the signal model, the original problem is

$$
\begin{aligned}
\min_{\substack{\{\vect{w}_{m,k}\}, \{\mat{Z}_m\}, \\ \{b_{mp}\}}} \quad &
\sum_{m=1}^{M} \Big( \sum_{k=1}^{K} \|\vect{w}_{m,k}\|^2 + \tr(\mat{Z}_m) \Big) \tag{P1-objective / (5a)} \\
\text{s.t.} \quad &
\text{SINR}_k^{\text{wc}} \geq \gamma_k, \quad \forall k \in \mathcal{K} \tag{P1-C1 / (5b)} \\
& \text{SINR}_{S,p} \geq \gamma_S^{\text{PoD}}, \quad \forall p \in \mathcal{P} \tag{P1-C2 / (5c)} \\
& \tr\big(\mat{J}_p^{\text{data}}\big) \geq \Gamma_{\text{Track},p}, \quad \forall p \in \mathcal{P} \tag{P1-C3 / (5d)} \\
& \sum_{m=1}^{M} b_{mp} = N_{\text{req}}, \quad \forall p \in \mathcal{P}^{\text{active}} \tag{P1-C4 / (5e)} \\
& \sum_{k=1}^{K} \|\vect{w}_{m,k}\|^2 + \tr(\mat{Z}_m) \leq P_{\max}, \quad \forall m \in \mathcal{M} \tag{P1-C5 / (5f)} \\
& \mat{Z}_m \succeq \mat{0}, \quad \forall m \in \mathcal{M} \tag{P1-C6 / (5g)} \\
& b_{mp} \in \{0, 1\}, \quad \forall m \in \mathcal{M}, p \in \mathcal{P} \tag{P1-C7 / (5h)}
\end{aligned}
$$

**Constraint labeling convention.** The `(5a)–(5h)` numbering is the legacy
labeling carried forward from earlier drafts of this paper. The `(P1-Cx)`
numbering is the labeling used in `math_derivation{,_en}.tex` and
`MASTER_CONSTRAINTS.md`. Both are in use; the `(P1-Cx)` scheme is preferred
in new writing because it survives the (P1)→(P2)→(P3) chain cleanly
(the (5a)-(5h) labels apply only to (P1)).

**Why (P1) is hard.** (P1) is a non-convex mixed-integer nonlinear program
(MINLP): NC1 SINR fractions, NC2 semi-infinite robust constraints, NC3 binary
selection, NC4 rank-1 (after lifting), NC5 beam-sensing bilinear coupling,
NC6 PCRB matrix inverse. See `CONVEXIFICATION_CHAIN.md` §2 for the
non-convexity source identification and §3 for the 7-step chain.

---

## 5. The Lifted Form (P2) — Strictly Equivalent to (P1)

Plate II opens by replacing the per-AP sensing covariances
$\{\mat{Z}_m\}_{m \in \mathcal{M}}$ with the block-diagonal joint covariance
$\mat{Z} = \text{blkdiag}(\mat{Z}_1, \ldots, \mat{Z}_M) \in \mathbb{H}_+^{MN_t}$,
and the per-user beamforming vectors $\{\vect{w}_{m,k}\}_{m, k}$ with the
joint beamforming covariance $\mat{W}_k = \vect{w}_k \vect{w}_k^H \in
\mathbb{H}_+^{MN_t}$. The rank-1 constraint (P2-C7) makes this lifting a
bijection.

$$
\begin{aligned}
\min_{\substack{\{\mat{W}_k\}, \mat{Z}, \\ \{b_{mp}\}}} \quad &
\sum_{k=1}^K \tr(\mat{W}_k) + \tr(\mat{Z}) \tag{P2} \\
\text{s.t.} \quad &
\tr(\vect{g}_p \vect{g}_p^H \mat{Z}) \geq \gamma_S^{\text{PoD}} \sigma_s^2, \quad \forall p \tag{P2-C2} \\
& \tr(\mat{F}_p \mat{R}_X) \geq \Gamma_{\text{Track},p}, \quad \forall p \tag{P2-C3} \\
& \tr(\mat{E}_m \mat{R}_X) \leq P_{\max}, \quad \forall m \tag{P2-C4} \\
& \mat{W}_k \succeq \mat{0}, \quad \forall k \tag{P2-C5} \\
& \mat{Z} \succeq \mat{0} \tag{P2-C6} \\
& \rank(\mat{W}_k) = 1, \quad \forall k \tag{P2-C7} \\
& b_{mp} \in \{0, 1\}, \quad \forall m, p \tag{P2-C8}
\end{aligned}
$$

where

- $\mat{R}_X = \sum_{k=1}^K \mat{W}_k + \mat{Z}$
- $\mat{E}_m = \text{diag}(\underbrace{0,\ldots,0}_{(m-1)N_t}, \underbrace{1,\ldots,1}_{N_t}, \underbrace{0,\ldots,0}_{(M-m)N_t})$ is the per-AP antenna selector
- $\mat{F}_p = \frac{2}{\sigma_s^2} \Real\{\nabla_{\boldsymbol{\theta}_p} \vect{g}_p^H \cdot \nabla_{\boldsymbol{\theta}_p}^H \vect{g}_p\} \in \mathbb{H}_+^{MN_t}$

**Strict equivalence (P1) ↔ (P2).** The lifting $\vect{w}_k \leftrightarrow
\mat{W}_k = \vect{w}_k \vect{w}_k^H$ is a bijection on the rank-1
manifold, and the constraints are rewritten via the identity
$\tr(\vect{x}\vect{x}^H \mat{W}) = \vect{x}^H \mat{W} \vect{x}$. No relaxation
is introduced. See `math_derivation{,_en}.tex` §2 for the proof.

---

## 6. The Convex SDP (P3) — One-Step View

(P3) is the convex SDP obtained by applying all 7 convexification steps to
(P2). Its full form is in `math_derivation{,_en}.tex` §6 (最终凸 SDP 问题)
and the per-constraint reference card is in `MASTER_CONSTRAINTS.md` §2.
The step-by-step chain is documented in `CONVEXIFICATION_CHAIN.md` §3.

**Plate II is closed at (P3):** the original MINLP has been reduced to a
convex SDP, and (P3) is the problem we hand to MOSEK/CVX. The remaining
plates deal with engineering feasibility (Plate III, rank-one recovery via
Gaussian randomization) and the closed-form baseline (Plate IV).

---

## 7. Symbol Reference

The full symbol reference is in `MASTER_CONSTRAINTS.md` §5. The summary
here covers only symbols introduced in (P1) that (P2) and (P3) reuse.

| Symbol | Definition | First defined |
| --- | --- | --- |
| $\vect{w}_{m,k}$ | Per-AP beamforming vector | §1.3 |
| $\mat{Z}_m$ | Per-AP sensing covariance | §1.3 |
| $b_{mp}$ | AP selection indicator | §1.3 |
| $\vect{h}_{m,k}, \vect{g}_{m,p}$ | Per-AP channels | §1.2 |
| $\hat{\vect{h}}_{m,k}, \hat{\vect{g}}_{m,p}$ | Channel estimates | §1.2 |
| $\Delta\vect{h}_{m,k}, \Delta\vect{g}_{m,p}$ | Channel errors (bounded) | §1.2 |
| $\epsilon_{h,m}, \epsilon_{g,m}$ | Per-AP error bounds | §1.2 |
| $\sigma_c^2, \sigma_s^2$ | Communication / sensing noise power | §2 |
| $\gamma_k$ | Per-user SINR target | §3 |
| $\gamma_S^{\text{PoD}}$ | Probability-of-detection threshold | §4 |
| $\Gamma_{\text{Track},p}$ | PCRB tracking threshold | §4 |
| $P_{\max}$ | Per-AP peak power budget | §4 |
| $N_{\text{req}}$ | Required APs per active target | §4 |
| $\nabla_{\boldsymbol{\theta}_p} \vect{g}_p$ | Sensing steering-vector gradient w.r.t. target state | §2.3 |

The lifted variables ($\mat{W}_k, \mat{Z}, \mat{R}_X$) and the
problem-specific constant matrices ($\mat{A}_k, \mat{E}_m, \mat{F}_p$) are
defined in `MASTER_CONSTRAINTS.md` §5 and §6.

---

## 8. Pointer to Companion Documents

- `math_derivation{,_en}.tex` — the LaTeX-compilable source of record for (P1), (P2), (P3) and the full 7-step chain. This Markdown file is the prose companion.
- `MASTER_CONSTRAINTS.md` — the master formula reference. Always authoritative on the exact form of any equation.
- `CONVEXIFICATION_CHAIN.md` — the prose companion to `math_derivation{,_en}.tex` §3–§6. Step-by-step convexification narrative.
- `PAPER_OUTLINE_IEEE.md` — the 4-plate IMRaD outline that drives the structure of this file and `CONVEXIFICATION_CHAIN.md`.

If a conflict arises between this file and `MASTER_CONSTRAINTS.md` or
`math_derivation{,_en}.tex`, the latter two win. This file is a narrative
companion, not the source of record for the math.
