# Problem Formulation

**Plate**: I (Physical Modeling) + Plate II opening (Mathematical Reformulation start)
**Source of math**: `math_derivation{,_en}.tex` §§1–3
**Master reference**: `MASTER_CONSTRAINTS.md`
**Status**: v4.0 (2026-07-24). Revised to the dedicated per-target sensing
covariance ($\mat{S}_p$) model: per-AP $\mat{Z}_m$ / aggregate $\mat{Z}$, the
Sum Big-M power gate, and the affine $\tr(\mat{F}_p \mat{R}_X) \geq \Gamma$
PCRB shorthand are superseded. Supersedes v3.0 (2026-06-30).

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

**Channel error scaling.** The uncertainty radius is relative to the estimate:
$\|\Delta\vect{h}_k\| \leq \epsilon_h \|\hat{\vect{h}}_k\|$. The final (P3-C1)
LMI carries the $\|\hat{\vect{h}}_k\|^2$ factor in its lower-right entry (the
"intermediate" S-Procedure form), so **no channel normalization is required**;
the simulation channels are scaled to an SNR operating point rather than to
$\|\hat{\vect{h}}_k\| = 1$. See `MASTER_CONSTRAINTS.md` §6 for why the
intermediate form is canonical.

For the full S-Procedure mathematical statement see `math_derivation{,_en}.tex`
§2-C.

### 1.3 Decision variables

| Symbol | Domain | Description |
| --- | --- | --- |
| $\vect{w}_{m,k} \in \mathbb{C}^{N_t}$ | — | Per-AP beamforming vector at AP $m$ for user $k$ |
| $\mat{S}_p \in \mathbb{H}_+^{MN_t}$ | PSD cone | Dedicated sensing covariance for target $p$ (joint stacked matrix; the waveform may still be synthesized cooperatively across APs) |
| $b_{mp} \in \{0, 1\}$ | binary | Sensing-cluster admission indicator: $b_{mp} = 1$ authorizes AP $m$ to allocate dedicated sensing power to target $p$. It never gates communication transmission, and it does not by itself guarantee nonzero sensing power. |

The lifted variables used downstream:

$$
\vect{w}_k = \big[\vect{w}_{1,k}^T, \ldots, \vect{w}_{M,k}^T\big]^T \in \mathbb{C}^{MN_t},
\quad
\mat{W}_k = \vect{w}_k \vect{w}_k^H \in \mathbb{H}_+^{MN_t}
$$

The total transmit covariance is $\mat{R}_X \triangleq \sum_{k=1}^K \mat{W}_k + \sum_{p=1}^P \mat{S}_p$.

---

## 2. Signal Model

### 2.1 Communication signal

AP $m$ transmits $\sum_{k=1}^K \vect{w}_{m,k} s_k + [\textstyle\sum_p \mat{S}_p^{1/2} \vect{x}_p]_m$ where
$s_k \sim \mathcal{CN}(0, 1)$ is user $k$'s data symbol and
$\vect{x}_p \sim \mathcal{CN}(\vect{0}, \mat{I})$ is the dedicated sensing stream for target $p$
($[\cdot]_m$ extracts the AP-$m$ antenna block). The received signal at user $k$ is

$$
y_k = \vect{h}_{k}^H \Big(\sum_{j=1}^K \vect{w}_{j} s_j + \sum_{p=1}^P \mat{S}_p^{1/2} \vect{x}_p\Big) + n_k,
\quad n_k \sim \mathcal{CN}(0, \sigma_c^2)
$$

Unless receiver-side cancellation is explicitly assumed, the aggregate dedicated
sensing covariance $\sum_p \mat{S}_p$ appears as interference in the
communication SINR denominator (§3).

### 2.2 Sensing signal

The echo from target $p$ is driven only by the dedicated waveform for that
target:

$$
\vect{r}_{p} = \vect{g}_{p}^H \mat{S}_p \vect{g}_{p} \, s_p + \vect{n}_{p}^{(\text{sens})}
$$

The sensing signal-to-noise ratio is

$$
\text{SINR}_{S,p} = \frac{\vect{g}_p^H \mat{S}_p \vect{g}_p}{\sigma_s^2}
$$

### 2.3 PCRB on the target state

The data Fisher information matrix for target $p$ is evaluated from the
dedicated sensing covariance only (communication covariances receive no PCRB
credit):

$$
\mat{J}_p^{\text{data}}(\mat{S}_p) = \frac{2}{\sigma_s^2} \Real\Big\{\mat{D}_p^H \mat{S}_p \mat{D}_p\Big\}
$$

where $\mat{D}_p = \partial \vect{g}_p / \partial \boldsymbol{\theta}_p$ is the
target-response derivative w.r.t. the target state $\boldsymbol{\theta}_p$. The
corresponding position-PCRB is $\text{tr}\big((\mat{J}_p^{\text{data}})^{-1}\big)$,
constrained by $\Gamma_{\text{Track},p}$ and converted losslessly into a Schur
LMI with auxiliary $\mat{M}_p$ (see `math_derivation{,_en}.tex` §2-D).

---

## 3. The Worst-Case Communication SINR (Constraint (5b) / (P1-C1))

The single most consequential non-convex constraint in the original problem.
For user $k$, the instantaneous receive SINR under the actual channel
$\vect{h}_k = \hat{\vect{h}}_k + \Delta\vect{h}_k$ is

$$
\text{SINR}_k(\Delta\vect{h}_k) = \frac{|\vect{h}_k^H \vect{w}_k|^2}{\sum_{j \neq k} |\vect{h}_k^H \vect{w}_j|^2 + \vect{h}_k^H \big(\sum_p \mat{S}_p\big) \vect{h}_k + \sigma_c^2}
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
\min_{\substack{\{\vect{w}_{m,k}\}, \{\mat{S}_p\}, \\ \{b_{mp}\}}} \quad &
\sum_{m=1}^{M} \sum_{k=1}^{K} \|\vect{w}_{m,k}\|^2 + \sum_{p=1}^{P} \tr(\mat{S}_p) \tag{P1-objective / (5a)} \\
\text{s.t.} \quad &
\text{SINR}_k^{\text{wc}} \geq \gamma_k, \quad \forall k \in \mathcal{K} \tag{P1-C1 / (5b)} \\
& \vect{g}_p^H \mat{S}_p \vect{g}_p \geq \gamma_S^{\text{PoD}} \sigma_s^2, \quad \forall p \in \mathcal{P} \tag{P1-C2 / (5c)} \\
& \tr\big((\mat{J}_p^{\text{data}}(\mat{S}_p))^{-1}\big) \leq \Gamma_{\text{Track},p}, \quad \forall p \in \mathcal{P} \tag{P1-C3 / (5d)} \\
& \tr(\mat{E}_m \mat{S}_p) \leq P_{\max} \, b_{mp}, \quad \forall m \in \mathcal{M}, p \in \mathcal{P} \tag{P1-C4'a / (5e)} \\
& \tr(\mat{E}_m \mat{R}_X) \leq P_{\max}, \quad \forall m \in \mathcal{M} \tag{P1-C4'b / (5f)} \\
& \sum_{m=1}^{M} b_{mp} = N_{\text{req}}, \quad \forall p \in \mathcal{P}^{\text{active}} \tag{P1-C5' / (5g)} \\
& \mat{S}_p \succeq \mat{0}, \quad \forall p \in \mathcal{P} \tag{P1-C6 / (5h)} \\
& b_{mp} \in \{0, 1\}, \quad \forall m \in \mathcal{M}, p \in \mathcal{P} \tag{P1-C7' / (5i)}
\end{aligned}
$$

(P1-C1) is the worst-case SINR of §3 (its denominator includes the aggregate
dedicated sensing interference $\sum_p \mat{S}_p$). (P1-C4'a) is the
**association gate**: $b_{mp} = 1$ authorizes AP $m$ to allocate up to
$P_{\max}$ of dedicated sensing power to target $p$, and $b_{mp} = 0$ forces
the AP-$m$ block of $\mat{S}_p$ to zero; it does not gate communication
transmission. (P1-C4'b) is the per-AP hardware ceiling on the total transmit
covariance $\mat{R}_X = \sum_k \vect{w}_k \vect{w}_k^H + \sum_p \mat{S}_p$.

**Constraint labeling convention.** The `(5a)–(5h)` numbering is the legacy
labeling carried forward from earlier drafts of this paper. The `(P1-Cx)`
numbering is the labeling used in `math_derivation{,_en}.tex` and
`MASTER_CONSTRAINTS.md`. Both are in use; the `(P1-Cx)` scheme is preferred
in new writing because it survives the (P1)→(P2)→(P3) chain cleanly
(the (5a)-(5h) labels apply only to (P1)).

**Why (P1) is hard.** (P1) is a non-convex mixed-integer nonlinear program
(MINLP): NC1 SINR fractions, NC2 semi-infinite robust constraints, NC3 binary
association, NC4 rank-1 (after lifting), NC6 PCRB matrix inverse. The former
NC5 (beam-sensing bilinear coupling) is eliminated by the dedicated-$\mat{S}_p$
formulation: the association gate (P1-C4'a) and the hardware ceiling (P1-C4'b)
are both linear. See `MASTER_CONSTRAINTS.md` §0.3 for the current NC taxonomy.

**Per-constraint convexity analysis (in the (P1) variable space).** The
non-convexity sources distribute across the (P1) constraints as follows
(detailed analysis in `math_derivation{,_en}.tex` §1, immediately after the
(P1) MINLP): the communication worst-case SINR (P1-C1) carries a fractional
quadratic form in $(\{\vect{w}_k\}, \{\mat{S}_p\})$ on top of a semi-infinite
worst-case quantifier $\forall \|\Delta\vect{h}_k\| \leq \epsilon_h \|\hat{\vect{h}}_k\|$
(Boyd and Vandenberghe, 2004, §4.3.2; Luo et al., 2004); the position-PCRB
(P1-C3) involves $\tr((\mat{J}_p^{\text{data}})^{-1})$ with $\mat{J}_p$ affine
in $\mat{S}_p$; although $\tr(\mat{J}^{-1})$ is convex on $\mat{J} \succ \mat{0}$,
the matrix inverse requires a Schur-complement LMI before standard SDP solvers
can handle it; the association indicators (P1-C7') and the cardinality
equality (P1-C5') impose a discrete $\{0,1\}^{M \times P}$ structure that is
not closed under convex combinations. The association gate (P1-C4'a), the
per-AP hardware ceiling (P1-C4'b), the PSD cone (P1-C6) and the linear
sensing SINR (P1-C2) are themselves convex. The rank-one constraint
(NC4) does not appear in (P1); it is introduced by the covariance lifting
to (P2) and is relaxed there by SDR.

---

## 5. The Lifted Form (P2) — Strictly Equivalent to (P1)

Plate II opens by lifting the per-user beamforming vectors
$\{\vect{w}_{m,k}\}_{m, k}$ into the joint beamforming covariance
$\mat{W}_k = \vect{w}_k \vect{w}_k^H \in \mathbb{H}_+^{MN_t}$; the dedicated
sensing covariances $\{\mat{S}_p\}$ are already joint matrices and carry over
unchanged. The rank-1 constraint (P2-C8) makes this lifting a bijection.

$$
\begin{aligned}
\min_{\substack{\{\mat{W}_k\}, \{\mat{S}_p\}, \\ \{b_{mp}\}}} \quad &
\sum_{k=1}^K \tr(\mat{W}_k) + \sum_{p=1}^P \tr(\mat{S}_p) \tag{P2} \\
\text{s.t.} \quad &
\text{(P2-C1) worst-case comm SINR, cf.\ (P1-C1)}, \quad \forall k \\
& \tr(\vect{g}_p \vect{g}_p^H \mat{S}_p) \geq \gamma_S^{\text{PoD}} \sigma_s^2, \quad \forall p \tag{P2-C2} \\
& \tr\big((\mat{J}_p^{\text{data}}(\mat{S}_p))^{-1}\big) \leq \Gamma_{\text{Track},p}, \quad \forall p \tag{P2-C3} \\
& \tr(\mat{E}_m \mat{S}_p) \leq P_{\max} \, b_{mp}, \quad \forall m, p \tag{P2-C4'a} \\
& \tr(\mat{E}_m \mat{R}_X) \leq P_{\max}, \quad \forall m \tag{P2-C4'b} \\
& \sum_{m=1}^{M} b_{mp} = N_{\text{req}}, \quad \forall p \in \mathcal{P}^{\text{active}} \tag{P2-C5'} \\
& \mat{W}_k \succeq \mat{0}, \quad \forall k \tag{P2-C6} \\
& \mat{S}_p \succeq \mat{0}, \quad \forall p \tag{P2-C7} \\
& \rank(\mat{W}_k) = 1, \quad \forall k \tag{P2-C8} \\
& b_{mp} \in \{0, 1\}, \quad \forall m, p \tag{P2-C9}
\end{aligned}
$$

where

- $\mat{R}_X = \sum_{k=1}^K \mat{W}_k + \sum_{p=1}^P \mat{S}_p$
- $\mat{E}_m = \text{diag}(\underbrace{0,\ldots,0}_{(m-1)N_t}, \underbrace{1,\ldots,1}_{N_t}, \underbrace{0,\ldots,0}_{(M-m)N_t})$ is the per-AP antenna selector
- $\mat{J}_p^{\text{data}}(\mat{S}_p) = \frac{2}{\sigma_s^2} \Real\{\mat{D}_p^H \mat{S}_p \mat{D}_p\} \in \mathbb{H}_+^{N_\theta \times N_\theta}$ is affine in $\mat{S}_p$ only

**Strict equivalence (P1) ↔ (P2).** The lifting $\vect{w}_k \leftrightarrow
\mat{W}_k = \vect{w}_k \vect{w}_k^H$ is a bijection on the rank-1
manifold, and the constraints are rewritten via the identity
$\tr(\vect{x}\vect{x}^H \mat{W}) = \vect{x}^H \mat{W} \vect{x}$. No relaxation
is introduced. See `math_derivation{,_en}.tex` §2-A for the proof.

---

## 6. The DC-Penalty SDP (P3) — One-Step View

(P3) is the DC-penalty SDP obtained by applying the convexification chain
(lifting, SDR, cross-multiplication, S-Procedure, PCRB Schur LMI, box + DC
penalty for the binary association) to (P2). Its full form is in
`math_derivation{,_en}.tex` §2-F and the per-constraint reference card is in
`MASTER_CONSTRAINTS.md` §2. Its feasible region is convex; the only
non-convexity is the DC objective (rank-1 penalty + binarity penalty), handled
by the dual DC-penalty SCA of `math_derivation{,_en}.tex` §3.

**Plate II is closed at (P3):** the original MINLP has been reduced to a
DC-penalty SDP whose every SCA subproblem is a standard convex SDP handed to
MOSEK/SDPT3 via CVX. The remaining plates deal with engineering feasibility
(rank-one recovery and binary candidate validation) and the closed-form
baseline.

---

## 7. Symbol Reference

The full symbol reference is in `MASTER_CONSTRAINTS.md` §5. The summary
here covers only symbols introduced in (P1) that (P2) and (P3) reuse.

| Symbol | Definition | First defined |
| --- | --- | --- |
| $\vect{w}_{m,k}$ | Per-AP beamforming vector | §1.3 |
| $\mat{S}_p$ | Dedicated sensing covariance for target $p$ | §1.3 |
| $b_{mp}$ | Sensing-cluster admission indicator | §1.3 |
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
| $\mat{D}_p$ | Sensing derivative matrix w.r.t. target state | §2.3 |

The lifted variables ($\mat{W}_k, \mat{R}_X$) and the
problem-specific constant matrices ($\mat{A}_k, \mat{E}_m, \mat{D}_p$) are
defined in `MASTER_CONSTRAINTS.md` §5 and §6.

---

## 8. Pointer to Companion Documents

- `math_derivation{,_en}.tex` — the LaTeX-compilable source of record for (P1), (P2), (P3) and the full convexification chain. This Markdown file is the prose companion.
- `MASTER_CONSTRAINTS.md` — the master formula reference. Always authoritative on the exact form of any equation.
- `CONVEXIFICATION_CHAIN.md` — the prose companion to `math_derivation{,_en}.tex` §3–§6. Step-by-step convexification narrative.
- `PAPER_OUTLINE_IEEE.md` — the 4-plate IMRaD outline that drives the structure of this file and `CONVEXIFICATION_CHAIN.md`.

If a conflict arises between this file and `MASTER_CONSTRAINTS.md` or
`math_derivation{,_en}.tex`, the latter two win. This file is a narrative
companion, not the source of record for the math.
