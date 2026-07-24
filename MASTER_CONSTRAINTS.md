# Master Constraints Table — ISAC Research

**Status**: Human-readable reference. Does NOT participate in LaTeX compilation. **Revised 2026-07 to the dedicated per-target sensing covariance ($\mat{S}_p$) model**: the aggregate $\mat{Z}$ / per-AP $\mat{Z}_m$, the Sum Big-M power gate, and the affine $\tr(\mat{F}_p \mat{R}_X) \geq \Gamma$ PCRB shorthand are superseded (see §2 revision note).
**Authoritative source**: `math_derivation.tex` (中文, 220 lines) and `math_derivation_en.tex` (English, 217 lines). All formula content in this table is transcribed from those two files. The four `report*.tex` main-paper drafts (deleted 2026-06-30) are no longer a source.
**Audience**: Anyone who needs to look up the exact form, label, or physical meaning of a P1/P2/P3 constraint without re-reading the full derivation.

---

## 0. Naming Conventions

### 0.1 Chain structure (current)

As of the 2026-07 model revision (dedicated per-target sensing covariance $\mat{S}_p$), the derivation in `math_derivation{,_en}.tex` is organized as: §A covariance lifting (P1→P2), §B SDR (rank-1 drop), §C cross-multiplication + S-Procedure, §D PCRB Schur-complement LMI, §E binary association via box relaxation + DC penalty, §F consolidation (P3), and §3 the dual DC-penalty SCA that jointly recovers rank-1 and binarity. The earlier "seven-step" count (Steps 1–7, with Step 7 an outer path-loss AP-selection heuristic) is **superseded**: the Step 7 heuristic survives only as an SCA cold-start aid in §E. The large-scale-fading top-$N_{\text{req}}$ sort is no longer the solver.

### 0.2 Constraint labels (P1-Cx / P2-Cx / P3-Cx)

| Tag | Source | Used in |
| --- | --- | --- |
| `(P1-C1)` through `(P1-C7')` | The constraints of the original (P1) MINLP, incl. primed `(P1-C4'a)/(P1-C4'b)/(P1-C5')/(P1-C7')` | `math_derivation{,_en}.tex` §1 |
| `(P2-C1)` through `(P2-C9)` | The nine constraints of the lifted (P2) | `math_derivation{,_en}.tex` §2-A |
| `(P3-C1)` through `(P3-C9)` | The constraints of the final (P3), incl. `(P3-C3b)`, `(P3-C4'a)`, `(P3-C4'b)`, `(P3-C5')` | `math_derivation{,_en}.tex` §2-F (with `\label{eq:p3c1}`–`\label{eq:p3c9}`) |

The legacy `(5a)`–`(5h)` numbering is also used in some prose
contexts (see `PROBLEM_FORMULATION.md` §4 for the legacy ↔ `(P1-Cx)`
mapping). The `(P1-Cx)` scheme is preferred in new writing because it
survives the (P1)→(P2)→(P3) chain cleanly.

### 0.3 NC tags (non-convexity sources)

The non-convexity sources are tagged NC1 through NC6. Under the dedicated-$\mat{S}_p$ formulation, NC5 no longer occurs:

| NC | Source | Targeted by |
| --- | --- | --- |
| NC1 | SINR fractional structure | §C Step 1 (cross-multiplication) |
| NC2 | Semi-infinite robust constraints | §C Step 2 (S-Procedure) |
| NC3 | Binary AP--target association | §E (box + DC penalty), recovered jointly in §3 |
| NC4 | Implicit rank-one | §B (SDR), recovered jointly in §3 |
| NC5 | Beam-sensing bilinear coupling | **Eliminated**: the association gate (P1-C4'a) and the hardware ceiling (P1-C4'b) are both linear |
| NC6 | Matrix inverse in PCRB | §D (Schur complement, lossless) |

---

## 1. The three problems (P1 → P2 → P3) — the story in one paragraph

`math_derivation.tex` walks the problem through three equivalent / relaxed forms:

- **(P1)** — Original problem with per-AP beamforming vectors $\vect{w}_{m,k}$, per-target dedicated sensing covariances $\mat{S}_p \in \mathbb{H}_+^{MN_t}$, and binary AP--target association $b_{mp} \in \{0,1\}$ (a sensing-cluster admission indicator; it authorizes dedicated sensing power but never gates communication transmission). (math_derivation.tex §问题建模.)
- **(P2)** — Lifted covariance form. Replace $\vect{w}_k$ with $\mat{W}_k = \vect{w}_k \vect{w}_k^H$; the $\mat{S}_p$ carry over unchanged. Adds rank-1 constraint (P2-C8). **Strictly equivalent** to (P1) by the lifting $\vect{w} \leftrightarrow \vect{w}\vect{w}^H$. (math_derivation.tex §优化问题求解-A.)
- **(P3)** — DC-penalty SDP after dropping rank-1 (SDR) and relaxing the binary association to the box (P3-C9) plus a DC penalty $\eta_b \sum_{m,p}(b_{mp} - b_{mp}^2)$ in the objective. Introduces S-Procedure slack $\mu_k \geq 0$ for the worst-case channel error $\|\Delta\vect{h}_k\| \leq \epsilon_h \|\hat{\vect{h}}_k\|$, and PCRB auxiliaries $\mat{M}_p$ for the exact trace-of-inverse Schur LMI. The aggregate dedicated sensing covariance $\sum_p \mat{S}_p$ enters $\mat{A}_k$ as UE interference. (math_derivation.tex §优化问题求解-F.)

---

## 2. (P3) — Final DC-penalty SDP (the master form)

Decision variables: $\mat{W}_k \in \mathbb{H}_+^{MN_t}$ (per-user beamforming covariance), $\mat{S}_p \in \mathbb{H}_+^{MN_t}$ (per-target dedicated sensing covariance), $\mu_k \geq 0$ (S-Procedure slack), $\mat{M}_p \in \mathbb{H}^{N_\theta \times N_\theta}$ (PCRB auxiliary), $b_{mp} \in [0,1]$ (soft association).

Key definitions: $\mat{R}_X \triangleq \sum_{k=1}^K \mat{W}_k + \sum_{p=1}^P \mat{S}_p$; $\mat{A}_k \triangleq \frac{1}{\gamma_k} \mat{W}_k - \sum_{j \neq k} \mat{W}_j - \sum_{p} \mat{S}_p$ (the aggregate dedicated sensing covariance is interference at the UEs unless receiver-side cancellation is explicitly assumed); $\mat{J}_p^{\text{data}}(\mat{S}_p) = \frac{2}{\sigma_s^2} \Real\{\mat{D}_p^H \mat{S}_p \mat{D}_p\}$ (affine in $\mat{S}_p$ only — communication covariances receive no PCRB credit).

$$
\begin{aligned}
\min_{\{\mat{W}_k\}, \{\mat{S}_p\}, \{\mu_k\}, \{\mat{M}_p\}, \{b_{mp}\}} \quad & \sum_{k=1}^{K} \tr(\mat{W}_k) + \sum_{p=1}^{P} \tr(\mat{S}_p) + \eta_b \sum_{m,p} (b_{mp} - b_{mp}^2) \tag{P3} \\
\text{s.t.} \quad & \begin{bmatrix} \mat{A}_k + \mu_k \mat{I} & \mat{A}_k \hat{\vect{h}}_k \\[1.5ex] \hat{\vect{h}}_k^H \mat{A}_k & \hat{\vect{h}}_k^H \mat{A}_k \hat{\vect{h}}_k - \sigma_c^2 - \mu_k \epsilon_h^2 \|\hat{\vect{h}}_k\|^2 \end{bmatrix} \succeq \mat{0}, \quad \forall k \tag{P3-C1} \\
& \tr(\vect{g}_p \vect{g}_p^H \mat{S}_p) \geq \gamma_S^{\text{PoD}} \sigma_s^2, \quad \forall p \tag{P3-C2} \\
& \begin{bmatrix} \mat{M}_p & \mat{I} \\ \mat{I} & \mat{J}_p^{\text{data}}(\mat{S}_p) \end{bmatrix} \succeq \mat{0}, \quad \forall p \tag{P3-C3} \\
& \tr(\mat{M}_p) \leq \Gamma_{\text{Track},p}, \quad \forall p \tag{P3-C3b} \\
& \tr(\mat{E}_m \mat{S}_p) \leq P_{\max} \, b_{mp}, \quad \forall m, p \tag{P3-C4'a} \\
& \tr(\mat{E}_m \mat{R}_X) \leq P_{\max}, \quad \forall m \tag{P3-C4'b} \\
& \sum_{m=1}^{M} b_{mp} = N_{\text{req}}, \quad \forall p \in \mathcal{P}^{\text{active}} \tag{P3-C5'} \\
& \mat{W}_k \succeq \mat{0}, \quad \forall k \tag{P3-C6} \\
& \mat{S}_p \succeq \mat{0}, \quad \forall p \tag{P3-C7} \\
& \mu_k \geq 0, \quad \forall k \tag{P3-C8} \\
& 0 \leq b_{mp} \leq 1, \quad \forall m, p \tag{P3-C9}
\end{aligned}
$$

> **Source of truth**: `math_derivation{,_en}.tex` §2-F (最终凸 SDP 问题). The DC term $\eta_b \sum_{m,p}(b_{mp} - b_{mp}^2)$ and the rank-1 penalty are handled jointly by the dual DC-penalty SCA of §3 therein; each SCA subproblem (P3-SCA-$t$) linearizes both penalties and is a standard convex SDP.
>
> **Model revision (2026-07)**: the aggregate sensing covariance $\mat{Z}$ (and the per-AP $\mat{Z}_m$) was replaced by per-target dedicated covariances $\{\mat{S}_p\}$; the Sum Big-M power gate $\tr(\mat{E}_m \mat{R}_X) \leq P_{\max} \sum_p b_{mp}$ was replaced by the per-(AP, target) association gate (P3-C4'a); the PCRB constraint changed from the affine FIM lower bound $\tr(\mat{F}_p \mat{R}_X) \geq \Gamma$ to the exact trace-of-inverse Schur LMI (P3-C3)+(P3-C3b) evaluated on $\mat{S}_p$ only. $b_{mp}$ is a sensing-cluster *admission indicator*: it authorizes, but does not force, dedicated sensing power.

### Per-constraint reference card

| Tag | Type | Variables | Physical meaning |
| --- | --- | --- | --- |
| `(P3-C1)` | LMI ($MN_t{+}1$ dim.) | $\mat{W}_k, \{\mat{S}_p\}, \mu_k$ | Communication worst-case SINR under $\|\Delta\vect{h}_k\| \leq \epsilon_h \|\hat{\vect{h}}_k\|$, with sensing interference inside $\mat{A}_k$ |
| `(P3-C2)` | Linear inequality | $\mat{S}_p$ | Sensing detection probability (PoD criterion at $\gamma_S^{\text{PoD}}$) |
| `(P3-C3)` | LMI ($2N_\theta$ dim.) | $\mat{S}_p, \mat{M}_p$ | Schur-complement form of $\mat{M}_p \succeq (\mat{J}_p^{\text{data}})^{-1}$ |
| `(P3-C3b)` | Linear inequality | $\mat{M}_p$ | PCRB trace upper bound $\Gamma_{\text{Track},p}$ |
| `(P3-C4'a)` | Linear inequality | $\mat{S}_p, b_{mp}$ | Association gate: AP $m$ may allocate dedicated sensing power to target $p$ only if $b_{mp} = 1$ |
| `(P3-C4'b)` | Linear inequality | $\mat{R}_X$ | Per-AP peak power (hardware ceiling) |
| `(P3-C5')` | Linear equality | $b_{mp}$ | Exactly $N_{\text{req}}$ APs authorized per active target |
| `(P3-C6)` | PSD cone | $\mat{W}_k$ | SDR relaxation of $\rank(\mat{W}_k) = 1$ |
| `(P3-C7)` | PSD cone | $\mat{S}_p$ | Dedicated sensing covariance PSD |
| `(P3-C8)` | Non-negative orthant | $\mu_k$ | S-Procedure slack |
| `(P3-C9)` | Box | $b_{mp}$ | Continuous relaxation of $b_{mp} \in \{0,1\}$ (binarity restored by the DC penalty) |

---

## 3. (P2) — Lifted but still non-convex (one step before SDR)

(P2) is what (P3) relaxes. The differences are exactly the two non-convexities that SDR and the DC penalty handle:

$$
\begin{aligned}
\text{(P2) = (P3) with two changes: } & \\
& \rank(\mat{W}_k) = 1, \quad \forall k \tag{P2-C8} \\
& b_{mp} \in \{0,1\} \text{ instead of box + DC penalty}, \quad \forall m, p \tag{P2-C9}
\end{aligned}
$$

(P2) is **strictly equivalent** to (P1) because the rank-1 constraint makes the lifting $\vect{w} \leftrightarrow \mat{w}\mat{w}^H$ a bijection. The PCRB constraint (P2-C3) keeps the trace-of-inverse form $\tr((\mat{J}_p^{\text{data}}(\mat{S}_p))^{-1}) \leq \Gamma_{\text{Track},p}$, which (P3) converts losslessly into the Schur pair (P3-C3)+(P3-C3b).

---

## 4. (P1) — Original problem

```text
min    Σ_m Σ_k ‖w_{m,k}‖² + Σ_p tr(S_p)
s.t.   (P1-C1)   worst-case comm SINR ≥ γ_k   ∀k   (sensing covariance Σ_p S_p counts as UE interference)
       (P1-C2)   g_p^H S_p g_p / σ_s² ≥ γ_S^PoD  ∀p
       (P1-C3)   tr( (J_p^data(S_p))^{-1} ) ≤ Γ_Track,p  ∀p
       (P1-C4'a) tr(E_m S_p) ≤ P_max · b_mp   ∀m, p   (association gate)
       (P1-C4'b) tr(E_m R_X) ≤ P_max          ∀m      (hardware ceiling)
       (P1-C5')  Σ_m b_{mp} = N_req           ∀p active
       (P1-C6)   S_p ⪰ 0                     ∀p
       (P1-C7')  0 ≤ b_{mp} ≤ 1              ∀m, p   (binary in the original; box is the relaxation)
```

### (P1) → (P3) constraint correspondence

| (P1) | (P3) | Transform | Convexity |
| --- | --- | --- | --- |
| (P1-C1) comm SINR | (P3-C1) LMI | lifting + SDR + S-Procedure + fraction cross-multiplication | LMI (convex) |
| (P1-C2) sensing SINR | (P3-C2) linear | lifting ($\vect{g}_p^H \mat{S}_p \vect{g}_p = \tr(\vect{g}_p \vect{g}_p^H \mat{S}_p)$) | linear (convex) |
| (P1-C3) PCRB | (P3-C3)+(P3-C3b) LMI/linear | Schur complement (lossless) | LMI (convex) |
| (P1-C4'a) association gate | (P3-C4'a) linear | identity (linear in $(\mat{S}_p, b_{mp})$) | linear (convex) |
| (P1-C4'b) per-AP power | (P3-C4'b) linear | lifting via $\mat{E}_m$ selector | linear (convex) |
| (P1-C5') service count | (P3-C5') linear equality | identity | linear (convex) |
| (P1-C6) PSD | (P3-C7) PSD | identity | PSD cone (convex) |
| (P1-C7') box/binary | (P3-C9) box + DC penalty | continuous relaxation + $\eta_b \sum (b - b^2)$ | box (convex) + DC objective |
| rank-1 (implicit in lifting) | (P3-C6) PSD | SDR | PSD cone (convex, **not tight**) |

---

## 5. Symbol conventions (enforced via `\newcommand` in both math_derivation files)

| Symbol | Definition | First defined in |
| --- | --- | --- |
| $\mat{W}_k$ | Per-user beamforming covariance, $\mat{W}_k = \vect{w}_k \vect{w}_k^H \in \mathbb{H}_+^{MN_t}$ | `math_derivation.tex` line 89 |
| $\mat{S}_p$ | Dedicated sensing covariance for target $p$, $\mat{S}_p \in \mathbb{H}_+^{MN_t}$ | `math_derivation{,_en}.tex` §1 |
| $\mat{R}_X$ | Total transmit covariance, $\mat{R}_X \triangleq \sum_{k=1}^{K} \mat{W}_k + \sum_{p=1}^{P} \mat{S}_p$ | `math_derivation{,_en}.tex` §2-A |
| $\mat{A}_k$ | $\mat{A}_k \triangleq \frac{1}{\gamma_k} \mat{W}_k - \sum_{j \neq k} \mat{W}_j - \sum_{p} \mat{S}_p$ | `math_derivation{,_en}.tex` §2-C |
| $\hat{\vect{h}}_k$ | Estimated channel for user $k$ | `math_derivation.tex` line 63 |
| $\Delta\vect{h}_k$ | Channel error, $\|\Delta\vect{h}_k\| \leq \epsilon_h \|\hat{\vect{h}}_k\|$ | `math_derivation.tex` line 63, 116 |
| $\epsilon_h$ | Channel error bound (relative to $\|\hat{\vect{h}}_k\|$) | `math_derivation.tex` line 63 |
| $\mu_k$ | S-Procedure slack, $\mu_k \geq 0$ | `math_derivation.tex` line 130–132 |
| $\sigma_c^2$ | Communication noise power (per receive antenna) | `math_derivation.tex` line 63 |
| $\sigma_s^2$ | Sensing noise power | `math_derivation.tex` line 53 |
| $\gamma_S^{\text{PoD}}$ | Probability-of-Detection threshold | `math_derivation.tex` line 53 |
| $\Gamma_{\text{Track},p}$ | PCRB tracking threshold for target $p$ | `math_derivation.tex` line 54 |
| $P_{\max}$ | Per-AP peak power budget | `math_derivation.tex` line 56 |
| $\vect{g}_p$ | Sensing steering vector toward target $p$ | `math_derivation.tex` line 53 |
| $\mat{J}_p^{\text{data}}$ | Data FIM for target $p$, $\mat{J}_p^{\text{data}}(\mat{S}_p) = \frac{2}{\sigma_s^2} \Real\{\mat{D}_p^H \mat{S}_p \mat{D}_p\} \in \mathbb{H}_+^{N_\theta \times N_\theta}$ (affine in $\mat{S}_p$ only) | `math_derivation{,_en}.tex` §1 (`eq:clustered-fim`, `eq:fim-original`) |
| $\mat{D}_p$ | Effective target-response derivative, $\mat{D}_p = \partial \vect{g}_p / \partial \boldsymbol{\theta}_p \in \mathbb{C}^{MN_t \times N_\theta}$ | `math_derivation{,_en}.tex` §1 |
| $\mat{M}_p$ | PCRB auxiliary matrix, $\mat{M}_p \succeq (\mat{J}_p^{\text{data}})^{-1}$ via the Schur LMI | `math_derivation{,_en}.tex` §2-D |
| $b_{mp}$ | AP--target association (admission indicator), binary relaxed to $[0,1]$ + DC penalty | `math_derivation{,_en}.tex` §1, §2-E |
| $\mat{E}_m$ | Per-AP antenna selection matrix (block diagonal indicator on $\mat{R}_X$ or $\mat{S}_p$) | `math_derivation{,_en}.tex` §1 |
| $\gamma_k$ | Per-user SINR target | `math_derivation.tex` line 63 |
| $K, M, N_t, P$ | Cardinalities: users, APs, antennas/AP, targets | `math_derivation.tex` line 48, 51 |
| $\mathcal{M}, \mathcal{K}, \mathcal{P}$ | Index sets of APs, users, targets | `math_derivation.tex` line 48 |

### LaTeX command conventions (must be reused, not reinvented)

`math_derivation.tex` and `math_derivation_en.tex` both define:

```latex
\newcommand{\vect}[1]{\mathbf{#1}}
\newcommand{\mat}[1]{\mathbf{#1}}
\newcommand{\norm}[1]{\|#1\|_2}
\newcommand{\tr}{\text{tr}}
\newcommand{\Real}{\text{Re}}
```

When extending any of these files (or `convex_block_*.tex`, which use the same commands), **reuse these commands** rather than introducing new notation. `convex_block_*.tex` does not redefine them — it relies on whatever preamble the embedding master document provides. If you extract `convex_block_*.tex` as a standalone, copy this preamble block.

---

## 6. The two "USER PROFILE" strict preferences — locked in here

### Per-AP power (NOT sum-power)

The repo uses **per-AP peak power** $P_{\max}$ as the hard constraint. The relevant constraint is **(P3-C4'b)**: $\tr(\mat{E}_m \mat{R}_X) \leq P_{\max}$ for all $m$, with $\mat{E}_m$ the per-AP selection matrix and $\mat{R}_X = \sum_k \mat{W}_k + \sum_p \mat{S}_p$. Do not switch to sum-power $\tr(\mat{R}_X) \leq M P_{\max}$ even informally. The association gate (P3-C4'a), $\tr(\mat{E}_m \mat{S}_p) \leq P_{\max} b_{mp}$, is a separate per-(AP, target) authorization and must not be merged into (P3-C4'b).

### S-Procedure LMI form — intermediate form is canonical

The **(P3-C1)** LMI is written with the $\|\hat{\vect{h}}_k\|^2$ factor (the "intermediate" form):

$$
\begin{bmatrix} \mat{A}_k + \mu_k \mat{I} & \mat{A}_k \hat{\vect{h}}_k \\[1.2ex] \hat{\vect{h}}_k^H \mat{A}_k & \hat{\vect{h}}_k^H \mat{A}_k \hat{\vect{h}}_k - \sigma_c^2 - \mu_k \epsilon_h^2 \|\hat{\vect{h}}_k\|^2 \end{bmatrix} \succeq \mat{0},
\qquad \mat{A}_k = \frac{1}{\gamma_k} \mat{W}_k - \sum_{j \neq k} \mat{W}_j - \sum_{p} \mat{S}_p
$$

with $\mu_k \epsilon_h^2 \|\hat{\vect{h}}_k\|^2$ in the lower-right entry. This is the canonical form: it is valid for arbitrary (non-normalized) channel estimates, and the MATLAB implementation (`sim/matlab/solve_p3_sca_t.m`) relies on it because its channels are scaled to an SNR operating point, not to $\|\hat{\vect{h}}_k\| = 1$. The short form without the $\|\hat{\vect{h}}_k\|^2$ factor is equivalent **iff** $\|\hat{\vect{h}}_k\| = 1$ and should not be used in code. Note also that $\mat{A}_k$ now subtracts the aggregate dedicated sensing covariance $\sum_p \mat{S}_p$ (UE interference); dropping that term makes the LMI strictly weaker and is the P0 validator bug class fixed in `validate_solution.m`.

---

## 7. Convexification chain (current structure) — what each step does

Source: `math_derivation{,_en}.tex` §2-A…§2-F and §3.

| Step | What it removes | How | Equivalence |
| --- | --- | --- | --- |
| §A | Lifting (P1 → P2) | $\vect{w}_k \vect{w}_k^H = \mat{W}_k$ | Strictly equivalent (rank-1 ensures bijection) |
| §B | Rank-1 (NC4) | Drop $\rank(\mat{W}_k) = 1$, keep PSD; recovered by the rank penalty $\eta_{\text{rank}} \sum_k \rho(\mat{W}_k)$ in §3 | **Lower bound**; no general tightness theorem for cell-free robust case |
| §C-1 | SINR fraction (NC1) | Cross-multiply (denominator > 0) to quadratic form | Strictly equivalent (provided denominator > 0) |
| §C-2 | Worst-case comm SINR (NC2) | S-Procedure lifting to LMI with slack $\mu_k$ | **If-and-only-if** on the norm ball (Boyd–Vandenberghe §4.3, Thm 4.1, ball-outside convention); no power-margin loss |
| §D | PCRB matrix inverse (NC6) | Schur complement: $\mat{M}_p \succeq (\mat{J}_p^{\text{data}}(\mat{S}_p))^{-1}$ as an LMI + $\tr(\mat{M}_p) \leq \Gamma_{\text{Track},p}$ | Strictly equivalent (lossless) |
| §E | Binary association (NC3) | Box relaxation (P3-C9) + DC penalty $\eta_b \sum_{m,p} (b_{mp} - b_{mp}^2)$, jointly linearized with the rank penalty in §3; path-loss top-$N_{\text{req}}$ sort only as cold start | Penalty heuristic; monotone SCA objective, KKT stationary point; no global optimality guarantee |
| §F | Consolidation | Collect into (P3) | — |

### Tightness caveats (§B and §E)

- **§B (SDR)**: For our per-user SINR + cell-free cooperative + robust problem, no general SDR tightness theorem applies. The multicast $G_k \leq 2$ condition and the robust MISO $N_t \leq 2$ condition do not transfer. When $\rank(\mat{W}_k^*) > 1$ at the SDR solution, **Gaussian randomization** with $L = 100$–$1000$ candidates is the engineering fix. No worst-case tightness bound.
- **§E (DC penalty for $b$)**: the penalty drives $b_{mp} \to \{0,1\}$ only for sufficiently large $\eta_b$ (see `thm:sca-conv`); the MATLAB pipeline additionally rounds to exactly-$N_{\text{req}}$ binary candidates and re-solves with $b$ fixed, accepting a candidate only after physical-feasibility validation (`sim/matlab/baseline_alg2.m`). The old Step 7 outer heuristic (fixed AP service set by path-loss sort) is superseded and survives only as a cold-start generator.

---

## 8. Open issues (carried forward from 2026-06-30 audit)

1. **Dead `??` cross-references in `convex_block_*.tex`**: Resolved in commit `5c84426` (soft-reset in `19ff970`, then `convex_block_{en,zh}.tex` deleted in `19ff970`). The `eq:p3c1`–`eq:p3c7` labels are now defined in both `math_derivation{,_en}.tex` (§6) and in any future re-introduction of `convex_block_*` files. The recommended naming `eq:p3c1`–`eq:p3c7` (matches the inline `(P3-Cx)` tag) was applied.

2. **`sec:problem` reference in `convex_block_*.tex` line 6**: Resolved by the deletion of `convex_block_*.tex` in `19ff970`. The replacement anchor `sec:p1` is now defined in both `math_derivation{,_en}.tex` §1 and is the canonical anchor for "Original Problem" going forward.

3. **`math_derivation_en.tex` self-reference `sec:convexification`**: Defined and used only within that file; works. No issue.

4. **Channel normalization not stated explicitly**: **Resolved 2026-07.** The final (P3-C1) now carries the $\|\hat{\vect{h}}_k\|^2$ term (intermediate form is canonical; see §6), so no normalization assumption is needed anywhere.

5. **Label convention for the lifted matrices**: `math_derivation.tex` line 89 writes $\mat{W}_k = \vect{w}_k \vect{w}_k^H$ but $\vect{w}_k$ is the *stacked* beam (line 66), not the per-AP beam. The per-AP $\vect{w}_{m,k}$ are the original (P1) variables. The connection (P1) $\vect{w}_{m,k} \to \vect{w}_k \to \mat{W}_k$ is correct but spans two forms (P1 and P2). A future revision could include a one-line summary table for the variable progression.
