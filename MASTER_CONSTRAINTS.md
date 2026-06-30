# Master Constraints Table — ISAC Research

**Status**: Human-readable reference. Does NOT participate in LaTeX compilation.
**Authoritative source**: `math_derivation.tex` (中文, 220 lines) and `math_derivation_en.tex` (English, 217 lines). All formula content in this table is transcribed from those two files. The four `report*.tex` main-paper drafts (deleted 2026-06-30) are no longer a source.
**Audience**: Anyone who needs to look up the exact form, label, or physical meaning of a P1/P2/P3 constraint without re-reading the full derivation.

---

## 1. The three problems (P1 → P2 → P3) — the story in one paragraph

`math_derivation.tex` walks the problem through three equivalent / relaxed forms:

- **(P1)** — Original problem with per-AP beamforming vectors $\vect{w}_{m,k}$, per-AP sensing covariance $\mat{Z}_m$, and binary AP selection $b_{mp} \in \{0,1\}$. (math_derivation.tex §原始问题, line 48–59.)
- **(P2)** — Lifted covariance form. Replace $\vect{w}_k$ with $\mat{W}_k = \vect{w}_k \vect{w}_k^H$, and stack per-AP $\mat{Z}_m$ into a single $\mat{Z} \in \mathbb{H}_+^{MN_t}$. Adds rank-1 constraint (P2-C7). **Strictly equivalent** to (P1) by the lifting $\vect{w} \leftrightarrow \vect{w}\vect{w}^H$. (math_derivation.tex §协方差提升形式, line 73–85.)
- **(P3)** — Convex SDP after dropping rank-1 (Step 2 SDR) and AP-selection (Step 7 heuristic). Introduces S-Procedure slack $\mu_k \geq 0$ to handle the worst-case channel error $\|\Delta\vect{h}_k\| \leq \epsilon_h \|\hat{\vect{h}}_k\|$. **Lower bound**: $P_{\text{P3}}^* \leq P_{\text{P2}}^* = P_{\text{P1}}^*$. (math_derivation.tex §最终凸 SDP 问题, line 171–182.)

---

## 2. (P3) — Final convex SDP (the master form)

Decision variables: $\mat{W}_k \in \mathbb{H}_+^{MN_t}$ (per-user beamforming covariance), $\mat{Z} \in \mathbb{H}_+^{MN_t}$ (sensing covariance), $\mu_k \geq 0$ (S-Procedure slack).

$$
\begin{aligned}
\min_{\{\mat{W}_k\}, \mat{Z}, \{\mu_k\}} \quad & \sum_{k=1}^{K} \tr(\mat{W}_k) + \tr(\mat{Z}) \tag{P3} \\
\text{s.t.} \quad & \begin{bmatrix} \mat{A}_k + \mu_k \mat{I} & \mat{A}_k \hat{\vect{h}}_k \\[1.5ex] \hat{\vect{h}}_k^H \mat{A}_k & \hat{\vect{h}}_k^H \mat{A}_k \hat{\vect{h}}_k - \sigma_c^2 - \mu_k \epsilon_h^2 \end{bmatrix} \succeq \mat{0}, \quad \forall k \tag{P3-C1} \\
& \tr(\vect{g}_p \vect{g}_p^H \mat{Z}) \geq \gamma_S^{\text{PoD}} \sigma_s^2, \quad \forall p \tag{P3-C2} \\
& \tr(\mat{F}_p \mat{R}_X) \geq \Gamma_{\text{Track},p}, \quad \forall p \tag{P3-C3} \\
& \tr(\mat{E}_m \mat{R}_X) \leq P_{\max}, \quad \forall m \tag{P3-C4} \\
& \mat{W}_k \succeq \mat{0}, \quad \forall k \tag{P3-C5} \\
& \mat{Z} \succeq \mat{0} \tag{P3-C6} \\
& \mu_k \geq 0, \quad \forall k \tag{P3-C7}
\end{aligned}
$$

> **Source of truth**: `math_derivation.tex` line 171–182, verbatim. `math_derivation_en.tex` carries the same form (English) with identical `(P3-Cx)` tags.

### Per-constraint reference card

| Tag | Type | Variables | Physical meaning | Source in `math_derivation.tex` |
| --- | --- | --- | --- | --- |
| `(P3-C1)` | LMI ($MN_t{+}1$ dim.) | $\mat{W}_k, \mu_k$ | Communication worst-case SINR under $\|\Delta\vect{h}_k\| \leq \epsilon_h \|\hat{\vect{h}}_k\|$ | line 175 (final); intermediate S-Procedure form (with $\|\hat{\vect{h}}_k\|^2$ term) at line 132 |
| `(P3-C2)` | Linear inequality | $\mat{Z}$ | Sensing detection probability (PoD criterion at $\gamma_S^{\text{PoD}}$) | line 176; derivation line 153 |
| `(P3-C3)` | Linear inequality | $\mat{R}_X$ | Tracking accuracy upper bound (PCRB at $\Gamma_{\text{Track},p}$) | line 177; derivation line 147 |
| `(P3-C4)` | Linear inequality | $\mat{R}_X$ | Per-AP peak power | line 178; derivation line 159 |
| `(P3-C5)` | PSD cone | $\mat{W}_k$ | SDR relaxation of $\rank(\mat{W}_k) = 1$ | line 179 |
| `(P3-C6)` | PSD cone | $\mat{Z}$ | Sensing covariance PSD | line 180 |
| `(P3-C7)` | Non-negative orthant | $\mu_k$ | S-Procedure slack | line 181 |

---

## 3. (P2) — Lifted but still non-convex (one step before SDR)

(P2) is what (P3) relaxes. The differences are exactly the two constraints that SDR drops:

$$
\begin{aligned}
\text{(P2) = (P3) with two extra constraints: } & \\
& \rank(\mat{W}_k) = 1, \quad \forall k \tag{P2-C7} \\
& b_{mp} \in \{0,1\}, \quad \forall m, p \tag{P2-C8}
\end{aligned}
$$

`math_derivation.tex` line 73–85. (P2) is **strictly equivalent** to (P1) because the rank-1 constraint makes the lifting $\vect{w} \leftrightarrow \mat{w}\mat{w}^H$ a bijection.

---

## 4. (P1) — Original problem

```text
min    Σ_m ( Σ_k ‖w_{m,k}‖² + tr(Z_m) )
s.t.   (P1-C1)  worst-case comm SINR   ≥ γ_k     ∀k
       (P1-C2)  sensing SINR            ≥ γ_S^PoD  ∀p
       (P1-C3)  PCRB                    ≥ Γ_p     ∀p
       (P1-C4)  Σ_m b_{mp} = N_req      ∀p
       (P1-C5)  Σ_k ‖w_{m,k}‖² + tr(Z_m) ≤ P_max  ∀m
       (P1-C6)  Z_m ⪰ 0                  ∀m
       (P1-C7)  b_{mp} ∈ {0,1}           ∀m, p
```

> `math_derivation.tex` line 48–59 (objective and P1-C2 through P1-C7) plus line 63 (P1-C1, the worst-case SINR definition).

### (P1) → (P3) constraint correspondence

Source: `math_derivation.tex` line 188–202 (the correspondence table at end of §最终凸 SDP 问题).

| (P1) | (P3) | Transform | Convexity |
| --- | --- | --- | --- |
| (P1-C1) comm SINR | (P3-C1) LMI | Steps 1, 2, 3, 4 (lifting + SDR + S-Procedure + fraction linearization) | LMI (convex) |
| (P1-C2) sensing SINR | (P3-C2) linear | Steps 1, 2, 5b | linear (convex) |
| (P1-C3) PCRB | (P3-C3) linear | Steps 1, 2, 5a | linear (convex) |
| (P1-C4) AP selection | (P3-C4) on fixed set | Step 7 | linear (convex) |
| (P1-C5) per-AP power | (P3-C4) linear | Steps 1, 6 | linear (convex) |
| (P1-C6) PSD | (P3-C6) PSD | identity | PSD cone (convex) |
| (P1-C7) binary | dropped | Step 7 | heuristic |
| (P1-C8) rank-1 | (P3-C5) PSD | Step 2 | PSD cone (convex, **not tight**) |

> Note: the original numbering `(P1-C1) … (P1-C7)` in `math_derivation.tex` line 49–58 jumps from (P1-C2) sensing to (P1-C3) PCRB without an explicit (P1-C1) constraint in the equation block. (P1-C1) is defined separately at line 63 as the worst-case SINR. The correspondence table on line 188–202 adds an extra "(P1-C8) rank-1" entry to align with the lifted (P2) form. This is consistent within `math_derivation.tex`; the rank-1 is implicit in the lifting (P1 → P2), not stated as a separate constraint in (P1).

---

## 5. Symbol conventions (enforced via `\newcommand` in both math_derivation files)

| Symbol | Definition | First defined in |
| --- | --- | --- |
| $\mat{W}_k$ | Per-user beamforming covariance, $\mat{W}_k = \vect{w}_k \vect{w}_k^H \in \mathbb{H}_+^{MN_t}$ | `math_derivation.tex` line 89 |
| $\mat{Z}$ | Stacked sensing covariance, $\mat{Z} \in \mathbb{H}_+^{MN_t}$ | `math_derivation.tex` line 90 (via $\mat{R}_X$ def) |
| $\mat{R}_X$ | Total transmit covariance, $\mat{R}_X \triangleq \sum_{k=1}^{K} \mat{W}_k + \mat{Z}$ | `math_derivation.tex` line 90 |
| $\mat{A}_k$ | $\mat{A}_k \triangleq \frac{1}{\gamma_k} \mat{W}_k - \sum_{j \neq k} \mat{W}_j$ | `math_derivation.tex` line 126, 184 |
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
| $\mat{F}_p$ | $\mat{F}_p = \frac{2}{\sigma_s^2} \Real\{\nabla_{\boldsymbol{\theta}_p} \vect{g}_p^H \cdot \nabla_{\boldsymbol{\theta}_p}^H \vect{g}_p\} \in \mathbb{H}_+^{MN_t}$ | `math_derivation.tex` line 143 |
| $\mat{E}_m$ | Per-AP antenna selection matrix (block diagonal indicator on $\mat{R}_X$) | `math_derivation.tex` line 91 |
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

The repo uses **per-AP peak power** $P_{\max}$ as the hard constraint. The relevant constraint is **(P3-C4)**: $\tr(\mat{E}_m \mat{R}_X) \leq P_{\max}$ for all $m$, with $\mat{E}_m$ the per-AP selection matrix. Do not switch to sum-power $\tr(\mat{R}_X) \leq M P_{\max}$ even informally. Source: `math_derivation.tex` line 56 (P1-C5) and line 178 (P3-C4).

### S-Procedure LMI form — exact, with the $\|\hat{\vect{h}}_k\|^2$ term absorbed via normalization

The **(P3-C1)** form as written in the final problem (line 175) is:

$$
\begin{bmatrix} \mat{A}_k + \mu_k \mat{I} & \mat{A}_k \hat{\vect{h}}_k \\[1.5ex] \hat{\vect{h}}_k^H \mat{A}_k & \hat{\vect{h}}_k^H \mat{A}_k \hat{\vect{h}}_k - \sigma_c^2 - \mu_k \epsilon_h^2 \end{bmatrix} \succeq \mat{0}
$$

This **omits** the $\|\hat{\vect{h}}_k\|^2$ term. It is mathematically equivalent to the "intermediate" S-Procedure form written at line 132:

$$
\begin{bmatrix} \mat{A}_k + \mu_k \mat{I} & \mat{A}_k \hat{\vect{h}}_k \\[1.2ex] \hat{\vect{h}}_k^H \mat{A}_k & \hat{\vect{h}}_k^H \mat{A}_k \hat{\vect{h}}_k - \sigma_c^2 - \mu_k \epsilon_h^2 \|\hat{\vect{h}}_k\|^2 \end{bmatrix} \succeq \mat{0}
$$

**iff** the channel is normalized: $\|\hat{\vect{h}}_k\| = 1$. `math_derivation.tex` does not state this normalization explicitly in the preamble, but the absence of the $\|\hat{\vect{h}}_k\|^2$ term in the final (P3-C1) form (and the prose discussion of the S-Procedure at line 130) implies $\|\hat{\vect{h}}_k\| = 1$ is adopted. **If a future revision drops this normalization, the intermediate form (line 132) is the canonical one — not the (P3-C1) form.**

This is the resolution of the **P3-C1 LMI inconsistency** flagged in the 2026-06-30 audit: the removed `report.tex` (zh) used the intermediate form; the removed English `report_*.tex` used the final form. `math_derivation.tex` contains both, and the final form is canonical under $\|\hat{\vect{h}}_k\| = 1$.

---

## 7. Convexification chain (Steps 1–7) — what each step does

Source: `math_derivation.tex` line 108–163 (逐步凸化) and line 193–200 (correspondence table).

| Step | What it removes | How | Equivalence |
| --- | --- | --- | --- |
| 1 | Lifting (P1 → P2) | $\vect{w}_k \vect{w}_k^H = \mat{W}_k$ | Strictly equivalent (rank-1 ensures bijection) |
| 2 | Rank-1 (NC4) | Drop $\rank(\mat{W}_k) = 1$, keep PSD | **Lower bound**; no general tightness theorem for cell-free robust case |
| 3 | Worst-case comm SINR (NC2) | S-Procedure lifting to LMI with slack $\mu_k$ | Conservative; trades ≈1.75 dB power margin for closed-form LMI |
| 4 | SINR fraction (NC1) | Cross-multiply (denominator > 0) to quadratic form | Strictly equivalent (provided denominator > 0) |
| 5a | PCRB matrix inverse (NC6) | Use cyclic property of trace; FIM is affine in $\mat{R}_X$ | Strictly equivalent under Assumption 1 (current-slot $\nabla \vect{g}_p$ known) |
| 5b | Sensing SINR (NC1, sensing part) | Cross-multiply; $\mat{Z} \succeq 0$ guarantees non-negative numerator | Strictly equivalent |
| 6 | Per-AP power (NC5 bilinear) | Stack per-AP into $\mat{R}_X$, use $\mat{E}_m$ selector | Strictly equivalent |
| 7 | Binary AP selection (NC3) | Two-step: outer sort by large-scale fading → inner SDP on fixed set | Heuristic; no optimality guarantee |

### Tightness caveats (Step 2 and Step 7)

- **Step 2 (SDR)**: For our per-user SINR + cell-free cooperative + robust problem, no general SDR tightness theorem applies. The multicast $G_k \leq 2$ condition and the robust MISO $N_t \leq 2$ condition do not transfer. When $\rank(\mat{W}_k^*) > 1$ at the SDR solution, **Gaussian randomization** with $L = 100$–$1000$ candidates is the engineering fix. No worst-case tightness bound.
  - Source: `math_derivation.tex` line 114.
- **Step 7 (AP heuristic)**: The outer AP-set selection by large-scale-fading sort is heuristic. The selected set may not be the true optimum (the true problem is NP-hard, a $K$-medoid variant). The inner SDP on a fixed AP set is still convex and tight. The combination can be either a lower bound or an incomparable quantity depending on the actual AP set chosen.
  - Source: `math_derivation.tex` line 163, 215.

---

## 8. Open issues (carried forward from 2026-06-30 audit)

1. **Dead `??` cross-references in `convex_block_*.tex`**: Resolved in commit `5c84426` (soft-reset in `19ff970`, then `convex_block_{en,zh}.tex` deleted in `19ff970`). The `eq:p3c1`–`eq:p3c7` labels are now defined in both `math_derivation{,_en}.tex` (§6) and in any future re-introduction of `convex_block_*` files. The recommended naming `eq:p3c1`–`eq:p3c7` (matches the inline `(P3-Cx)` tag) was applied.

2. **`sec:problem` reference in `convex_block_*.tex` line 6**: Resolved by the deletion of `convex_block_*.tex` in `19ff970`. The replacement anchor `sec:p1` is now defined in both `math_derivation{,_en}.tex` §1 and is the canonical anchor for "Original Problem" going forward.

3. **`math_derivation_en.tex` self-reference `sec:convexification`**: Defined and used only within that file; works. No issue.

4. **Channel normalization not stated explicitly**: `math_derivation.tex` does not write "assume $\|\hat{\vect{h}}_k\| = 1$" anywhere, but the (P3-C1) form implies it. Future work: state the normalization in the preamble of both `math_derivation` files, or carry the $\|\hat{\vect{h}}_k\|^2$ term through to the final (P3-C1) form to avoid the implicit assumption.

5. **Label convention for the lifted matrices**: `math_derivation.tex` line 89 writes $\mat{W}_k = \vect{w}_k \vect{w}_k^H$ but $\vect{w}_k$ is the *stacked* beam (line 66), not the per-AP beam. The per-AP $\vect{w}_{m,k}$ are the original (P1) variables. The connection (P1) $\vect{w}_{m,k} \to \vect{w}_k \to \mat{W}_k$ is correct but spans two forms (P1 and P2). A future revision could include a one-line summary table for the variable progression.
