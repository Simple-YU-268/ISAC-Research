# S-Procedure Working Notes (Deprecated)

**Status**: v1.0 → v1.1 (2026-07-07). The "1.75 dB gap" claim made here was
based on a *sufficient-only* S-Procedure interpretation that is **not the
form used in `math_derivation.tex`**. The form in `math_derivation.tex`
(S3.1) uses the ball-outside description convention and is if-and-only-if
by Boyd-Vandenberghe, 2004, §4.3, Theorem 4.1.

The content below is preserved as a working derivation that **does not
match the current thesis**. It is kept for archaeological purposes only
and should not be propagated to CONVEXIFICATION_CHAIN.md, MASTER_CONSTRAINTS.md,
or any external document without independent numerical verification.

---

## Original Derivation (v1.0, 2026-06-30 — superseded)

> This section is preserved verbatim for the record. It uses the
> ball-inside description and concludes the S-Procedure is sufficient-only.
> If you adopt the ball-outside description (as `math_derivation.tex`
> does), the gap claim collapses to zero and the LMI becomes if-and-only-if.
> The numerical 1.75 dB estimate below has **never been independently
> Monte-Carlo verified** and should be treated as a planning figure, not
> a measured result.

### Setup

We work under the channel normalization $\|\hat{\vect{h}}_k\|_2 = 1$.

The worst-case robust SINR constraint is:

$$
\forall \Delta\vect{h}_k \in \mathcal{B}_\epsilon :
\frac{|(\hat{\vect{h}}_k + \Delta\vect{h}_k)^H \vect{w}_k|^2}
     {\sum_{j \neq k} |(\hat{\vect{h}}_k + \Delta\vect{h}_k)^H \vect{w}_j|^2 + \sigma_c^2}
\geq \gamma_k \tag{*}
$$

The actual worst-case SINR is

$$
\gamma_k^{\text{true}} = \min_{\Delta\vect{h}_k \in \mathcal{B}_\epsilon}
\frac{|(\hat{\vect{h}}_k + \Delta\vect{h}_k)^H \vect{w}_k|^2}
     {\sum_{j \neq k} |(\hat{\vect{h}}_k + \Delta\vect{h}_k)^H \vect{w}_j|^2 + \sigma_c^2}
$$

Define $\mat{A}_k \triangleq \frac{1}{\gamma_k} \mat{W}_k - \sum_{j \neq k} \mat{W}_j$
where $\mat{W}_j = \vect{w}_j \vect{w}_j^H$. The S-Procedure (ball-inside
description) asserts:

$$
\exists \mu_k \geq 0 : \begin{bmatrix}
\mat{A}_k + \mu_k \mat{I} & \mat{A}_k \hat{\vect{h}}_k \\
\hat{\vect{h}}_k^H \mat{A}_k & \hat{\vect{h}}_k^H \mat{A}_k \hat{\vect{h}}_k - \sigma_c^2 - \mu_k \epsilon_h^2
\end{bmatrix} \succeq \mat{0} \quad \Longrightarrow \quad (*)
$$

The implication goes one way: satisfying the LMI guarantees $(*)$, but
$(*)$ can hold even when the LMI does not.

### Loss functional

Define the **LMI-feasible** $\gamma$ and **true** $\gamma$:

$$
\gamma_k^{\text{LMI}} = \text{largest } \gamma \text{ such that the LMI is feasible}
$$
$$
\gamma_k^{\text{true}} = \text{true worst-case SINR for the given } \mat{W}, \mat{Z}, \mu
$$

The loss is $\gamma_k^{\text{LMI}} - \gamma_k^{\text{true}}$.

### Closed-form upper bound in the interference case

For $K \geq 2$, the gap arises because the interference term is a sum
of quadratic forms. The standard remedy is to bound the interference
**uniformly** by its worst case on the ball $\mathcal{B}_\epsilon$:

$$
\max_{\Delta\vect{h} \in \mathcal{B}_\epsilon} \sum_{j \neq k} |(\hat{\vect{h}} + \Delta\vect{h})^H \vect{w}_j|^2
\;\leq\; \sum_{j \neq k} \max_{\Delta\vect{h} \in \mathcal{B}_\epsilon} |(\hat{\vect{h}} + \Delta\vect{h})^H \vect{w}_j|^2
$$

The right-hand side decouples into $K-1$ independent S-Procedures,
one per interfering user. Each contributes a slack $\mu_{k,j} \geq 0$.

**Per-interferer closed form.** In the high-SNR limit and at the
worst-case over beamforming orientations,

$$
\delta\gamma_{k,j}^{\text{SP}} \;\leq\; \frac{\epsilon_h^2 \|\vect{w}_j\|^2}
{\sum_{\ell \neq k, j} |(\hat{\vect{h}}_k + \Delta\vect{h}_{k,j}^\star)^H \vect{w}_\ell|^2}
$$

For the cell-free geometry, the worst case is when all interferers
align with the channel error direction:

$$
\delta\gamma_k^{\text{SP}} \;\leq\; (K - 1) \cdot \frac{\epsilon_h^2 \cdot \|\vect{w}_j\|^2}
{\|\hat{\vect{h}}_k\|^2 \cdot \sum_{\ell \neq k, j} \|\vect{w}_\ell\|^2}
$$

In the cell-free operating regime ($\|\vect{w}_j\|^2 \approx P_{\max}/K$):

$$
\eta_h = \frac{\delta\gamma_k^{\text{SP}}}{\gamma_k} \approx \frac{\epsilon_h^2 \cdot (K-1)}{K \cdot \gamma_k}
$$

The "$\eta_h$" notation in `CONVEXIFICATION_CHAIN.md` §3.3 corresponds
to this ratio.

### The "1.75 dB" planning figure (NOT a measured result)

For the operating point $\epsilon_h = 0.10$ and $\gamma_k = 0$ dB
(SINR target = 1, i.e. unity gain), the high-SNR asymptotic
approximation gives

$$
\eta_h \approx \frac{0.10^2 \cdot (K-1)}{K \cdot 1} \approx 0.01 \cdot \frac{K-1}{K}
$$

For $K = 2$, $\eta_h \approx 0.005$, which is $0.043$ dB. Much
smaller than 1.75 dB. The 1.75 dB is the **finite-SNR** correction
from $\sigma_c^2$ in the denominator.

### Important caveat (added in v1.1)

The "1.75 dB" figure should **not** be propagated into the thesis or
any simulation without an actual Monte-Carlo evaluation. The thesis's
current position (post v12) is that the (S3.1) LMI is if-and-only-if
equivalent to the original worst-case constraint, with no gap, by the
ball-outside convention and Boyd-Vandenberghe Theorem 4.1.

If a future variant of the LMI becomes sufficient-only (e.g., by
explicitly taking the ball-inside as $f_1$, or by further approximations),
the gap analysis above would apply and the 1.75 dB figure would be a
candidate — but this is reserved for future work.

### Pointer to companion documents

- `math_derivation.tex` §IV-C — the (S3.1) LMI block in context.
- `CONVEXIFICATION_CHAIN.md` §3.3 — updated to "strictly equivalent" (v12 onward).
- `MASTER_CONSTRAINTS.md` §6 — channel normalization and the (P3-C1) form.
- `PAPER_OUTLINE_IEEE.md` §III.4 — tightness analysis section in the paper outline.
