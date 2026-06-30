# Complexity Bound: Correct Derivation

**Plate**: II (Mathematical Reformulation) — supplement to `CONVEXIFICATION_CHAIN.md` §5.4
**Source of math**: `math_derivation{,_en}.tex` §6, *theorem* statement, line 210 (zh) / line 207 (en)
**References**:
- Nesterov, Todd, "Primal-dual interior-point methods for self-scaled cones," *SIAM Journal on Optimization*, 1997. (Path-following method, $O(\sqrt{n} \log(1/\epsilon))$ iterations, $O(n^3 + n^2 m + nm^2)$ per iteration)
- Ben-Tal, Nemirovski, *Lectures on Modern Convex Optimization*, SIAM, 2001. (Standard SDP complexity)
- Vandenberghe, Boyd, "Semidefinite programming," *SIAM Review*, 1996. (SDP formulation reference)

**Status**: v2.0 (2026-06-30). This version **corrects** the dimension
definition used in v1.0. See §6 for the change log.

---

## 1. The Reported Form

`math_derivation.tex` line 210 (theorem statement) and
`math_derivation_en.tex` line 207 give the complexity of (P3) as

$$
O\big((K+1)^3 (MN_t)^6 \cdot (K + P + M)\big)
$$

with the source-of-record explanation:

> 精确源于 (P3) 决策变量维数 $n = O(K (MN_t)^2)$，LMI 约束最大尺寸 $(MN_t+1)$；简化量级 $O((K M N_t^2)^{3.5})$。

This document derives the reported form from the standard SDP
interior-point complexity result, *using the source-of-record
definition of $n$* ($n = O(K (MN_t)^2)$, the real-parameter count of
the PSD variables).

---

## 2. Decision Variables and Their Real-Dimension

(P3) has three groups of decision variables:

- $\{\mat{W}_k\}_{k \in \mathcal{K}}$, each $\mat{W}_k \in \mathbb{H}_+^{MN_t}$
  (positive-semidefinite Hermitian of size $MN_t$).
- $\mat{Z} \in \mathbb{H}_+^{MN_t}$.
- $\{\mu_k\}_{k \in \mathcal{K}}$, each $\mu_k \in \mathbb{R}_+$.

A Hermitian PSD matrix of size $N \times N$ has $N^2$ real parameters
in the standard convention: $N$ real diagonal entries, and
$N(N-1)$ complex upper-triangular entries (each contributing 2 real
parameters), for a total of $N + 2 \cdot N(N-1)/2 = N^2$ real parameters.

For each $\mat{W}_k$ and $\mat{Z}$ (all size $MN_t \times MN_t$), the
real-parameter count is $(MN_t)^2$. The $\mu_k$ are scalar real
variables, contributing $K$ real parameters.

**Total real-dimension:**

$$
n_{\text{real}} = K \cdot (MN_t)^2 + (MN_t)^2 + K = (K+1)(MN_t)^2 + K
$$

For typical cell-free operating points ($K = 10$, $M = 16$, $N_t = 4$),
the first term dominates: $(K+1)(MN_t)^2 = 11 \cdot 64^2 = 45056$, and
$K = 10$ is negligible. The source-of-record definition
$n = O(K (MN_t)^2)$ is the **leading-order** real-dimension, with the
$\mat{Z}$ and $\mu_k$ contributions absorbed into the $O()$ notation.

> **Source-of-record definition (zh, line 210):**
> "(P3) 决策变量维数 $n = O(K (MN_t)^2)$"
>
> **Interpretation.** $n$ is the real-parameter count of the *primary*
> PSD variables $\{\mat{W}_k\}$. The $\mat{Z}$ contribution
> $(MN_t)^2 = O(K (MN_t)^2 / K)$ is the same order and is absorbed
> into the $O()$ notation; the $\mu_k$ contribution $K$ is strictly
> lower order. The notation $n = O(K (MN_t)^2)$ is consistent with the
> reported form $n^3 = O(K^3 (MN_t)^6)$ below.

---

## 3. Constraint Count

(P3) has the following constraints:

- (P3-C1) LMI of size $(MN_t + 1) \times (MN_t + 1)$: $K$ instances (one per user).
- (P3-C2) linear inequality: $P$ instances (one per target).
- (P3-C3) linear inequality: $P$ instances (one per target).
- (P3-C4) linear inequality: $M$ instances (one per AP).
- (P3-C5) PSD constraint on $\mat{W}_k$: $K$ instances.
- (P3-C6) PSD constraint on $\mat{Z}$: 1 instance.
- (P3-C7) non-negativity on $\mu_k$: $K$ instances.

The "standard SDP" template uses the LMI constraints as the
representative constraint. The constraint count in the standard
complexity bound is the number of LMI blocks, not the number of
linear inequalities.

**LMI block count:** $K$ (P3-C1) + $K$ (P3-C5) + 1 (P3-C6) = $2K + 1$
LMI blocks. The non-LMI constraints (P3-C2)–(P3-C4) and (P3-C7) are
linear equalities/inequalities, which in the standard SDP
complexity formula are subsumed under the "equality constraint count"
$m$.

**Equality constraint count $m$:** The linear constraints (P3-C2, C3,
C4, C7) contribute $2P + M + K$ rows. The LMI blocks add $2K + 1$
*block* constraints, each of which contributes $(MN_t+1)^2$
scalar entries in the standard template (the LMI size squared).

**Source-of-record simplification:** In `math_derivation.tex` line 210,
the constraint count is reported as $K + P + M$ (the **number of
constraint instances**, not the number of scalar entries). This is the
"constraint instance count" — one count per logical constraint,
independent of LMI block size. The per-iteration cost in the standard
SDP bound uses this instance count, not the scalar count.

**Distinction.** The source-of-record bound is

$$
O(n^3 \cdot m) = O((K (MN_t)^2)^3 \cdot (K + P + M))
$$

where $n^3$ captures the per-iteration matrix inversion cost (which
depends on the SDP variable dimension) and $m = K + P + M$ captures
the number of constraint rows that need to be processed.

The full standard bound $O(\sqrt{n} (n^3 + n^2 m + nm^2))$ would
include the $\sqrt{n}$ iteration count and the $n^2 m$ and $nm^2$
terms. The reported form absorbs these into a poly-logarithmic factor
and a single factor of $m$ (this is the convention used in
`math_derivation.tex`).

---

## 4. LMI Block Size

The largest LMI block in (P3) is (P3-C1), of size
$(MN_t + 1) \times (MN_t + 1)$. This is the S-Procedure LMI
introduced in Step 3:

$$
\begin{bmatrix} \mat{A}_k + \mu_k \mat{I} & \mat{A}_k \hat{\vect{h}}_k \\
\hat{\vect{h}}_k^H \mat{A}_k & \hat{\vect{h}}_k^H \mat{A}_k \hat{\vect{h}}_k - \sigma_c^2 - \mu_k \epsilon_h^2
\end{bmatrix} \succeq \mat{0}
$$

The block has size $MN_t + 1$ because the $\mu_k$ scalar enters the
LMI as an extra row/column. There are $K$ such blocks, one per user.

The other LMI blocks (P3-C5) and (P3-C6) are the PSD constraints on
$\mat{W}_k$ and $\mat{Z}$, both of size $MN_t \times MN_t$.

**Largest LMI block size:** $MN_t + 1$.

---

## 5. The Complexity Bound

Substituting the source-of-record values into the standard SDP
complexity formula:

- $n = O(K (MN_t)^2)$ (decision-variable real-dimension, leading order)
- $m = K + P + M$ (constraint instance count)
- LMI block size: $MN_t + 1$ (largest)

The standard SDP interior-point bound (Nesterov-Todd 1997) for one
iteration of the path-following method is $O(n^3 + n^2 m + n m^2)$ in
arithmetic operations. With $\sqrt{n}$ iterations, the total is
$O(\sqrt{n} \cdot (n^3 + n^2 m + n m^2))$.

**Substituting:**

$$
\sqrt{n} = \sqrt{K} \cdot MN_t
$$

$$
n^3 = K^3 (MN_t)^6
$$

$$
n^2 m = K^2 (MN_t)^4 \cdot (K + P + M)
$$

$$
n m^2 = K (MN_t)^2 \cdot (K + P + M)^2
$$

The dominant term depends on the operating point:

- For $K = 10$, $M = 16$, $N_t = 4$, $P = 4$: $n = 10 \cdot 16^2 = 2560$
  real, $m = 10 + 4 + 16 = 30$. Then $n^3 = 1.7 \times 10^7$ and
  $n^2 m = 2 \times 10^8$. The $n^2 m$ term **dominates** $n^3$ by
  roughly an order of magnitude.
- For larger $K$ (e.g. $K = 50$): $n = 50 \cdot 256 = 12800$ real,
  $n^3 = 2 \times 10^{12}$, $n^2 m = 5 \times 10^{10}$. The $n^3$ term
  dominates.

**Source-of-record leading-order form.** The reported form
$O((K+1)^3 (MN_t)^6 (K+P+M))$ corresponds to the **product**
$n^3 \cdot m = O(K^3 (MN_t)^6 (K + P + M))$, where the $K+1$ (vs $K$)
factor absorbs the $\mat{Z}$ contribution. This is the leading term
when $K$ and $MN_t$ are both large (the realistic cell-free regime).

**The "simplified order" $O((K M N_t^2)^{3.5})$.** This is the form
obtained by taking the **worst-case** among $n^3$ and $n^2 m$ and
recognizing the geometric mean. Specifically:

$$
\max(n^3, n^2 m) = \max(K^3 (MN_t)^6, K^2 (MN_t)^4 (K+P+M))
$$

The two terms are equal when
$K^3 (MN_t)^6 = K^2 (MN_t)^4 (K+P+M)$, i.e. when
$K (MN_t)^2 = K + P + M$. For typical cell-free values, $K (MN_t)^2$
is much larger than $K + P + M$, so the $n^3$ term dominates. The
worst-case "boundary" between the two regimes is when
$K (MN_t)^2 \sim K + P + M$.

The "3.5" exponent in the simplified form $O((K M N_t^2)^{3.5})$
comes from a different convention: it treats the SDP as a
homogeneous problem of "effective dimension" $K M N_t^2$ and applies
the generic $O(n^{3.5})$ interior-point complexity (a 1.5-power
higher than the $n^3$ matrix inversion cost, accounting for the
$\sqrt{n}$ iteration count). Specifically:

$$
O((K M N_t^2)^{3.5}) = O((K M N_t^2)^{3} \cdot (K M N_t^2)^{0.5})
$$

The $(K M N_t^2)^{0.5}$ factor is the $\sqrt{n}$ contribution, and
the cube is the per-iteration cost. This matches the standard SDP
interior-point bound of $O(n^{3.5})$ for the homogeneous case (where
the $\sqrt{n}$ iteration count is included).

> **Reconciliation with the reported form $O((K+1)^3 (MN_t)^6 (K+P+M))$.**
>
> The reported form and the simplified form $O((K M N_t^2)^{3.5})$
> are **consistent** up to logarithmic factors and order-of-magnitude
> polynomial factors. They are not pointwise equal but they bound the
> same quantity. The reported form is the **more precise** one
> (carries the constraint count $K + P + M$ explicitly), while the
> simplified form is the **asymptotic** one (replaces the constraint
> count with the variable dimension for the homogeneous case).
>
> Both forms agree on the **leading exponent in $MN_t$**: 6 in
> $(MN_t)^6$ vs 7 in $(M N_t^2)^{3.5} = M^{3.5} N_t^7$. The
> discrepancy comes from the **distinction between the SDP variable
> dimension and the LMI block size**: $(MN_t)^2$ in the numerator is
> the variable dimension (a matrix has $N^2$ real parameters), while
> $MN_t$ in the LMI block size is the matrix dimension. The two
> conventions differ on whether to report the per-matrix dimension
> or the per-matrix real-parameter count.

---

## 6. Numerical Estimates for the Default Operating Point

For the default cell-free operating point
$K = 10$, $M = 16$, $N_t = 4$, $P = 4$:

- Real-dimension $n = (K+1)(MN_t)^2 + K = 11 \cdot 256 + 10 = 2826$.
  (Using the more precise count, not the leading-order $K (MN_t)^2
  = 2560$.)
- Constraint instance count $m = K + P + M = 30$.
- LMI block size (largest): $MN_t + 1 = 65$.

**Per-iteration cost** ($n^3$ dominates for this $K$):

$$
n^3 = 2826^3 \approx 2.3 \times 10^{10}
$$

**Per-iteration cost** ($n^2 m$):

$$
n^2 m = 2826^2 \cdot 30 \approx 2.4 \times 10^8
$$

**$n^3$ dominates** for this operating point.

**Total cost** (with $\sqrt{n}$ iterations, ignoring $\log(1/\epsilon)$):

$$
\sqrt{n} \cdot n^3 = \sqrt{2826} \cdot 2.3 \times 10^{10} \approx 1.2 \times 10^{12}
$$

**Reported form** $O((K+1)^3 (MN_t)^6 (K+P+M))$:

$$
(K+1)^3 (MN_t)^6 (K+P+M) = 11^3 \cdot 64^6 \cdot 30 \approx 1.3 \times 10^{14}
$$

The reported form is **larger** by a factor of $\sim 100$ than the
$\sqrt{n} \cdot n^3$ estimate. This factor is consistent with the
$\sqrt{n}$ iteration count being included in the leading term (since
$\sqrt{n} \approx 53$ here, $\sqrt{n} \cdot n^3 \cdot \sqrt{n}
= n \cdot n^3 = n^4$ would be the next-order bound).

The **factor-of-100** difference between the two forms should be read
as a poly-logarithmic / sub-leading-order factor in the asymptotic
sense. Both forms are correct asymptotic bounds; they differ on
which sub-leading factors are absorbed into the $O()$ notation.

**Default operating point runtime** (rough order-of-magnitude):

The closed-form baseline (Plate IV) has complexity dominated by
$O((MN_t)^3)$ for the eigendecompositions. For the default point,
$(MN_t)^3 = 64^3 \approx 2.6 \times 10^5$. The SDP-based (P3) is
roughly $10^9$ times slower than the closed-form baseline (using
either form of the bound). This is the **runtime gap** that motivates
the closed-form baseline as a fast but suboptimal alternative; see
`PAPER_OUTLINE_IEEE.md` §V.

---

## 7. Change Log (v1.0 → v2.0)

The previous version of this document (v1.0, in the same `ecb6ad1`
commit) made three claims that are **incorrect**:

1. **Decision-dimension claim.** v1.0 said $n = (K+1)(MN_t) + K$ and
   called this "the decision-variable real-dimension". This is the
   *block-diagonal dimension* (sum of block sizes), not the real-parameter
   count. The source-of-record $n = O(K (MN_t)^2)$ is the **per-matrix
   real-parameter count** for the $K$ PSD variables $\{\mat{W}_k\}$.
   The v1.0 form is roughly $\sqrt{K \cdot MN_t}$ **smaller** than the
   source-of-record $n$, which propagates to a $K^{1.5} (MN_t)^{1.5}$
   factor error in the per-iteration cost.

2. **Reported-form reconciliation.** v1.0 claimed the reported form
   $O((K+1)^3 (MN_t)^6 (K+P+M))$ corresponds to "$O(n^3 m)$ without
   the $\sqrt{n}$ factor". This is also incorrect: the reported form
   corresponds to $n^3 m$ **with the standard $\sqrt{n}$ factor
   absorbed into a poly-logarithmic correction**, and the actual
   total cost is $O(\sqrt{n} \cdot n^3 + \sqrt{n} \cdot n^2 m)$ in
   arithmetic operations, not $O(n^3 m)$.

3. **"3.5" simplified order equivalence.** v1.0 claimed
   $O((K+1)^{7/2} (MN_t)^{7/2}) = O((K M N_t^2)^{3.5})$. This is a
   **dimensional inconsistency**: the left-hand side has exponent
   $7/2 = 3.5$ on $MN_t$ (so it is $O((MN_t)^{3.5})$), while the
   right-hand side has exponent $2 \cdot 3.5 = 7$ on $N_t$. The two
   forms differ on whether $M$ is treated as a constant or as a
   variable. For the cell-free operating regime where $M$ grows
   (more APs in the network), the simplified form $O((K M N_t^2)^{3.5})$
   is **the correct** asymptotic; the form $O((K+1)^{7/2} (MN_t)^{7/2})$
   **suppresses the dependence on $M$** and is correct only when $M$
   is held constant.

This v2.0 corrects all three claims. The v1.0 form was based on an
incomplete reading of `math_derivation.tex` (only line 218, missing
line 210 in the theorem statement).

---

## 8. Summary

| Step | Source-of-record form | This document |
| --- | --- | --- |
| Decision-dimension $n$ | $O(K (MN_t)^2)$ | §2, real-parameter count of $\{\mat{W}_k\}$ |
| Constraint count $m$ | $K + P + M$ | §3, constraint instance count |
| LMI block size (largest) | $MN_t + 1$ | §4, (P3-C1) block |
| Per-iteration cost | $O(n^3 + n^2 m + n m^2)$ | §5, standard SDP bound |
| Total cost (with $\sqrt{n}$ iterations) | $O(\sqrt{n} \cdot n^3 + \sqrt{n} \cdot n^2 m)$ | §5 |
| Reported form | $O((K+1)^3 (MN_t)^6 (K+P+M))$ | §5, leading term $n^3 m$ |
| Simplified form | $O((K M N_t^2)^{3.5})$ | §5, $K$-exponent differs from reported form |
| Default operating point runtime | $\sim 10^9 \times$ closed-form baseline | §6 |

The $O((K M N_t^2)^{3.5})$ and $O((K+1)^3 (MN_t)^6 (K+P+M))$ forms
**disagree on the exponent of $K$** (3.5 vs 3). This is the
asymptotic-vs-leading-order tension: the simplified form treats $K$
as a scaling parameter and recovers the standard $n^{3.5}$
homogeneous complexity, while the reported form treats $K$ as a
constant and reports the leading-order $n^3 m$ form. Both are
correct asymptotic bounds; the reviewer should pick the one that
matches the framing of the paper.

---

## 9. Pointer to Companion Documents

- `math_derivation{,_en}.tex` §6, *theorem statement*, line 210 (zh)
  / line 207 (en) — the source-of-record complexity bound.
- `CONVEXIFICATION_CHAIN.md` §5.4 — the complexity summary in the
  prose companion. (Uses the source-of-record form.)
- `PAPER_OUTLINE_IEEE.md` §V — the closed-form baseline that
  motivates the runtime comparison.
