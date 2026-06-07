# Dynamic Louvain (`df_louvain.cu`) — Implementation & Considerations

Notes on the from-scratch GPU Louvain in
[`../algorithm/df_louvain.cu`](../algorithm/df_louvain.cu). It replaces the older
`cuda_dynamic_louvain.cu` / `cuda_dynamic_louvain_nodebased.cu`. The correctness
history is in [bugfixes.md](bugfixes.md); the algorithms follow the DF Louvain
paper (Sahu, arXiv:2404.19634) and the Delta-Screening paper (Zarayeneh &
Kalyanaraman, IEEE TNSE 2021).

## 1. Overview

A single process computes **Static**, **Naive-Dynamic (ND)**, **Dynamic-Frontier
(DF)**, and **Delta-Screening (DS)** Louvain. All four are produced by *one*
shared driver, `louvain()`, which differs only in (a) the warm-start community it
is given and (b) the initial *affected* mask:

| Variant | Warm start | Affected (pass-0) mask | Frontier growth |
|---------|-----------|------------------------|-----------------|
| Static | identity (each vertex its own community) | all vertices | n/a |
| ND     | previous communities $C^{t-1}$ | all vertices | n/a |
| DF     | $C^{t-1}$ | endpoints of intra-community deletions / inter-community insertions | yes (grows during local moving) |
| DS     | $C^{t-1}$ | source + neighbours + whole affected communities | no (fixed range) |

Everything is one `.cu` file with a single `louvain()` and one `localMove_kernel`;
there are no separate per-variant entry points and no node-based vs edge-based split.

---

## 2. Algorithm

### 2.1 Modularity and the move gain

With each undirected edge stored as two directed arcs, let $M$ = total directed
weight ($=2m$), $K_i$ = weighted degree of $i$, $\Sigma_c$ = total weighted degree
of community $c$ (which includes $i$ when $i\in c$), and $K_{i\to c}$ = weight of
arcs from $i$ into community $c$ (self-loops excluded). The modularity computed on
the host for reporting is

$$Q = \sum_c\!\left[\frac{\sigma_c}{M} - \left(\frac{\Sigma_c}{M}\right)^{\!2}\right],$$

and the gain of moving $i$ from its community $d$ to $c$, used by the kernel, is

$$\Delta Q_{i:d\to c} = \frac{2\,(K_{i\to c} - K_{i\to d})}{M}
 - \frac{2\,K_i\,(K_i + \Sigma_c - \Sigma_d)}{M^2}.$$

(This is the standard combined remove-then-add gain; $\Sigma_d$ includes $i$.)

### 2.2 The two phases

`louvain()` alternates a **local-moving** phase (move each vertex to the
neighbouring community of highest $\Delta Q$) and an **aggregation** phase
(collapse each community into a super-vertex), warm-starting from $C^{t-1}$ for the
dynamic variants. Passes repeat until convergence or a low-shrink/​pass cap.

### 2.3 Affected-vertex marking (host functions)

- **`markND`** — every vertex affected and in range.
- **`markDF`** (DF, initial set) — for an intra-community deletion $(i,j)$ or an
  inter-community insertion $(i,j)$, mark $i$ and $j$. The set then **grows on the
  GPU**: whenever a vertex moves, its neighbours are added to the range.
- **`markDS`** (DS) — deletions: mark $i$, its neighbours, and all of $C^{t-1}_i$.
  Insertions: per source $i$, screen the inserted edges and mark $i$, its
  neighbours, and the best target community $c^\ast$. Propagate to neighbours and
  whole communities. The range is **fixed** (no frontier growth).
  - This is the **parallel** adaptation of DS (DF-paper Alg. 3): the screening
    gain uses only the *newly inserted* edge weights as $K_{i\to c}$, and the
    original's `gain1 ≥ gain2` deferral is dropped (it over-marks → safe superset).

Only the **first pass** uses the affected mask; later passes (on the much smaller
aggregated graph) process every super-vertex.

---

## 3. Data structures

CSR is stored as **structure-of-arrays** (better coalescing than the old `Edge`
AoS):

```cpp
struct DGraph { int N, M; int* off; int* dst; double* w; };   // off[N+1], dst[M], w[M]
```

Per-graph device state in `louvain()`:

| Array | Meaning |
|-------|---------|
| `d_C[v]` | community of current (super-)vertex `v` |
| `d_K[v]` | weighted degree $K_v$, recomputed from CSR each graph (`computeK_kernel`) |
| `d_Sigma[c]` | community total degree $\Sigma_c$, maintained with atomics during moves |
| `d_orig[v]` | dendrogram: the current super-vertex that original vertex `v` lives in |
| `d_active[v]` / `d_range[v]` | scheduled-this-iteration / allowed-to-process masks |
| `d_commLock[c]` | per-community spin-lock for the verified commit (Section 4.2) |
| `d_htOff`, `d_htKey`, `d_htVal` | per-vertex open-addressing hashtable (Section 4.1) |

`d_orig` replaces the old `LouvainState`/`d_original_community`: after each pass it
is folded through the renumbered communities (`fold_kernel`) so the
original-vertex → final-community map survives aggregation.

---

## 4. Local-moving phase (`localMove_kernel`)

**Vertex-parallel** — one thread per vertex (not edge-parallel/warp-leader as
before), and **one kernel launch per iteration** (no cooperative kernels /
`grid.sync`). The host loops iterations, reading back the per-iteration $\Delta Q$
to test convergence against the tolerance $\tau$.

### 4.1 Per-vertex hashtable

To pick the best community in $O(\deg)$, each vertex owns a slice of a global
open-addressing hashtable (`community → accumulated weight`), sized to
`nextPow2(deg+1)` so hubs never overflow and linear probing always terminates.
Offsets are built per pass (`htCap_kernel` + exclusive scan). The thread scans its
neighbours into the table, reads $K_{i\to d}$, then scans the table for the
community maximising $\Delta Q$.

### 4.2 Verified commit under a community-pair lock (the correctness fix)

The original lock-free design committed moves scored against a **stale** `Σ`,
which on dense graphs drove modularity *below* the all-singletons baseline (see
[bugfixes.md](bugfixes.md)). The current kernel instead:

1. picks `best_c` **optimistically** (possibly-stale `Σ`); if no positive gain, return;
2. locks the pair `(d, best_c)` with `atomicCAS` in `min,max` order;
3. **re-reads** `Σ[d]`, `Σ[best_c]` and **re-scans** $K_{i\to d}$, $K_{i\to best\_c}$
   — while both communities are locked their membership is frozen, so these are
   exact;
4. commits `C[v]=best_c`, `Σ[d]-=K_v`, `Σ[best_c]+=K_v` only if the exact
   `dQ2 > MOVE_EPS`; `__threadfence()`; releases the locks.

`C[v]` is written only by `v`'s own thread, so no per-vertex lock is needed.
**Guarantee:** every committed move increases Q, so Q is monotonically
non-decreasing across the phase — sub-singleton results are impossible, and 2-cycle
swaps cannot oscillate (the loser re-evaluates and sees `dQ2 ≤ 0`). No deadlock:
ordered locks + Ampere (sm_86) Independent Thread Scheduling let a lock holder
progress while same-warp threads spin.

### 4.3 Vertex pruning and frontier

On a successful move the moved vertex's neighbours are re-activated (`active=1`);
for **DF** they are also added to the range (`range=1`), which is how the frontier
grows. A vertex is processed once per iteration and pruned until a neighbour
re-activates it.

---

## 5. Aggregation (`aggregate`, Thrust)

1. Relabel every arc to `(C[src], C[dst])` and form a 64-bit key `src*nc + dst`.
2. `thrust::sort_by_key` + `thrust::reduce_by_key` combine duplicate super-arcs,
   summing weights (intra-community arcs become self-loops carrying $\sigma_c$).
3. Decode keys back to `(src,dst)`, count per-source with atomics, exclusive-scan
   to offsets.
4. New $K$ = super-vertex degrees (`computeK_kernel`); new $\Sigma = K$; new $C$ =
   identity. Self-loops are **kept** — they carry the internal weight needed for
   correct degrees and modularity, so $Q$ is preserved across aggregation.

---

## 6. Pass control (constants at the top of the file)

| Constant | Role | Default |
|----------|------|---------|
| `TOLERANCE` | initial per-iteration $\Delta Q$ tolerance $\tau$ | `1e-2` |
| `TOLERANCE_DROP` | $\tau$ shrinks by this each pass | `10` |
| `AGG_TOLERANCE` | stop if a pass shrinks the community count by `< (1-this)` | `0.8`* |
| `MAX_ITERATIONS` | local-moving iterations per pass | `20` |
| `MAX_PASSES` | aggregation passes | `10`* |
| `MOVE_EPS` | accept a move only if $\Delta Q$ exceeds this | `1e-12` |

\* `AGG_TOLERANCE`/`MAX_PASSES` are the main tuning knobs for coarsening depth (the
DF paper suggests `AGG_TOLERANCE`→1 to disable the low-shrink break on real-world
graphs). A pass loop ends on: local-move converged in ≤1 iteration, low shrink, or
`nc == g.N`.

---

## 7. Batch pipeline (`main`)

```
read initial graph → build host adjacency (each undirected edge once)
Static  = louvain(identity, all-affected)              → report Q / comms / time
keep three independent running states: sND, sDF, sDS   (each = community + K + Σ)
for each batch:
    apply deletions/insertions to host adjacency; rebuild CSR; recompute K, Σ
    ND : markND;  louvain(sND.comm, all)         → report; sND ← result
    DF : markDF;  louvain(sDF.comm, frontier)    → report; sDF ← result
    DS : markDS;  louvain(sDS.comm, fixed range) → report; sDS ← result
write DF final communities to argv[1]
```

Each method carries its **own** state across batches (so DF batch *t* warm-starts
from DF batch *t-1*), which is the correct way to benchmark the variants against
each other. Modularity, community count, affected-set size, and time are printed
per method per batch.

I/O is unchanged from the old format: `n m`, then `m` undirected `u v` edges, then
`n_batches`, then per batch `n_del n_ins` followed by the deletion and insertion
`u v` lines. Edge weights are read as `u v` and treated as 1.

---

## 8. Numerical & correctness notes

- Weights, $K$, $\Sigma$, $Q$, and the $\Delta Q$ accumulator are all `double`
  (`atomicAdd(double)` needs sm ≥ 6.0).
- Modularity is computed on the **host** only for reporting; the GPU uses $\Delta Q$.
- The move threshold `MOVE_EPS = 1e-12` rejects float-noise moves.

---

## 9. Known residual issues (post-fix)

The sub-singleton collapse and the static under-coarsening are **fixed**; static
SNAP graphs now match NetworkX (see the `df_vs_nx_*` comparison notes). Remaining,
secondary:

1. **Large temporal-initial Static under-coarsens** (sx-askubuntu/superuser stop at
   ~3 passes); the dynamic variants recover it — a pass-stop tuning matter
   (`AGG_TOLERANCE`).
2. **ER n10000 over-merges** to 2 communities (structureless-graph edge case).
3. **sx-superuser batch-5 ND** collapses on one batch (variant instability; DF/DS
   are steadier there).
4. **Fixed-size launches** (`TPB=256`, `<<<256,256>>>` for helpers); no
   occupancy-tuned grid.
5. **Host CSR rebuild per batch** (no incremental graph/​auxiliary maintenance),
   so the per-batch cost is dominated by the rebuild rather than the (small)
   affected region — the DF paper's incremental $K/\Sigma$ maintenance is not
   implemented.
6. **GPU memory** scales with the per-vertex hashtable (~$2(M+N)$ entries) plus
   Thrust sort scratch; multi-million-edge graphs can be tight on small cards.

---

## 10. Build

```bash
cd algorithm
nvcc -O3 -arch=sm_86 df_louvain.cu -o df_louvain     # sm_86 = Ampere (RTX 30xx)
```

`-arch=sm_86` matters: the verified-commit spin-lock relies on Volta+ Independent
Thread Scheduling for intra-warp forward progress. Requires the CUDA Toolkit, a GPU
with compute capability ≥ 6.0 (≥ 7.0 recommended for the locks), Thrust (bundled
with CUDA), and a C++ STL. No cooperative-groups / `-rdc=true` requirement
(launch-per-iteration replaced the cooperative kernels).
