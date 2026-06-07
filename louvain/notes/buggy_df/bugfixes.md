# df_louvain.cu — Bug-fix log

A running log of correctness fixes to the from-scratch GPU Louvain
([`df_louvain.cu`](../algorithm/df_louvain.cu)). Newest first.

---

## 2026-06-07 — Sub-singleton modularity from stale-Σ commits in the local move

### Status
Implemented in source (verified-commit-under-lock local-move). **Pending recompile
+ re-run** to confirm on the benchmarks.

### Symptom (what the benchmarking showed)
On dense / weak-structure graphs the **Static** Louvain returned modularity
*below* the all-singletons baseline — which a correct Louvain can never do
(it starts at singletons and only accepts improving moves, so the final Q must be
≥ the singleton Q):

| Graph | df Static Q | singleton-Q ≈ | NetworkX Q |
|-------|------------:|--------------:|-----------:|
| ER n100 p0.10 | **−0.042** | −0.011 | 0.273 |
| ER n500 p0.05 | **−0.012** | −0.002 | 0.168 |
| ER n1000 p0.02 | **−0.014** | −0.001 | 0.203 |
| CollegeMsg (temporal initial) | **−0.158** | ≈0 | 0.257 |

Because ND/DF/DS all **warm-start from the Static partition**, a poisoned Static
result propagated downstream: on CollegeMsg every dynamic variant stayed negative
across all 5 batches (DF climbed −0.145 → −0.095 but never crossed 0), and the
later-batch "drift" on sx-askubuntu/superuser was the same effect compounding.
Full data in [df_vs_nx_random_comparison.md](df_vs_nx_random_comparison.md) and
[df_vs_nx_temporal_comparison.md](df_vs_nx_temporal_comparison.md).

Sparse, well-structured graphs (sx-mathoverflow, the static SNAP collaboration
graphs) did **not** go negative — there the move kernel behaved and the remaining
gap was a separate *under-coarsening* issue (see
[df_vs_nx_snap_comparison.md](df_vs_nx_snap_comparison.md), not addressed here).

### Root cause
The original local-move kernel was **asynchronous and lock-free**. One thread per
vertex did:

```
d   = C[v];                         // own community
ki  = K[v];                         // degree
... scan neighbours -> hashtable K_{v->c} ...
Sd  = Sigma[d];                     // <-- non-atomic read, may be stale
for each candidate c:
    Sc = Sigma[c];                  // <-- non-atomic read, may be stale
    dQ = 2*(Kic-Kid)/M - 2*ki*(ki+Sc-Sd)/M^2;
    track best
if best_dQ > eps:
    atomicAdd(&Sigma[d], -ki);  atomicAdd(&Sigma[best_c], +ki);  C[v]=best_c;
```

`Sigma[]` (per-community total degree) is being mutated by many threads via
`atomicAdd` while each thread reads it **non-atomically** to score its move. So a
move is committed based on a `Σ` snapshot that no longer holds at commit time.
Two concrete failure modes:

1. **2-cycle swap.** Two singletons `u,v` joined by one edge, processed in the
   same launch. Both read `Σ[u]=Σ[v]=1`, both compute `dQ=+0.5`, both commit:
   `C[u]=v`, `C[v]=u`. They just *swap labels* — still two singletons, real
   ΔQ = 0 — yet `dQ_accum` reports +1.0, so the phase never "converges" and keeps
   oscillating to the iteration cap.

2. **Stale-Σ overshoot (the one that drives Q negative).** Vertex `a` decides to
   join community `c` while `Σ_c` is still small (stale). By the time it commits,
   many other vertices have piled into `c`, so the true penalty term
   `2·k_a·Σ_c/M²` is far larger than what `a` scored against. The move is actually
   modularity-*decreasing* but is committed anyway. Accumulated over iterations on
   dense graphs, the partition ends up **below the singleton baseline**.

### Why parameter tuning could not fix it
We first raised `AGG_TOLERANCE` 0.8 → 0.98 and `MAX_PASSES` 10 → 20, and
considered lowering `TOLERANCE`. These only affect *when passes/iterations stop* —
they target the *under-coarsening* problem, not this one. Worse, a lower
`TOLERANCE` makes the swap oscillation run **longer** (the inflated `dQ_accum`
stays above the tolerance), so the kernel churns to `MAX_ITERATIONS` in a bad
state. The negative Q is a **correctness defect in the commit**, not a
stopping-criterion choice, so no constant fixes it.

### The fix — verified commit under a community-pair lock
Keep the fast optimistic search, but make the **commit** exact. A vertex now:

1. Scans neighbours and picks its best target `best_c` **optimistically** (against
   possibly-stale `Σ`), exactly as before. If no positive move, return (no lock).
2. To commit, acquires a spin-lock on the **pair of communities** `(d, best_c)`,
   always in `min,max` order:
   ```
   lo = min(d,best_c); hi = max(d,best_c);
   while (atomicCAS(&commLock[lo],0,1)!=0){}
   while (atomicCAS(&commLock[hi],0,1)!=0){}
   ```
3. **Re-scans** its neighbours to recompute `K_{v→d}` and `K_{v→best_c}`, and
   re-reads `Σ[d]`, `Σ[best_c]` — now under the lock — and recomputes `dQ2`.
4. Commits `C[v]=best_c` and `Σ[d]-=ki; Σ[best_c]+=ki` **only if `dQ2 > eps`**;
   `__threadfence()`; releases the locks.

`C[v]` is written only by `v`'s own thread, so no per-vertex lock is needed —
only the two community locks.

New state: an `int* commLock` array (size `N`, indexed by community id) is
allocated in `louvain()`, `cudaMemset` to 0 before each pass's iteration loop
(locks also self-reset since every acquire is paired with a release), and freed at
the end. The kernel takes it as a trailing parameter.

### Why it is correct
- **Frozen membership ⇒ exact gain.** While a thread holds the locks on `d` and
  `best_c`, no other vertex can join or leave either community — doing so would
  require the very same locks. Therefore the under-lock `Σ[d]`, `Σ[best_c]` and the
  re-scanned `K_{v→d}`, `K_{v→best_c}` are **exact at commit time**, so `dQ2` is the
  true modularity change. (Neighbours moving *between other* communities don't
  affect weights to `d`/`best_c`, so they're harmless.)
- **Monotonic non-decrease.** Every committed move has exact `dQ2 > eps`, so each
  commit strictly increases Q. Q is therefore non-decreasing across the whole
  local-move phase ⇒ final Q ≥ singleton Q. **Sub-singleton results are now
  impossible.**
- **Swap eliminated.** In the 2-cycle, whichever vertex commits second
  re-evaluates against the winner's already-applied move and sees `dQ2 ≤ 0`, so it
  stays put. No oscillation.

### Why it does not deadlock
- **No lock-ordering cycle:** locks are always taken `min` before `max`.
- **Intra-warp progress:** on the target GPU (RTX 3050, sm_86, Ampere) Independent
  Thread Scheduling lets a lock-holder make progress while other threads in the
  same warp spin — without ITS (pre-Volta) this spin-lock pattern can deadlock.
  Compile with `-arch=sm_86` (already in `compile_dyn.sh`). The prior edge-based
  kernel used this identical `atomicCAS` community-lock pattern successfully on
  this hardware.

### Performance trade-off
Reintroduces locks (removed in the original lock-free design for speed) plus a
second neighbour scan for **moving** vertices, performed while the two community
locks are held. Cost scales with lock contention (vertices targeting the same
community) and vertex degree. Expected impact: modest on sparse graphs (low
contention, low degree); larger on dense hubs (CollegeMsg) but those graphs are
tiny. Correctness was the priority; this is the proven-correct trade.

Note: an optimistic pick that is invalidated under the lock (`dQ2 ≤ eps`) is simply
**not** committed, and the vertex stays pruned for the iteration (standard vertex
pruning). This cannot cause incorrectness (it only ever *skips* a move); if it is
later found to worsen coarsening, re-activating `active[v]` on a failed commit is a
one-line follow-up.

### What this does NOT fix
The **under-coarsening** on sparse structured graphs (df leaves 3–110× more
communities than NetworkX on the static SNAP set) is a separate issue — likely the
pass-stop conditions / parallel local-move not merging aggressively enough — and is
tracked independently. See [df_vs_nx_snap_comparison.md](df_vs_nx_snap_comparison.md).

### How to verify the fix
1. Recompile: `cd algorithm && nvcc -O3 -arch=sm_86 df_louvain.cu -o df_louvain`.
2. Re-run the benchmarks (`generate/run_df.sh`, `snap_temporal/run_df.sh`,
   `real_graphs/run_df.sh`).
3. **Pass criterion:** Static `Q ≥ 0` (and ≥ the singleton baseline) on **every**
   graph, including ER n100 and CollegeMsg. CollegeMsg Static should now be
   positive and the dynamic variants should hold/​grow modularity across batches
   instead of staying negative.
4. Regenerate the comparison markdowns and check the gap to NetworkX has closed on
   the previously-failing graphs.
