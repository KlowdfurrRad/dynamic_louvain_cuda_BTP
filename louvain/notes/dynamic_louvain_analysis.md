# df_louvain — Performance Analysis (GPU Memory & Runtime Overhead)

Analysis of the GPU memory footprint and the runtime overhead of
[`../algorithm/df_louvain.cu`](../algorithm/df_louvain.cu). Implementation details
are in [dynamic_louvain.md](dynamic_louvain.md).

Throughout, **M = directed arcs = 2 × (undirected edges E)** and **N = nodes**.

---

## 1. GPU Memory Requirements

### 1.1 Where the memory goes

**Persistent (held for the whole run):**

| Buffer | Size |
|--------|------|
| CSR `dst` (`int`) + `w` (`double`) | `4M + 8M` = 12 B/arc |
| per-vertex hashtable `htKey` + `htVal` | `12 × htTotal`, with `htTotal ≈ ` up to `2(M+N)` → ~24 B/arc |
| 9 per-node arrays (`K, C, Σ, orig, active, range, present, cap, commLock`) + `off` + `htOff` | ~56 B/node |

→ persistent floor ≈ **36 B/arc + 56 B/node ≈ 72 B/edge + 56 B/node**.

The hashtable capacity per vertex is `nextPow2(deg+1)`, so `htTotal = Σ_v nextPow2(deg_v+1)`
lies between `M+N` and `2(M+N)`; the table above uses the `2(M+N)` worst case.

**Transient peak — the first `aggregate()` on the *full* graph.** Aggregation uses
Thrust `sort_by_key` + `reduce_by_key`, allocating six temporary arrays
(`src, key, w2, okey, ow, osrc`) plus sort scratch ≈ **56 B/arc**, *on top of* the
persistent buffers (the hashtable is not freed first). This is the high-water mark.

→ peak ≈ **92 B/arc + 80 B/node ≈ 184 B/edge + 80 B/node** (worst case; typically
~25 % less, since `htTotal` is usually below `2(M+N)`).

### 1.2 Footprint of the actual benchmark graphs

RTX 3050 has 4 GB, of which ~3.5 GB is usable after the driver/CUDA context.

| Graph | Edges | Nodes | Peak (worst) | Fits 4 GB? |
|-------|------:|------:|-------------:|:----------:|
| dolphins / karate | ~160 | ~60 | ~30 KB | ✅ |
| CollegeMsg (initial) | 11 K | 1.9 K | ~2 MB | ✅ |
| email-Enron | 184 K | 37 K | ~37 MB | ✅ |
| com-amazon | 926 K | 335 K | ~197 MB | ✅ |
| com-dblp | 1.05 M | 317 K | ~218 MB | ✅ |
| com-Youtube | 2.99 M | 1.13 M | ~640 MB | ✅ |
| web-Google | 4.32 M | 876 K | ~865 MB | ✅ |
| **com-LiveJournal** | **34.7 M** | **4.04 M** | **~6.7 GB** | ❌ **OOM** |

Everything up to ~web-Google fits comfortably. The practical ceiling on a 4 GB card
is roughly **~18 M edges**, so **LiveJournal (35 M edges) does not fit**.

### 1.3 The bottleneck

The peak is set by the **first aggregation**, where the Thrust sort temporaries
(~56 B/arc) coexist with the persistent per-vertex hashtable (~24 B/arc) — together
about 2.3× the graph itself.

### 1.4 Ways to push to larger graphs (4 GB card)

- **Free `htKey`/`htVal` before `aggregate()`** (rebuild after) — drops peak by ~24 B/arc.
- **`float` edge weights** instead of `double` — saves 4 B/arc each in `w`, `w2`, `ow`.
- **32-bit aggregation keys** when `nc` is small enough (`src*nc + dst < 2³²`) —
  halves the `key`/`okey` temporaries.
- **Chunked aggregation** of the edge array.

Together these could roughly halve the peak (~90 B/edge), pushing the ceiling toward
~35 M edges — enough for LiveJournal.

---

## 2. Runtime Overhead (why small graphs are "slow")

### 2.1 The observation

On the classic graphs, `df_louvain`'s reported (algorithm) time is large relative
to the trivial amount of work:

- **karate** (78 edges): 140 ms
- **dolphins** (159 edges): 93 ms

while NetworkX does the same graphs in ~3 ms. This is **not** the graph computation —
it is fixed GPU overhead.

### 2.2 Evidence: time barely depends on graph size

| Graph | Edges | df time |
|-------|------:|--------:|
| dolphins | 159 | 93 ms |
| karate | 78 | 140 ms |
| ER n1000 | 9,925 | 82 ms |
| ER n10000 | 250,012 | 304 ms |
| com-amazon | 925,872 | 174 ms |
| com-dblp | 1,049,866 | 251 ms |

A **1-million-edge** graph takes about the **same time as a 78-edge** graph. If the
runtime were the actual graph work, com-dblp would be ~13,000× slower than karate —
instead it is ~2×. So the ~100 ms is almost entirely **fixed setup cost**, paid
regardless of graph size.

### 2.3 Where the ~100 ms goes

1. **CUDA context initialisation (dominant).** The first GPU call in the process
   creates the CUDA context — loads the driver, allocates context memory, and
   loads/JITs the compiled kernels. This is ~100 ms (more on first launch) and is a
   one-time, size-independent cost. The internal timer wraps `louvain()`, whose
   first `cudaMalloc` triggers exactly this.
2. **Per-call overhead** — dozens of `cudaMalloc`/`cudaFree`, kernel launches, and
   Thrust `sort`/`scan` setup, each a few–tens of microseconds.
3. **Per-iteration host↔device sync** — each local-move iteration copies the
   per-iteration `ΔQ` back to the host and synchronises (≈ MAX\_ITERATIONS × passes
   times). Cheap individually, but real.

The actual work on tens of nodes is *microseconds* — utterly dwarfed by the above.
NetworkX is faster on these because there is **no device to set up**.

### 2.4 Conclusion

This is inherent to using a GPU for tiny inputs: high fixed cost, enormous
throughput. The GPU is the wrong tool for a 60-node graph and the right tool for a
million-edge one — where this same `df_louvain` beats NetworkX by **100–336×**
(e.g. com-amazon 174 ms vs 58 s). The classic small graphs are therefore useful as
**correctness** checks (matching NetworkX's modularity), not speed comparisons.

### 2.5 Reducing the small-graph overhead

The only meaningful lever is **context/buffer reuse across graphs** — process all
graphs in a single process (and reuse device buffers) instead of relaunching the
binary per graph. The context-initialisation floor (~100 ms per process) remains,
so the GPU will still lose to the CPU on individual tiny graphs; the win is only
amortised across many of them.

---

## 3. Graph structure vs. performance — why some graphs do badly

`df_louvain`'s **relative** speed (vs cuGraph / NetworKit) is governed by the input's
**degree distribution**, not its size. It dominates on bounded-degree graphs and loses
its advantage on scale-free social networks with extreme hub vertices.

### 3.1 Degree statistics of the *converted* benchmark graphs

All figures are measured on the **converted** graphs (the simple, undirected, 0-indexed
form fed to every implementation — e.g. web-Google is symmetrised from its directed raw
form), not the raw SNAP files. **Top-0.1% E** = % of edge endpoints incident to the top
0.1% highest-degree vertices (a hub-concentration measure).

| Graph | Type | Vertices | Edges | Avg deg | **Max deg** | Max/avg | Top-0.1% E | df vs NetworKit |
|-------|------|---------:|------:|--------:|------------:|--------:|-----------:|:---------------:|
| com-dblp | collaboration | 317,080 | 1,049,866 | 6.6 | **343** | 52 | 2% | **4.5× faster** |
| com-amazon | co-purchasing | 334,863 | 925,872 | 5.5 | **549** | 99 | 2% | **5.4× faster** |
| web-Google | web | 875,713 | 4,322,051 | 9.9 | **6,332** | 641 | 8% | **15× faster** |
| com-Youtube | social | 1,134,890 | 2,987,624 | 5.3 | **28,754** | 5,461 | 16% | 1.2× (≈ tie) |
| com-LiveJournal | social | 3,997,962 | 34,681,189 | 17.4 | **14,815** | 854 | 4% | 0.8× (slower) |
| com-Orkut | social | 3,072,441 | 117,185,083 | 76.3 | **33,313** | 437 | 3% | 0.3× (3.4× slower) |

The good/bad split tracks **maximum degree**: amazon/dblp (max < 600) and web-Google
(max 6,332) → df fast; Youtube/LiveJournal/Orkut (max 15k–33k) → df slow. Modularity stays
competitive on all (highest on LiveJournal) — this is a **throughput**, not a **quality**,
effect. **Top-0.1% E** adds nuance: Youtube is the most hub-concentrated (16%), while Orkut
— though densest — is not (3%), so Orkut's cost is **density**, not hub concentration.

### 3.2 Code-level causes

1. **SIMT load imbalance (primary).** The local-move kernel is vertex-parallel: one thread
   per vertex doing O(deg) work (scan neighbours → per-vertex hashtable → re-scan for the
   best community). A warp of 32 threads runs in lock-step, so it cannot retire until its
   slowest lane finishes. One hub of degree 28,754 in a warp ⇒ the other 31 lanes sit idle
   for ~28,754 steps. Power-law graphs have hundreds of such hubs ⇒ many warps stalled ⇒
   device badly under-utilised. Bounded-degree graphs have balanced lanes ⇒ the simple
   one-thread-per-vertex scheme is efficient. Standard remedy (**not implemented**): assign
   threads/warps to a vertex ∝ its degree (edge-parallel, or warp-per-hub with degree binning).
2. **Per-vertex hashtable memory ∝ density.** `htTotal = Σ_v nextPow2(deg_v+1)` grows with
   average degree; on dense Orkut (avg 76) this becomes the dominant buffer and is why
   **com-Orkut OOMs the 16 GB T4** (must run on the A100 40 GB). Hubs also force large
   global-memory tables that the (linear-probing) hub thread scans slowly.
3. **Commit-lock + Σ-atomic contention ∝ density / community coarseness.** Verified-commit
   locks the `(d, best_c)` community pair and atomicAdds Σ[d], Σ[best_c]. Orkut collapses to
   ~30 communities, so nearly every vertex contends on ~30 locks / Σ counters ⇒ serialised
   commit. The good graphs have hundreds–thousands of communities ⇒ contention spread thin.

Orkut compounds all three (hub imbalance + density + coarse-community contention) → worst
case (126 s, 3.4× slower than NetworKit). Youtube is bad almost purely from hubs (its
average degree is only 5.3).

### 3.3 Why the baselines don't suffer

- **NetworKit PLM** (OpenMP): independent threads + dynamic scheduling / work-stealing — a
  hub thread just runs longer; it doesn't freeze 31 lock-step siblings.
- **cuGraph**: degree-aware GPU load balancing (multiple threads per high-degree vertex).
- Irony: `df_louvain` is a GPU port of the *shared-memory* GVE-Louvain. The per-vertex
  hashtable is fine for independent CPU threads but a liability under GPU SIMT on power-law
  graphs without degree-aware work assignment.

### 3.4 Note — supersedes the §1.2 memory estimates

Empirically, **com-LiveJournal (34.7 M edges) DOES fit the 4 GB RTX 3050** — the §1.2
"~6.7 GB → OOM" was a ~2× pessimistic worst case; the real cost is ≈ 50 B/arc. The actual
memory wall is **com-Orkut** (117 M edges, avg degree 76): it OOMs the 16 GB T4 (both
cuGraph and df) and needs the A100. Literature: GPU-Louvain load balancing
[arXiv:1805.10904], GVE-Louvain [arXiv:2312.04876], ν-Louvain [arXiv:2501.19004].
