# DF Louvain (GPU) vs NetworkX — Synthetic Erdős–Rényi Dynamic Graphs

Comparison of the from-scratch [`df_louvain.cu`](../algorithm/df_louvain.cu)
(Static + Naive-Dynamic / Dynamic-Frontier / Delta-Screening, all in one GPU
process) against the NetworkX CPU baseline ([`nx_louvain.py`](../algorithm/nx_louvain.py),
which re-runs Louvain from scratch after every batch), on the generated
Erdős–Rényi graphs from [`generate/run_benchmarks.sh`](../generate/run_benchmarks.sh).

Data parsed from `generate/outputs/df/*.txt` and `generate/outputs/networkx/*.txt`.
Seed 42, 5 batches each.

## ⚠️ Read this first — ER graphs have no community structure

These are **uniform random** graphs. There are **no planted communities**, so
there is no "right" answer with high modularity — any positive modularity a
Louvain optimiser reports is *spurious* (it greedily carves a structureless
graph into blocks). This makes ER graphs a **worst case / stress test**, not a
representative quality benchmark. The point here is *relative* behaviour and,
more importantly, it surfaces a **correctness red flag in df_louvain** (below).
The real quality evaluation belongs on the structured temporal SNAP graphs.

Per-batch updates are tiny — `batch_pct = 0.001`, i.e. del = ins per batch is
~0.1 % of edges (1, 6, 9, 44, 249 edges respectively) — so the graph barely
changes across batches, and NetworkX's modularity is essentially flat.

---

## TL;DR

1. **NetworkX finds Q ≈ 0.13–0.27; df_louvain finds far less** (best variant
   0.02–0.13), a gap of **0.05–0.14** on every graph.
2. **df_louvain's Static and ND return Q ≤ 0 — sometimes *below the
   all-singletons baseline*** (e.g. n100 Static Q = −0.042 when singletons give
   ≈ −0.011). A correct Louvain can never end below its singleton start. This is
   a genuine bug signature: the asynchronous parallel local-move net-*decreases*
   modularity on structureless dense graphs (moves scored against stale Σ).
3. **DF and DS recover to positive Q** by perturbing the warm-start less, but
   still trail NetworkX badly.
4. **Speed: df_louvain wins big on large graphs** (≈127× faster end-to-end on
   n10000) but is **slower than NetworkX on the tiny n100** graph — its ~100 ms
   Static phase is fixed GPU overhead, not compute.

---

## NetworkX baseline (CPU, re-run from scratch each batch)

| Graph (n, p) | Nodes | Edges | Q (initial) | Q (batch 5) | Comms | Init t (s) | Total t, 6 runs (s) |
|--------------|------:|------:|------------:|------------:|------:|-----------:|--------------------:|
| n100, 0.10  | 100 | 474 | 0.272697 | 0.263201 | 7 | 0.0069 | 0.0393 |
| n500, 0.05  | 500 | 6,162 | 0.167718 | 0.166570 | 9–10 | 0.0933 | 0.5945 |
| n1000, 0.02 | 1,000 | 9,925 | 0.202503 | 0.201175 | 8–11 | 0.1721 | 1.0849 |
| n3000, 0.01 | 3,000 | 44,700 | 0.162802 | 0.166603 | 8–11 | 0.9494 | 5.6966 |
| n10000, 0.005 | 10,000 | 250,012 | 0.126959 | 0.128463 | 6–9 | 4.8810 | 34.1672 |

NetworkX modularity is flat across batches (batches are ~0.1 % of edges).

---

## df_louvain — Static (initial graph)

| Graph | Static Q | Comms | Passes | Time (ms) | singleton-Q ≈ | below singleton? |
|-------|---------:|------:|-------:|----------:|--------------:|:----------------:|
| n100   | **−0.042348** | 10 | 3 | 105 | −0.011 | **YES** ✗ |
| n500   | **−0.011666** | 16 | 3 | 98 | −0.002 | **YES** ✗ |
| n1000  | **−0.013796** | 36 | 4 | 115 | −0.001 | **YES** ✗ |
| n3000  | −0.000339 | 138 | 4 | 147 | −0.0002 | ~at baseline |
| n10000 | 0.003097 | 8 | 3 | 136 | −0.00004 | no (but ≈0) |

A from-singletons Louvain must finish with **Q ≥ Q(singletons)**. The first
three rows finish *below* it → the parallel local-moving is destroying
modularity, not improving it, on these dense structureless graphs.

---

## df_louvain — dynamic variants, final batch (batch 5)

| Graph | ND Q (comms) | DF Q (comms) | DS Q (comms) | DF affected₀ | DS affected₀ | DF t (ms) |
|-------|-------------:|-------------:|-------------:|-------------:|-------------:|----------:|
| n100   | −0.0588 (8) | **0.1321** (7)  | 0.1298 (7)  | 4    | 65     | 4 |
| n500   | −0.0128 (12)| 0.0178 (16)     | **0.1167** (6) | 12 | 459    | 1 |
| n1000  | −0.0249 (8) | 0.0131 (36)     | **0.0869** (4) | 20 | 1,000  | 1 |
| n3000  | −0.0246 (2) | **0.0245** (138)| 0.0009 (6)  | 85   | 2,999  | 2 |
| n10000 | 0.0074 (4)  | **0.0477** (6)  | 0.0096 (5)  | 498  | 10,000 | 3 |

(affected₀ = vertices marked affected at the start of the batch.)

---

## Quality gap — NetworkX vs best df_louvain variant

| Graph | NX Q (batch 5) | Best df Q | Best variant | **Q gap** | NX comms | df comms |
|-------|---------------:|----------:|--------------|----------:|---------:|---------:|
| n100   | 0.2632 | 0.1321 | DF | **−0.131** | 7 | 7 |
| n500   | 0.1666 | 0.1167 | DS | **−0.050** | 9 | 6 |
| n1000  | 0.2012 | 0.0869 | DS | **−0.114** | 11 | 4 |
| n3000  | 0.1666 | 0.0245 | DF | **−0.142** | 8 | 138 |
| n10000 | 0.1285 | 0.0477 | DF | **−0.081** | 7 | 6 |

df_louvain trails NetworkX by 0.05–0.14 modularity on every graph, and never
once matches it. On n3000 the best df variant (DF) is at 0.025 vs NX 0.167.

---

## Speed — end-to-end (Static + 5 batches)

"df DF trajectory" = Static + the five DF batch times. NetworkX total = its six
from-scratch invocations.

| Graph | df total (ms) | of which Static (ms) | NX total (ms) | speedup (df vs NX) |
|-------|--------------:|---------------------:|--------------:|-------------------:|
| n100   | 121 | 105 | 39 | **0.3× (slower)** |
| n500   | 103 | 98  | 595 | 5.8× |
| n1000  | 121 | 115 | 1,085 | 9.0× |
| n3000  | 153 | 147 | 5,697 | 37× |
| n10000 | 270 | 136 | 34,167 | **127×** |

Per-update, DF is 1–35 ms vs NetworkX's 6–7,000 ms per re-run. But df's Static
phase is ~100 ms of largely *fixed* GPU overhead (context, allocations, kernel
launches), which dominates on the small graphs and makes n100 a net loss.

---

## Critical observations

1. **The sub-singleton modularity is the headline problem.** Static/ND finishing
   below the all-singletons baseline (n100/n500/n1000) means the parallel
   local-move is accepting a *set* of moves that each looked positive against
   stale Σ but are jointly negative — the classic parallel-Louvain hazard. On
   structured graphs this is mild; on dense structureless ER graphs it is severe.
   **Fix direction:** a synchronous two-phase move with a swap/oscillation guard,
   or recomputing Σ between sub-rounds, or rejecting a pass whose *measured* ΔQ
   (not the summed per-move claims) is ≤ 0.

2. **ND is uniformly the worst df variant** (Q ≤ 0 on 4/5 graphs). Warm-starting
   from the already-bad Static partition and then re-moving *every* vertex just
   re-triggers the same thrash. The restricted variants (DF/DS) do better
   precisely because they perturb less.

3. **DF inherits Static's over-fragmentation.** Because DF only refines the local
   frontier and never re-coarsens, its community count stays pinned at Static's
   (e.g. n3000: 138 comms through all 5 batches). It nudges Q upward locally but
   cannot undo a bad global partition.

4. **DS's selectivity collapses on these graphs.** DS marks the *entire* affected
   community; with few large communities, that degenerates to ~all vertices
   (affected₀ = 10,000 / 10,000 on n10000), so DS ≈ ND in coverage but pays extra
   screening cost. It does, oddly, win on the mid-size graphs (n500/n1000)
   because marking everything lets it re-coarsen to few communities.

5. **NetworkX's spurious modularity rises as graphs get denser/smaller** (0.27 at
   n100 down to 0.13 at n10000) — expected Louvain overfitting on random graphs.

---

## Caveats & recommended next step

- **ER graphs are the wrong benchmark for *quality*.** With no planted
  structure, "better modularity" mostly measures who overfits more. Do **not**
  read these as df_louvain being 0.1 worse "in general."
- **But the sub-singleton Q is real and graph-independent in cause** — it must be
  fixed before any quality claim. Re-run after hardening the local-move and
  confirm Static Q ≥ singleton Q on every graph.
- **The meaningful comparison is the structured temporal SNAP graphs**
  (CollegeMsg, sx-mathoverflow, …), where NetworkX itself reaches Q ≈ 0.25–0.49.
  Run `snap_temporal/run_df.sh` + NetworkX on the same inputs and compare there;
  that is where DF/DS are designed to shine and where the collapse fix matters.
- **Speed is already a clear df win** on everything but the smallest graph, and
  grows with size (127× on n10000) — so once quality is fixed, the value
  proposition is strong.
