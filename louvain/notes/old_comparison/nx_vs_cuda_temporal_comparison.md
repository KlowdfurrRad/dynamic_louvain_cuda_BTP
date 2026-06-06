# NetworkX vs CUDA Louvain — SNAP Temporal Graphs

Companion to [nx_vs_cuda_comparison.md](nx_vs_cuda_comparison.md) and
[nx_vs_cuda_comparison1.md](nx_vs_cuda_comparison1.md), but for the **temporal**
SNAP datasets in [../real_graphs/snap_temporal/](../real_graphs/snap_temporal/).
All numbers below are parsed directly from
`real_graphs/snap_temporal/outputs/{networkx,cuda_static,normal,node_based}/`.

## Experimental setup

Each raw SNAP temporal edge stream (`SRC DST UNIXTS`) is processed by
[`convert_snap_temporal_to_dynamic.py`](../real_graphs/snap_temporal/convert_snap_temporal_to_dynamic.py)
into a dynamic-Louvain input: edges are deduped to a simple undirected graph,
sorted chronologically, then split into an **initial graph (earliest 80 % of
edges)** plus **5 batches of insertions** (the remaining 20 %, in time order).
[`run_benchmarks.sh`](../real_graphs/snap_temporal/run_benchmarks.sh) runs four
implementations:

| Output dir | Binary | What it processes |
|------------|--------|-------------------|
| `networkx/` | `nx_louvain.py` (CPU) | initial graph only (`*_nx.txt`) — no dynamic mode |
| `cuda_static/` | `static_louvain.exe` | initial graph only (batches ignored) |
| `normal/` | `dynamic_louvain.exe` (**edge-based** kernel) | static-from-scratch on initial graph, then naive / frontier / delta-screening updates per batch |
| `node_based/` | `dynamic_louvain_nodebased.exe` (**node-based** kernel) | same as above, node-parallel kernel |

The three dynamic variants (naive / frontier / delta-screening) are defined in
[dynamic_louvain.md](dynamic_louvain.md). The four graphs and their converted
sizes (total nodes / initial 80 %-edge count, from the run logs):

| Graph | Raw temporal edges | Nodes | Initial edges (80 %) |
|-------|-------------------:|------:|---------------------:|
| CollegeMsg | 59,835 | 1,899 | 11,070 |
| sx-mathoverflow | 506,550 | 24,759 | 150,388 |
| sx-askubuntu | 964,437 | 159,320* | 508,003* |
| sx-superuser | 1,443,339 | 192,409 | 571,656 |

\* The NetworkX log for sx-askubuntu reports 159,320 nodes / 508,003 initial
edges; the currently-on-disk `converted/sx-askubuntu.txt` header is 157,222 /
364,552 (regenerated with a different split after the run). Numbers in this doc
are taken from the run logs that actually produced the outputs.

---

## TL;DR

1. **The CUDA results are largely degenerate on temporal graphs.** Modularity
   collapses far below NetworkX — usually to ~0 — on every graph except
   sx-askubuntu node-based.
2. **The dynamic *update* step is what destroys quality.** The static-from-scratch
   partition built inside the dynamic binary is reasonable (e.g. edge-based
   mathover Q≈0.279, node-based askubuntu Q≈0.415), but the **first batch
   update collapses Q toward zero** (avalanche / over-merge).
3. **Node-based ≫ edge-based** on both speed (5–6× faster) and quality
   retention. It is the only variant that keeps a meaningful partition
   (sx-askubuntu Q≈0.285 vs NX 0.486).
4. **Several runs did not complete.** Standalone `static_louvain.exe` only
   finished CollegeMsg; the three large graphs were truncated mid-run.
   sx-superuser has **no** dynamic or NetworkX output at all (NX skipped for
   size, dynamic runs never produced a file).
5. **Where CUDA completes, it is also slower than NetworkX** on the comparable
   (initial-graph) work — by 30× (askubuntu, node-based) to 174× (askubuntu,
   edge-based).

---

## NetworkX (CPU baseline — initial graph only)

| Graph | Nodes | Edges | Modularity | Communities | Time (s) |
|-------|------:|------:|-----------:|------------:|---------:|
| CollegeMsg | 1,899 | 11,070 | 0.256886 | 289 | 0.34 |
| sx-mathoverflow | 24,759 | 150,388 | 0.307303 | 5,783 | 11.57 |
| sx-askubuntu | 159,320 | 508,003 | 0.486260 | 4,331 | 20.35 |
| sx-superuser | — | — | — | — | **SKIPPED** (initial edges > 200 k NX cutoff) |

---

## Standalone CUDA static (`cuda_static/`, initial graph only)

| Graph | Modularity | Communities | Time | Status |
|-------|-----------:|------------:|-----:|--------|
| CollegeMsg | 0.000181 | 192 | 1,205 ms | completed (degenerate) |
| sx-mathoverflow | — | — | — | **DNF** (log truncated mid-run) |
| sx-askubuntu | — | — | — | **DNF** (log truncated mid-run) |
| sx-superuser | — | — | — | **DNF** (log truncated mid-run) |

Only CollegeMsg produced a final line, and it is degenerate (Q≈0.0002, the
graph over-merged to 192 trivial communities). This standalone binary is a
*different* build from the static phase embedded in the dynamic binaries below,
and the two disagree sharply for CollegeMsg (192 comms / Q≈0 here vs 264 comms /
Q≈0.25 inside the edge-based binary) — worth reconciling.

---

## CUDA dynamic — edge-based (`normal/`)

"Static-inside" = the from-scratch static phase run by the dynamic binary on the
initial graph (its Q is the warm-start Q reported before the first batch).
"Final" columns = the last batch's `--- Final Comparison ---` (state after all
5 insertion batches). Time = summed across all 5 batches.

| Graph | Static-inside Q / comms / time | Naive Q (comms, total t) | Frontier Q (comms, total t) | Delta Q (comms, total t) |
|-------|-------------------------------|--------------------------|-----------------------------|--------------------------|
| CollegeMsg | 0.250 / 264 / 3.1 s | 0 (1, 1.6 s) | 0 (1, 1.8 s) | 0 (1, 1.9 s) |
| sx-mathoverflow | 0.279 / 5,140 / 575 s | 0.00047 (43, 98 s) | 0.00048 (44, 104 s) | 0.00048 (43, 99 s) |
| sx-askubuntu | 0.425 / 27,692 / **3,537 s** | 0.0530 (2,111, 2,972 s) | 0.0088 (2,237, 2,982 s) | 0.0531 (3,091, 2,938 s) |
| sx-superuser | — not run — | — | — | — |

## CUDA dynamic — node-based (`node_based/`)

| Graph | Static-inside Q / comms / time | Naive Q (comms, total t) | Frontier Q (comms, total t) | Delta Q (comms, total t) |
|-------|-------------------------------|--------------------------|-----------------------------|--------------------------|
| CollegeMsg | 0.005 / 95 / 0.3 s | 0 (1, 0.5 s) | 0 (1, 0.6 s) | 0 (1, 0.8 s) |
| sx-mathoverflow | 0.026 / 4,479 / 41 s | 0.00045 (39, 4.7 s) | 0.00045 (39, 5.5 s) | 0.00045 (39, 3.3 s) |
| sx-askubuntu | 0.415 / 31,750 / 619 s | **0.2846** (1,278, 683 s) | **0.2844** (1,343, 717 s) | **0.2846** (1,045, 678 s) |
| sx-superuser | — not run — | — | — | — |

---

## Best CUDA vs NetworkX (where both completed)

"Best CUDA" = highest final-batch True Q across all variants/kernels for that graph.

| Graph | NX Q | Best CUDA Q | Best variant | Q gap | NX comms | CUDA comms |
|-------|-----:|------------:|--------------|------:|---------:|-----------:|
| CollegeMsg | 0.257 | 0.000 | (all collapse to 1) | **−0.257** | 289 | 1 |
| sx-mathoverflow | 0.307 | 0.0005 | any | **−0.307** | 5,783 | ~40 |
| sx-askubuntu | 0.486 | 0.285 | node-based naive/delta | **−0.201** | 4,331 | ~1,000 |
| sx-superuser | n/a | n/a | — | — | — | — |

No CUDA configuration comes within 0.2 of NetworkX on any temporal graph.
mathoverflow and CollegeMsg are total collapses (Q≈0); sx-askubuntu is the only
graph where CUDA holds a non-trivial partition, and it still loses 0.20
modularity.

## Speed, on comparable work (initial graph build)

NetworkX and the CUDA static-inside phase both build the same 80 %-initial
graph from scratch, so this is the only apples-to-apples timing. (The dynamic
batches are *additional* incremental work NetworkX never does.)

| Graph | NX (s) | Node-based static-inside (s) | Edge-based static-inside (s) | NX vs node-based |
|-------|-------:|-----------------------------:|-----------------------------:|------------------|
| CollegeMsg | 0.34 | 0.31 | 3.10 | ~tie |
| sx-mathoverflow | 11.57 | 41.3 | 575 | NX **3.6× faster** |
| sx-askubuntu | 20.35 | 619 | 3,537 | NX **30× faster** |

The CUDA static phase is *slower* than NetworkX on every graph above
CollegeMsg, dramatically so for the edge-based kernel.

---

## Modularity collapse trajectory (the core problem)

Per-batch modularity, naive variant, "before update" → "after update":

| Graph / kernel | B1 | B2 | B3 | B4 | B5 |
|----------------|----|----|----|----|----|
| mathover edge — before | 0.279 | 0.001 | 0.001 | 0.001 | 0.001 |
| mathover edge — after | **0.003** | 0.001 | 0.001 | 0.000 | 0.000 |
| CollegeMsg edge — before | 0.250 | 0.001 | 0.000 | 0.000 | 0.001 |
| CollegeMsg edge — after | **0.009** | 0.000 | 0.000 | 0.000 | 0.000 |
| askubuntu node — before | 0.415 | 0.382 | 0.379 | 0.281 | 0.277 |
| askubuntu node — after (naive) | 0.301 | 0.017 | 0.010 | 0.009 | 0.285 |

- **The first batch update is where quality dies.** On mathover edge-based, a
  healthy 0.279 partition drops to 0.003 after one batch of insertions and never
  recovers. CollegeMsg behaves the same way (0.250 → 0.009 → collapse to a
  single community).
- **sx-askubuntu node-based is the exception.** Its warm-start ("before") Q stays
  high across batches (0.415 → 0.277) because the *frontier* and *delta* variants
  retain modularity even when *naive* collapses within a batch (naive drops to
  0.01–0.02 mid-stream, but the carried-forward partition is the better
  frontier/delta one). All three reconverge to ≈0.285 by the final batch.

---

## Key observations

1. **Temporal-update collapse is the dominant failure.** The static base
   partitions are mostly fine; the incremental local-move + aggregation on each
   insertion batch over-merges communities (avalanche), driving Q to ~0. This is
   the same over-eager-merge dynamic diagnosed in
   [nx_vs_cuda_comparison1.md](nx_vs_cuda_comparison1.md) §"Static collapsed on
   large sparse graphs" and
   [Profiling_and_Improving_edge_louvain.md](Profiling_and_Improving_edge_louvain.md):
   the move threshold of `1e-12` lets marginally-positive moves trigger cascades.
2. **Node-based kernel is strictly better here** — 5–6× faster than edge-based on
   the static phase (askubuntu 619 s vs 3,537 s; mathover 41 s vs 575 s) and the
   only variant retaining a usable partition (askubuntu Q≈0.285).
3. **Edge-based is impractically slow on large graphs.** Its static-inside phase
   on sx-askubuntu alone took ~59 minutes, with ~49 more minutes per dynamic
   variant — ~2.5 hours total for one graph.
4. **frontier/delta retain modularity better than naive across batches** on the
   one graph that works (askubuntu node-based), consistent with their design
   intent of touching fewer vertices and avoiding global re-cascade.
5. **Coverage is incomplete.** sx-superuser produced no NetworkX, no edge-based,
   and no node-based output, and standalone static was truncated — it is
   effectively unbenchmarked. Standalone `static_louvain.exe` only finished
   CollegeMsg.

## Required follow-up

1. **Fix the per-batch over-merge first** — raise the local-move acceptance
   threshold from `1e-12` to `1.0 / m_edges` in both kernels (the fix already
   recommended for the static collapse). Temporal results are not publishable
   until the first-batch collapse is gone.
2. **Re-run sx-superuser end to end** (or document why it is excluded — likely
   too slow for the edge-based kernel; try node-based only).
3. **Reconcile the standalone-static vs static-inside discrepancy** on
   CollegeMsg (192/Q≈0 vs 264/Q≈0.25) and get standalone static to finish the
   three large graphs.
4. **Drop the edge-based kernel from large-temporal runs** until its runtime is
   addressed; use node-based as the working baseline.
5. After fixes, regenerate this table — target keeping the warm-start Q (≈0.28
   on askubuntu, ≈0.28 on mathover initial) instead of collapsing it.
