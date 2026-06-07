# DF Louvain (GPU) vs NetworkX — SNAP Temporal Graphs

Comparison of the from-scratch [`df_louvain.cu`](../algorithm/df_louvain.cu)
(Static + ND + DF + DS in one GPU process) against the NetworkX CPU baseline
([`nx_louvain.py`](../algorithm/nx_louvain.py), which re-runs Louvain from
scratch after each batch), on the SNAP **temporal** graphs.

Data from `real_graphs/snap_temporal/outputs/{df,networkx}/`. Each graph: earliest
80 % of edges as the initial graph + 5 chronological insertion batches. Seed 42.
**All four graphs now have full per-batch NetworkX data** (same converted input,
both tools process the initial graph + the 5 batches).

---

## TL;DR

1. **At its peak df is within ~0.02–0.03 of NetworkX on the three structured Q&A
   graphs**, while running **100–175× faster** end-to-end.
2. **But df drifts downward over batches** — quality peaks at batch 1–3 then
   declines (askubuntu DF 0.428→0.350, superuser DS 0.391→0.238), whereas
   NetworkX (from scratch each time) stays flat. Incremental warm-starting
   accumulates error; periodic full recomputation would be needed.
3. **CollegeMsg is an outright failure** — df stays **negative** (−0.16 → −0.095)
   where NetworkX gets +0.26. Smallest/densest/weakest-structure graph; the
   parallel-local-move sub-singleton bug ([[df-louvain-sub-singleton-q]]).
4. **DF is the most stable variant; ND and DS are volatile** batch-to-batch.
5. **df under-coarsens** — it always ends with more communities than NetworkX
   (askubuntu b5: 8.5k vs 2.3k; superuser b5: 5.3k vs 1.8k).

---

## NetworkX baseline (re-run from scratch each batch)

| Graph | Nodes | Init edges | Q init | Q batch 5 | Comms init→b5 | Total time (6 runs) |
|-------|------:|-----------:|-------:|----------:|--------------:|--------------------:|
| CollegeMsg | 1,899 | 11,070 | 0.2569 | 0.2642 | 289 → 13 | 1.27 s |
| sx-mathoverflow | 24,759 | 150,388 | 0.3073 | 0.3097 | 5,783 → 81 | 33.20 s |
| sx-askubuntu | 157,222 | 364,552 | 0.4608 | 0.4677 | 33,132 → 2,273 | 172.55 s |
| sx-superuser | 192,409 | 571,656 | 0.4118 | 0.4176 | 46,488 → 1,833 | 231.85 s |

NetworkX modularity is essentially **flat across batches** while its community
count plummets (it re-coarsens hard each time).

---

## df_louvain — Static (initial graph)

| Graph | Static Q | Comms | Passes | Time (ms) |
|-------|---------:|------:|-------:|----------:|
| CollegeMsg | **−0.158** | 426 | 2 | 121 |
| sx-mathoverflow | 0.270 | 6,597 | 3 | 213 |
| sx-askubuntu | 0.364 | 43,998 | 3 | 630 |
| sx-superuser | 0.119 | 52,358 | 3 | 831 |

---

## Per-batch modularity — df (all 3 variants) vs NetworkX

**CollegeMsg** — df never reaches positive Q (poisoned warm start):

| Batch | ND | DF | DS | **NetworkX** |
|------:|-----:|-----:|-----:|---------:|
| 1 | −0.200 | −0.145 | −0.193 | 0.250 |
| 3 | −0.210 | −0.103 | −0.189 | 0.251 |
| 5 | −0.207 | **−0.095** | −0.195 | 0.264 |

**sx-mathoverflow** — DF tracks NetworkX within ~0.02 throughout:

| Batch | ND | DF | DS | **NetworkX** |
|------:|-----:|-----:|-----:|---------:|
| 1 | 0.297 | 0.284 | 0.295 | 0.312 |
| 2 | 0.195 | 0.285 | 0.197 | 0.299 |
| 3 | 0.280 | 0.289 | 0.087 | 0.305 |
| 4 | 0.219 | 0.290 | 0.232 | 0.315 |
| 5 | 0.277 | **0.291** | 0.070 | 0.310 |

**sx-askubuntu** — df peaks at batch 2–3 (DS 0.443, DF 0.428) then drifts down:

| Batch | ND | DF | DS | **NetworkX** |
|------:|-----:|-----:|-----:|---------:|
| 1 | 0.378 | 0.428 | 0.416 | 0.460 |
| 2 | 0.232 | **0.428** | 0.432 | 0.465 |
| 3 | 0.397 | 0.411 | **0.443** | 0.466 |
| 4 | 0.408 | 0.299 | 0.376 | 0.465 |
| 5 | 0.371 | 0.350 | 0.365 | 0.468 |

**sx-superuser** — DS peaks 0.391 at batch 4, then collapses; df stays ~0.03–0.10 under NX:

| Batch | ND | DF | DS | **NetworkX** |
|------:|-----:|-----:|-----:|---------:|
| 1 | 0.372 | 0.378 | 0.376 | 0.411 |
| 2 | 0.257 | 0.326 | 0.322 | 0.415 |
| 3 | 0.366 | 0.352 | 0.387 | 0.417 |
| 4 | 0.173 | 0.336 | **0.391** | 0.416 |
| 5 | 0.372 | 0.350 | 0.238 | 0.418 |

---

## Quality gap — df best vs NetworkX

| Graph | df peak Q (variant, batch) | NX same batch | peak gap | df Q batch 5 | NX batch 5 | b5 gap |
|-------|----------------------------|--------------:|---------:|-------------:|-----------:|-------:|
| CollegeMsg | −0.095 (DF, b5) | 0.264 | **0.359** | −0.095 | 0.264 | 0.359 |
| sx-mathoverflow | 0.291 (DF, b5) | 0.310 | **0.019** | 0.291 | 0.310 | 0.019 |
| sx-askubuntu | 0.443 (DS, b3) | 0.466 | **0.023** | 0.371 | 0.468 | 0.097 |
| sx-superuser | 0.391 (DS, b4) | 0.416 | **0.025** | 0.372 | 0.418 | 0.046 |

On the three structured graphs df's **best** result is within 0.02–0.03 of
NetworkX, but the **batch-5** gap is larger (0.02–0.10) because df drifts down.

---

## Speed — end-to-end (Static + 5 batches, DF trajectory)

| Graph | df total | NX total | **speedup** |
|-------|---------:|---------:|------------:|
| CollegeMsg | ~0.16 s | 1.27 s | ~8× |
| sx-mathoverflow | ~0.26 s | 33.20 s | **~126×** |
| sx-askubuntu | ~1.08 s | 172.55 s | **~160×** |
| sx-superuser | ~1.32 s | 231.85 s | **~175×** |

Per update, a df batch is **4–135 ms** vs NetworkX's **0.2–47 s** per from-scratch
re-run. The df totals above include *all* of Static+ND+DF+DS; the DF-only slice is
smaller still.

---

## Critical observations

1. **df is competitive at peak on the structured graphs** (within 0.02–0.03 of
   NetworkX) at **100–175× the speed** — the value proposition is real where the
   graph has structure.
2. **Downward drift across batches is the new headline issue.** df peaks early
   (batch 1–3) then declines: askubuntu DF 0.428→0.350, superuser DS 0.391→0.238.
   NetworkX, recomputing from scratch, stays flat. Warm-starting from an
   increasingly stale/over-fragmented partition compounds error — the DF paper's
   own remedy is an occasional full Static recompute (it suggests every ~1000
   batches; here drift is visible within 5, a symptom of the unfixed quality
   bugs).
3. **Variant behaviour:** **DF** is the steadiest; **DS** can hit the highest
   single value (askubuntu 0.443, superuser 0.391) but also collapses (mathover
   b5 0.070, superuser b5 0.238); **ND** is the most erratic (superuser b4 0.173).
   DS marks 20k–186k vertices vs DF's ~3k–19k, so it is costlier *and* less
   reliable here.
4. **CollegeMsg fails outright** (negative Q) — dense/weak-structure → the
   sub-singleton local-move bug. Same root cause as the ER random graphs.
5. **df under-coarsens** — always more communities than NetworkX at the same
   batch, mirroring the static-SNAP finding ([[df-louvain-undercoarsening]]).

---

## Recommended next step

Two fixes, in order of leverage: (a) the dense-graph sub-singleton local-move bug
(unblocks CollegeMsg and stops the drift from compounding); (b) the
under-coarsening / early pass-stop (raises peak Q toward NetworkX). After both,
re-run and check whether df holds quality across all 5 batches instead of drifting.
The speed headroom (100–175×) means even a periodic full Static recompute every
few batches would still leave df far ahead of NetworkX.
