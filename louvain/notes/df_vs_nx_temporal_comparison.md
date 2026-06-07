# DF Louvain (GPU) vs NetworkX — SNAP Temporal Graphs

Comparison of the from-scratch [`df_louvain.cu`](../algorithm/df_louvain.cu)
(Static + ND + DF + DS in one GPU process) against the NetworkX CPU baseline
([`nx_louvain.py`](../algorithm/nx_louvain.py), which re-runs Louvain from
scratch after each batch), on the SNAP **temporal** graphs.

Data from `real_graphs/snap_temporal/outputs/{df,networkx}/`. Each graph: earliest
80 % of edges as the initial graph + 5 chronological insertion batches. Seed 42.
**Post-fix** (verified-commit-under-lock local move — see [bugfixes.md](bugfixes.md)).

Each table reports **modularity per batch for every algorithm**. The `initial`
row is the partition of the 80 % initial graph: for df that is the single Static
run (so ND/DF/DS share it), for NetworkX it is its first Louvain run. Rows 1–5 are
after each insertion batch (df warm-starts; NetworkX recomputes from scratch).

---

## TL;DR

1. **CollegeMsg is fixed** — Static is now **+0.246** (was −0.158) and every batch
   tracks NetworkX at ~0.25–0.26.
2. **On sx-mathoverflow df beats NetworkX** — ND/DS reach **0.321** vs 0.310, ~108× faster.
3. **sx-askubuntu / sx-superuser are competitive and stable** (the earlier
   downward drift is gone), within ~0.04–0.05 of NetworkX, at **180–193×** speed.
4. **Residual:** Static under-coarsens on askubuntu/superuser (dynamic variants
   recover it); superuser ND collapses at batch 5.

---

## CollegeMsg — modularity per batch

initial graph: 1,899 nodes, 11,070 edges → 13,838 edges after batch 5.

| Batch | df ND | df DF | df DS | NetworkX |
|------:|------:|------:|------:|---------:|
| initial / Static | 0.2455 | 0.2455 | 0.2455 | 0.2569 |
| 1 | 0.2623 | 0.2524 | 0.2621 | 0.2499 |
| 2 | 0.2618 | 0.2501 | 0.2610 | 0.2562 |
| 3 | 0.2579 | 0.2456 | 0.2573 | 0.2510 |
| 4 | 0.2551 | 0.2437 | 0.2554 | 0.2563 |
| 5 | 0.2532 | 0.2425 | 0.2547 | 0.2642 |

df tracks NetworkX within ~0.01 throughout; ND/DS actually lead NX on batches 1–3.

## sx-mathoverflow — modularity per batch

24,759 nodes, 150,388 → 187,986 edges.

| Batch | df ND | df DF | df DS | NetworkX |
|------:|------:|------:|------:|---------:|
| initial / Static | 0.2699 | 0.2699 | 0.2699 | 0.3073 |
| 1 | 0.3069 | 0.2964 | 0.3066 | 0.3115 |
| 2 | 0.3123 | 0.2989 | 0.3120 | 0.2986 |
| 3 | 0.3171 | 0.3047 | 0.3170 | 0.3048 |
| 4 | 0.3190 | 0.3066 | 0.3184 | 0.3151 |
| 5 | **0.3213** | 0.3092 | 0.3205 | 0.3097 |

df ND/DS **exceed NetworkX from batch 2 onward**.

## sx-askubuntu — modularity per batch

157,222 nodes, 364,552 → 455,691 edges.

| Batch | df ND | df DF | df DS | NetworkX |
|------:|------:|------:|------:|---------:|
| initial / Static | 0.2154 | 0.2154 | 0.2154 | 0.4608 |
| 1 | 0.4206 | 0.4206 | 0.4211 | 0.4599 |
| 2 | 0.4289 | 0.4235 | 0.4291 | 0.4651 |
| 3 | 0.4305 | 0.4256 | 0.4305 | 0.4657 |
| 4 | 0.4301 | 0.4258 | 0.4307 | 0.4649 |
| 5 | 0.4304 | 0.4254 | 0.4306 | 0.4677 |

Static under-coarsens (0.2154) but the first batch's full reprocess recovers to
~0.42; thereafter df is flat and stable, ~0.04 under NetworkX.

## sx-superuser — modularity per batch

192,409 nodes, 571,656 → 714,570 edges.

| Batch | df ND | df DF | df DS | NetworkX |
|------:|------:|------:|------:|---------:|
| initial / Static | 0.3284 | 0.3284 | 0.3284 | 0.4118 |
| 1 | 0.3925 | 0.3952 | 0.3913 | 0.4108 |
| 2 | 0.4006 | 0.3978 | 0.4000 | 0.4147 |
| 3 | 0.4034 | 0.3992 | 0.2965 | 0.4173 |
| 4 | 0.4041 | 0.3994 | 0.3266 | 0.4163 |
| 5 | **0.2330** | 0.3245 | 0.3697 | 0.4176 |

Stable ~0.39–0.40 mid-stream, but **ND collapses at batch 5 (0.233)** and DS dips
on batches 3–5 — a residual instability (DF is the safest variant here).

---

## NetworkX baseline summary

| Graph | Nodes | Init edges | Q init | Q b5 | Comms init→b5 | Total (6 runs) |
|-------|------:|-----------:|-------:|-----:|--------------:|---------------:|
| CollegeMsg | 1,899 | 11,070 | 0.2569 | 0.2642 | 289 → 13 | 1.27 s |
| sx-mathoverflow | 24,759 | 150,388 | 0.3073 | 0.3097 | 5,783 → 81 | 33.20 s |
| sx-askubuntu | 157,222 | 364,552 | 0.4608 | 0.4677 | 33,132 → 2,273 | 172.55 s |
| sx-superuser | 192,409 | 571,656 | 0.4118 | 0.4176 | 46,488 → 1,833 | 231.85 s |

## Quality gap & speed (batch 5)

| Graph | df best Q (variant) | NX Q | gap | df total | NX total | speedup |
|-------|---------------------|-----:|----:|---------:|---------:|--------:|
| CollegeMsg | 0.255 (DS) | 0.264 | −0.009 | ~0.13 s | 1.27 s | ~10× |
| sx-mathoverflow | 0.321 (ND) | 0.310 | **+0.011** ✓ | ~0.31 s | 33.20 s | **~108×** |
| sx-askubuntu | 0.431 (DS) | 0.468 | −0.037 | ~0.95 s | 172.55 s | **~182×** |
| sx-superuser | 0.370 (DS) | 0.418 | −0.048 | ~1.20 s | 231.85 s | **~193×** |

---

## Observations

1. **The collapse fix carries through to the dynamic variants** — CollegeMsg now
   tracks NetworkX within ~0.01; the warm-start is no longer poisoned.
2. **df matches or beats NetworkX where structure is strong** (mathoverflow ND
   0.321 > NX 0.310) at two orders of magnitude less time.
3. **Inter-batch drift is largely resolved** — askubuntu holds 0.42–0.43 flat
   (pre-fix it drifted 0.428 → 0.350).
4. **Residual issues (secondary):** Static under-coarsens on the large temporal
   initials (askubuntu/superuser, 3 passes — dynamic variants recover it; try
   `AGG_TOLERANCE`→1.0); superuser batch-5 ND collapse (use DS/DF there).
5. **DF is the steadiest variant; ND usually highest but occasionally spikes down;
   DS strong with two dips (superuser b3–b5).**
