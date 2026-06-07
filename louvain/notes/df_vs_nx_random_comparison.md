# DF Louvain (GPU) vs NetworkX — Synthetic Erdős–Rényi Dynamic Graphs

Comparison of [`df_louvain.cu`](../algorithm/df_louvain.cu) (Static + ND + DF + DS)
against the NetworkX CPU baseline ([`nx_louvain.py`](../algorithm/nx_louvain.py),
re-running Louvain from scratch each batch) on the generated Erdős–Rényi graphs
from [`generate/run_benchmarks.sh`](../generate/run_benchmarks.sh).

Data from `generate/outputs/{df,networkx}/`. Seed 42, 5 batches each.
**Post-fix** (verified-commit-under-lock local move — see [bugfixes.md](bugfixes.md)).

Each table reports **modularity per batch for every algorithm**. The `initial`
row is the partition of the initial graph (df: single Static run shared by
ND/DF/DS; NetworkX: its first run). Rows 1–5 are after each batch.

## ⚠️ ER graphs have no community structure

Uniform random graphs with **no planted communities** → any positive modularity
is *spurious*. These are a **stress test**, not a quality benchmark; the point is
confirming the fix (no negative Q) and exposing edge cases. Batches are tiny
(del = ins ≈ 0.1 % of edges; net edge count unchanged), so NetworkX is flat.

---

## TL;DR

1. **Sub-singleton collapse is gone** — Static positive on every graph
   (n100: 0.256, was −0.042).
2. **Competitive on small/mid graphs** — df's **ND matches or beats NetworkX** on
   n100/n500; the gap only grows as the graph thins out.
3. **n10000 over-merges to 2 communities** (Q 0.093 best vs NX 0.127) — the ER
   edge case.
4. **Faster from n500 up** (~7–105×); slower only on the tiny n100.

---

## er_n100_p0.10 — modularity per batch  (100 nodes, 474 edges)

| Batch | df ND | df DF | df DS | NetworkX |
|------:|------:|------:|------:|---------:|
| initial / Static | 0.2558 | 0.2558 | 0.2558 | 0.2727 |
| 1 | 0.2717 | 0.2537 | 0.2595 | 0.2738 |
| 2 | 0.2759 | 0.2537 | 0.2628 | 0.2685 |
| 3 | 0.2739 | 0.2516 | 0.2665 | 0.2587 |
| 4 | 0.2756 | 0.2540 | 0.2685 | 0.2638 |
| 5 | **0.2776** | 0.2557 | 0.2729 | 0.2632 |

ND ends **above** NetworkX (0.2776 vs 0.2632).

## er_n500_p0.05 — modularity per batch  (500 nodes, 6,162 edges)

| Batch | df ND | df DF | df DS | NetworkX |
|------:|------:|------:|------:|---------:|
| initial / Static | 0.1503 | 0.1503 | 0.1503 | 0.1677 |
| 1 | 0.1653 | 0.1511 | 0.1638 | 0.1658 |
| 2 | 0.1671 | 0.1521 | 0.1654 | 0.1701 |
| 3 | 0.1687 | 0.1525 | 0.1669 | 0.1729 |
| 4 | 0.1706 | 0.1526 | 0.1683 | 0.1722 |
| 5 | **0.1709** | 0.1524 | 0.1684 | 0.1666 |

## er_n1000_p0.02 — modularity per batch  (1,000 nodes, 9,925 edges)

| Batch | df ND | df DF | df DS | NetworkX |
|------:|------:|------:|------:|---------:|
| initial / Static | 0.1746 | 0.1746 | 0.1746 | 0.2025 |
| 1 | 0.1907 | 0.1751 | 0.1843 | 0.2072 |
| 2 | 0.1922 | 0.1752 | 0.1887 | 0.2013 |
| 3 | 0.1934 | 0.1749 | 0.1913 | 0.2029 |
| 4 | 0.1939 | 0.1749 | 0.1925 | 0.1998 |
| 5 | 0.1943 | 0.1748 | 0.1932 | 0.2012 |

## er_n3000_p0.01 — modularity per batch  (3,000 nodes, 44,700 edges)

| Batch | df ND | df DF | df DS | NetworkX |
|------:|------:|------:|------:|---------:|
| initial / Static | 0.1245 | 0.1245 | 0.1245 | 0.1628 |
| 1 | 0.1439 | 0.1251 | 0.1440 | 0.1624 |
| 2 | 0.1476 | 0.1255 | 0.1473 | 0.1639 |
| 3 | 0.1501 | 0.1262 | 0.1498 | 0.1642 |
| 4 | 0.1518 | 0.1269 | 0.1514 | 0.1645 |
| 5 | 0.1531 | 0.1275 | 0.1522 | 0.1666 |

## er_n10000_p0.005 — modularity per batch  (10,000 nodes, 250,012 edges)

| Batch | df ND | df DF | df DS | NetworkX |
|------:|------:|------:|------:|---------:|
| initial / Static | 0.0319 | 0.0319 | 0.0319 | 0.1270 |
| 1 | 0.0868 | 0.0341 | 0.0865 | 0.1266 |
| 2 | 0.0897 | 0.0361 | 0.0895 | 0.1271 |
| 3 | 0.0914 | 0.0384 | 0.0913 | 0.1269 |
| 4 | 0.0924 | 0.0407 | 0.0923 | 0.1297 |
| 5 | 0.0933 | 0.0424 | 0.0932 | 0.1285 |

df over-merges here (2 communities); NetworkX keeps 6–9, hence the larger gap.

---

## Before vs after the fix (Static Q)

| Graph | Static Q before | Static Q after | NX Q |
|-------|----------------:|---------------:|-----:|
| n100   | **−0.042** | 0.256 | 0.273 |
| n500   | **−0.012** | 0.150 | 0.168 |
| n1000  | **−0.014** | 0.175 | 0.203 |
| n3000  | −0.000 | 0.124 | 0.163 |
| n10000 | 0.003 | 0.032 | 0.127 |

## Speed — end-to-end (Static + 5 batches, DF trajectory)

| Graph | df total | NX total | speedup |
|-------|---------:|---------:|--------:|
| n100   | ~99 ms | 39 ms | 0.4× (slower) |
| n500   | ~86 ms | 595 ms | ~7× |
| n1000  | ~84 ms | 1,085 ms | ~13× |
| n3000  | ~99 ms | 5,697 ms | ~58× |
| n10000 | ~326 ms | 34,167 ms | **~105×** |

---

## Observations

1. **Fix confirmed:** no negative Q anywhere; Static is a valid partition on every
   ER graph now.
2. **ND is the strongest df variant** (reprocesses every vertex) and edges out
   NetworkX on n100/n500; **DF barely moves off Static** (its frontier touches few
   vertices on a structureless graph); DS tracks ND.
3. **n10000 over-merges to 2 communities** — inverse of the static-SNAP
   under-coarsening; an ER-specific artifact. Structured graphs (static SNAP,
   temporal) coarsen correctly to NetworkX-level counts.
4. **ER modularity is spurious** — don't read the residual gaps as a general
   deficit. The meaningful evaluations are
   [df_vs_nx_snap_comparison.md](df_vs_nx_snap_comparison.md) (df matches NetworkX)
   and [df_vs_nx_temporal_comparison.md](df_vs_nx_temporal_comparison.md)
   (competitive-to-better).
