# DF Louvain (GPU) vs NetworkX — Static SNAP Graphs

Comparison of [`df_louvain.cu`](../algorithm/df_louvain.cu) against the NetworkX
CPU baseline ([`nx_louvain.py`](../algorithm/nx_louvain.py)) on the **static**
SNAP graphs (collaboration / social / email networks). These converted inputs
carry 0 batches, so this exercises df_louvain's **Static** path only — a pure
static-Louvain quality/speed comparison.

Data from `real_graphs/snap/outputs/{df,networkx}/`. Seed 42.

---

## TL;DR

1. **df is always positive (no collapse) but consistently lower-Q than NetworkX**,
   by **0.05–0.36**. Worst on the very sparse collaboration graphs (ca-HepTh,
   ca-GrQc), closest on the densest (facebook, email-Enron).
2. **df under-coarsens badly** — it stops merging too early and leaves **3×–110×
   more communities** than NetworkX (com-dblp: 24,225 vs 221). That over-fine
   partition is exactly why its modularity is lower.
3. **df is dramatically faster** — 2×–252×, growing with graph size: com-amazon
   232 ms vs NetworkX 58 s (**252×**), com-dblp 616 ms vs 80 s (**130×**).

---

## Full comparison

| Graph | Nodes | Edges | df Q | NX Q | **Q gap** | df comms | NX comms | comm ratio | df t | NX t | speedup |
|-------|------:|------:|-----:|-----:|----------:|---------:|---------:|-----------:|-----:|-----:|--------:|
| ca-GrQc      | 5,241   | 14,484    | 0.5510 | 0.8613 | **−0.310** | 2,065  | 388 | 5.3× | 149 ms | 0.32 s | 2.1× |
| facebook     | 4,039   | 88,234    | 0.7825 | 0.8347 | −0.052 | 89     | 14  | 6.4× | 132 ms | 0.79 s | 6.0× |
| ca-HepTh     | 9,875   | 25,973    | 0.4075 | 0.7669 | **−0.359** | 3,676  | 482 | 7.6× | 149 ms | 0.87 s | 5.8× |
| ca-HepPh     | 12,006  | 118,489   | 0.5525 | 0.6581 | −0.106 | 2,426  | 312 | 7.8× | 195 ms | 1.73 s | 8.9× |
| ca-AstroPh   | 18,771  | 198,050   | 0.5253 | 0.6254 | −0.100 | 1,898  | 328 | 5.8× | 188 ms | 3.39 s | 18× |
| email-Enron  | 36,692  | 183,831   | 0.5595 | 0.6122 | −0.053 | 4,131  | 1,219 | 3.4× | 215 ms | 3.77 s | 18× |
| com-amazon   | 334,863 | 925,872   | 0.8536 | 0.9259 | −0.072 | 6,606  | 238 | **27.8×** | 232 ms | 58.5 s | **252×** |
| com-dblp     | 317,080 | 1,049,866 | 0.7413 | 0.8221 | −0.081 | 24,225 | 221 | **109.6×** | 616 ms | 80.0 s | **130×** |

(df t = Static time; NX t = single Louvain run; speedup = NX t / df t.)

---

## Critical observations

1. **Under-coarsening is the core quality problem.** df produces far more
   communities than NetworkX everywhere, and pathologically so on the large
   sparse graphs (com-dblp 24,225 vs 221, com-amazon 6,606 vs 238). A finer
   partition has lower modularity, which fully accounts for the Q gap. df is *not*
   collapsing (Q is healthy-positive) — it is **stopping the agglomeration too
   early**.

2. **Likely cause: the pass-level stop conditions.** df breaks out of the
   pass loop when a pass shrinks the community count by < 20 %
   (`AGG_TOLERANCE = 0.8`) or when local-moving converges in ≤ 1 iteration. On
   graphs where coarsening is gradual, this halts after a few passes
   (com-dblp/amazon used 4 passes) while NetworkX keeps merging to a far coarser
   partition. Worth testing: lower `AGG_TOLERANCE` (e.g. 0.95–0.99) and/or raise
   `MAX_PASSES`, and confirm the community count drops toward NetworkX's.

3. **The gap is worst on sparse collaboration graphs** (ca-HepTh −0.36, ca-GrQc
   −0.31; avg degree ≈ 3–5). These have many small, well-separated communities;
   df's early stop leaves them fragmented. On denser graphs (facebook,
   email-Enron) df is within 0.05.

4. **No collapse, no negative Q** here — unlike CollegeMsg/ER graphs. The static
   SNAP graphs are sparse with strong structure, so the parallel local-move
   behaves; the issue is purely insufficient coarsening, not degradation.

5. **Speed is a decisive df win and scales with size.** On the two ~1 M-edge
   graphs df is 130–252× faster (sub-second vs ~1 minute). Even after fixing the
   coarsening (which will add passes and some time), df should remain far ahead.

---

## Recommended next step

Investigate the early-stop: re-run com-dblp / com-amazon with a higher
`AGG_TOLERANCE` and larger `MAX_PASSES`, and check whether the community count
falls toward NetworkX's ~200–300 and Q rises toward 0.82–0.93. This under-
coarsening is **separate** from the dense-graph sub-singleton bug
([[df-louvain-sub-singleton-q]]) seen on CollegeMsg/ER graphs — both should be
fixed before publishing quality numbers, but this one looks like a tuning/stop-
condition issue rather than a correctness defect in the move kernel.
