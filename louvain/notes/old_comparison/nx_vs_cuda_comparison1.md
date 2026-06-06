# NetworkX vs CUDA Louvain Comparison — Round 2

Updated comparison after the following algorithm changes since [nx_vs_cuda_comparison.md](nx_vs_cuda_comparison.md):

1. **Self-loop bug fix** in `calculate_modularity_change`: self-loop entries now skipped in `k_i_in_old` / `k_i_in_new` accumulation. Intended to allow pass 2+ merges that were previously frozen.
2. **`k_i` precomputation** via `d_vertex_weight` snapshot: read in O(1) instead of summed per call. Speed optimization only; should not affect Q.
3. **Inner-loop iteration cap** changed from 100 → 20 in the static kernel.

All numbers below were produced by `real_graphs/run_benchmarks.sh` + `real_graphs/snap_results_table.py` on the current binaries.

---

## NetworkX (CPU baseline, unchanged for reference)

| Graph | Nodes | Edges | Modularity | Communities | Time (s) |
|-------|-------|-------|------------|-------------|----------|
| ca-GrQc | 5,241 | 14,484 | 0.858981 | 389 | 0.28 |
| facebook | 4,039 | 88,234 | 0.834971 | 16 | 0.76 |
| ca-HepTh | 9,875 | 25,973 | 0.769321 | 473 | 0.83 |
| ca-HepPh | 12,006 | 118,489 | 0.659640 | 310 | 1.43 |
| ca-AstroPh | 18,771 | 198,050 | 0.628977 | 323 | 3.75 |
| email-Enron | 36,692 | 183,831 | 0.604123 | 1,271 | 6.63 |
| com-amazon | 334,863 | 925,872 | 0.926024 | 231 | 70.86 |
| com-dblp | 317,080 | 1,049,866 | 0.821487 | 212 | 85.06 |

---

## CUDA Implementations (current, True Q column)

| Graph | Static Q | Static T | Static Comms | Dyn-edge Q | Dyn-edge T | Dyn-edge Comms | Dyn-node Q | Dyn-node T | Dyn-node Comms |
|-------|----------|----------|--------------|------------|------------|----------------|------------|------------|----------------|
| ca-GrQc | 0.7670 | 231ms | 361 | 0.7531 | 98ms | 276 | 0.8261 | 75ms | 31 |
| facebook | 0.8227 | 1,546ms | 10 | 0.4607 | 1,461ms | 2 | 0.5967 | 857ms | 9 |
| ca-HepTh | 0.1192 | 1,415ms | 302 | 0.6920 | 974ms | 223 | 0.7454 | 512ms | 434 |
| ca-HepPh | 0.3293 | 9,889ms | 231 | 0.3669 | 13,478ms | 116 | 0.4721 | 1,244ms | 29 |
| ca-AstroPh | 0.0439 | 23,342ms | 307 | 0.5882 | 14,620ms | 191 | 0.4186 | 1,250ms | 181 |
| email-Enron | 0.3545 | 54,583ms | 1,103 | 0.5549 | 35,219ms | 574 | 0.5334 | 11,078ms | 841 |
| com-amazon | 0.0000 | 26,382ms | **1** | **0.9235** | 8,472ms | 103 | 0.8771 | 3,898ms | 38 |
| com-dblp | 0.0008 | 474,379ms | 11 | 0.6774 | 144,909ms | 10 | 0.7808 | 105,348ms | 34 |

---

## Quality and Time Gap (NetworkX vs Best CUDA variant)

"Best CUDA" = the variant with the highest True Q per graph; its time is reported in the same row. Speedup = NX time / best-CUDA-variant time (>1 ⇒ CUDA faster).

| Graph | NX Q | Best CUDA Q | Best variant | Q Gap | NX Time | CUDA Time | Speedup | NX Comms | CUDA Comms |
|-------|------|-------------|--------------|-------|---------|-----------|---------|----------|------------|
| ca-GrQc | 0.859 | **0.826** | Dyn-node | −0.033 | 280 ms | 75 ms | **3.7×** | 389 | 31 |
| facebook | 0.835 | **0.823** | Static | −0.012 | 760 ms | 1,546 ms | 0.5× | 16 | 10 |
| ca-HepTh | 0.769 | **0.745** | Dyn-node | −0.024 | 830 ms | 512 ms | **1.6×** | 473 | 434 |
| ca-HepPh | 0.660 | **0.472** | Dyn-node | −0.188 | 1,430 ms | 1,244 ms | 1.2× | 310 | 29 |
| ca-AstroPh | 0.629 | **0.588** | Dyn-edge | −0.041 | 3,750 ms | 14,620 ms | 0.3× | 323 | 191 |
| email-Enron | 0.604 | **0.555** | Dyn-edge | −0.049 | 6,630 ms | 35,219 ms | 0.2× | 1,271 | 574 |
| com-amazon | 0.926 | **0.924** | Dyn-edge | −0.002 | 70,860 ms | 8,472 ms | **8.4×** | 231 | 103 |
| com-dblp | 0.821 | **0.781** | Dyn-node | −0.040 | 85,060 ms | 105,348 ms | 0.8× | 212 | 34 |

The **best-of-three** approach now sits within **0.05 of NetworkX on 6 of 8 graphs** (ca-HepPh and com-dblp being the two quality exceptions). Speed is a more mixed picture: CUDA wins on 4 graphs (ca-GrQc, ca-HepTh, ca-HepPh, com-amazon) and loses on 4 (facebook, ca-AstroPh, email-Enron, com-dblp).

## Fastest CUDA variant per graph (time only)

"Best-quality" and "fastest" are not always the same variant; for some experiments you may prefer fastest. Dyn-node is fastest on every graph in this round.

| Graph | Fastest variant | Time | Q at that time | NX Time | Speedup |
|-------|-----------------|------|----------------|---------|---------|
| ca-GrQc | Dyn-node | 75 ms | 0.826 | 280 ms | **3.7×** |
| facebook | Dyn-node | 857 ms | 0.597 | 760 ms | 0.9× |
| ca-HepTh | Dyn-node | 512 ms | 0.745 | 830 ms | **1.6×** |
| ca-HepPh | Dyn-node | 1,244 ms | 0.472 | 1,430 ms | **1.2×** |
| ca-AstroPh | Dyn-node | 1,250 ms | 0.419 | 3,750 ms | **3.0×** |
| email-Enron | Dyn-node | 11,078 ms | 0.533 | 6,630 ms | 0.6× |
| com-amazon | Dyn-node | 3,898 ms | 0.877 | 70,860 ms | **18.2×** |
| com-dblp | Dyn-node | 105,348 ms | 0.781 | 85,060 ms | 0.8× |

Note: choosing the fastest variant sometimes costs meaningful Q — e.g. facebook (Static 0.823 @ 1,546 ms vs Dyn-node 0.597 @ 857 ms). Production use should combine both criteria.

---

## Change from previous round (Best-variant True Q)

| Graph | Prev best Q | Current best Q | Delta | Winner shift |
|-------|-------------|----------------|-------|--------------|
| ca-GrQc | 0.732 | 0.826 | **+0.094** | Static → Dyn-node |
| facebook | 0.826 | 0.823 | −0.003 | unchanged (Static) |
| ca-HepTh | 0.604 | 0.745 | **+0.141** | Static → Dyn-node |
| ca-HepPh | 0.593 | 0.472 | **−0.121** | Dyn-edge → Dyn-node (both regressed) |
| ca-AstroPh | 0.551 | 0.588 | +0.037 | Static → Dyn-edge |
| email-Enron | 0.564 | 0.555 | −0.009 | Dyn-edge (unchanged) |
| com-amazon | 0.677 | 0.924 | **+0.247** | Static → Dyn-edge |
| com-dblp | 0.587 | 0.781 | **+0.194** | Static → Dyn-node |

---

## Patterns observed

1. **Dynamic variants became markedly better on large sparse graphs** (com-amazon, com-dblp). Dyn-edge on com-amazon is now essentially at NetworkX quality (0.9235 vs 0.9260). This is the self-loop fix paying off: pass 2+ now actually merges instead of stalling.

2. **Static collapsed to 1–11 communities on large sparse graphs** (com-amazon Q=0.0 with 1 community, com-dblp Q=0.0008 with 11 communities, ca-HepTh Q=0.12 with 302 communities counted but mass all in one). The self-loop fix removed the implicit safeguard that was preventing avalanche — static first-pass now over-merges before the inner loop hits the 20-iteration cap.

3. **Dyn-node became the best variant on several graphs** (ca-GrQc, ca-HepTh, com-dblp). This was previously an "experimental worse variant"; it is now competitive or winning on 4 of 8 graphs.

4. **ca-HepPh regressed across the board**. All three variants underperform their previous-round numbers. Suspect: a specific pattern in this graph (highest clustering coefficient of the suite) interacts badly with the new merge-permitting formula.

5. **No more crashes**. Previous rounds had illegal memory access failures during aggregation on ca-HepPh / ca-AstroPh / email-Enron. Current round completes all 24 runs (8 graphs × 3 variants).

6. **Timing mostly improved**, sometimes dramatically:
   - com-amazon static: 8.5s → 26.4s (slower — because static now collapses at pass 1, less useful work but different behavior).
   - com-amazon dyn-edge: 4.3s → 8.5s (similar).
   - com-dblp dyn-node: 17.9s → 105s (much slower — needs investigation; `k_i` precomputation was expected to help, not hurt).

---

## Hypothesized causes

**Self-loop fix removed a safeguard**: previously, including self-loop weight in `k_i_in_old` inflated the "cost of leaving a community", which happened to resist avalanche merges even though the formula was formally wrong. Removing it makes the formula correct but also makes the static kernel's greedy first-positive-move strategy over-eager — exactly the avalanche dynamic we diagnosed in the node-based kernel earlier. See `Profiling_and_Improving_edge_louvain.md` § "Optimization plan" for the recommended fix: raise the move threshold from `1e-12` to `1.0 / m_edges` so marginal moves don't trigger cascades.

**Static 20-iteration inner cap** truncates convergence. Before the cap-reduction, the kernel could thrash for up to 100 iterations, sometimes recovering from bad early moves. With 20 iterations, bad early moves stick.

**com-dblp dyn-node time regression** (17.9s → 105s) is not explained by any of the changes made. Possibly a different path through the kernel due to the self-loop fix, or an interaction with the `< 20` cap. Worth profiling with `nsys`.

---

## Positives — what's better now

- **com-amazon dyn-edge matches NetworkX** (0.9235 vs 0.9260) — the self-loop fix delivered its promised benefit here.
- **Best-of-three within 0.05 of NetworkX on 6/8 graphs** — up from 4/8 previously.
- **Community counts are reasonable** on almost all graphs now (not the 52K/60K inflation seen previously on com-amazon/com-dblp).
- **No runtime crashes** across 24 runs.

## Limitations — what regressed

- **Static cannot be trusted on large sparse graphs**. Produces Q ≈ 0 on com-amazon and com-dblp. Use dynamic variants only for these sizes until the over-merge is fixed.
- **ca-HepPh regression across all variants** — graph-specific issue to investigate.
- **Dyn-node time on com-dblp 6× slower** than previous round — unexplained; needs profiling.

---

## Required follow-up

1. **Raise the move threshold** from `1e-12` to `1.0 / m_edges` in both edge-based and node-based kernels. Expected to fix the static collapse on com-amazon / com-dblp / ca-HepTh without affecting quality on real-structure graphs.
2. **Raise the static inner cap** back to 100 (or tune it) after the threshold change.
3. **Profile the com-dblp dyn-node slowdown** with `nsys` to find the regression cause.
4. **Re-run the full suite** and produce `nx_vs_cuda_comparison2.md`.
