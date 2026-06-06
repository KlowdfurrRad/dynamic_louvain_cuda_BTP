# Profiling and Improving the Edge-Based Louvain Kernel

Scope: the edge-parallel CUDA kernel in `algorithm/cuda_dynamic_louvain.cu` (same kernel lives in `cuda_static_louvain.cu`). The node-based kernel is tracked separately because it has distinct issues (see the "avalanche" discussion elsewhere).

This document has two parts:
1. **Profile first** — how to quantify where time actually goes.
2. **Optimization plan** — prioritized fixes, effort, expected speedup, and whether each fix changes algorithmic output.

---

## Part 1 — Profile before you optimize

### 1.1 Nsight Systems (timeline — which *phase* is slow?)

A previous Nsight Systems trace exists at `BTP/report1.nsys-rep` from UGRC-I. Re-run with the current binary:

```bash
nsys profile --stats=true --output=edge_profile \
    ./dynamic_louvain outfile.txt < input.txt
```

What to inspect:
- **Per-kernel time** — is `louvain_kernel` >90% of total? Or do `aggregate_graph` / `combine_edges` (the Thrust sort) eat meaningful time?
- **Host↔device memcpy** — the between-pass modularity checks copy `h_csr_adj`, `h_community`, `h_community_degree` every pass. Is this ≥ a few hundred ms?
- **CPU idle gaps** — time between kernel launches on the host side (Thrust sort, `rebuild_csr_offsets`, Python/logging). A gap means the host is the bottleneck.
- **Per-pass breakdown** — add `cudaEventRecord` around each phase to separate "kernel time" from "aggregation time".

Expected finding: kernel is 70–90% of time on ER graphs but drops to 30–50% on large real graphs where aggregation/sort dominate.

### 1.2 Nsight Compute (kernel deep dive — what's the *kernel* stalled on?)

```bash
ncu --set full --target-processes all --export edge_kernel \
    ./dynamic_louvain outfile.txt < input.txt
```

Critical metrics:
- `sm__warps_active.avg.pct_of_peak_sustained_active` — expected **~3%** because of warp-leader-only. A single number that proves the warp-leader guard is the bottleneck.
- `smsp__issue_active.avg.pct_of_peak_sustained_active` — how often threads issue instructions.
- `warp_cycles_per_issued_instruction` — high = heavy stalls.
- Top **stall reasons** (`smsp__average_warps_issue_stalled_*`):
  - `stall_long_scoreboard` → global memory latency (CSR reads).
  - `stall_lg_throttle` → atomic throttling (lock spins).
  - `stall_wait` → barrier waits (`grid.sync`).
- `l1tex__t_sectors_pipe_lsu_mem_global_op_*` — memory transaction count. High = un-coalesced loads.
- `dram__bytes.sum` vs `sm__throughput` → roofline: is the kernel memory-bound or compute-bound?

Predicted result: ~3% SM utilization, dominant stalls on `stall_lg_throttle` (atomic lock spins) and `stall_long_scoreboard` (CSR reads). That pinpoints lock contention + memory bandwidth.

### 1.3 Lightweight in-kernel timing (for quick iteration)

Per-iteration timing without full Nsight overhead:

```cpp
long long t0, t1;
if (tid == 0) t0 = clock64();
// ... phase work ...
grid.sync();
if (tid == 0) {
    t1 = clock64();
    printf("pass %d iter %d: %lld cycles\n", pass, iter, t1 - t0);
}
```

Useful to confirm "iteration 1 of pass 1 takes 90% of pass 1's time" — expected for the singleton-init pass.

### 1.4 What to measure per experiment

For every optimization attempt, record on a fixed graph (e.g. ER(10K, 0.005), seed 42):

| Metric | Tool |
|---|---|
| Total wall time | `time` or in-script timer |
| Louvain kernel time | Nsight Systems per-kernel |
| SM utilization | `ncu` `sm__warps_active` |
| Top stall reason | `ncu` warp stall breakdown |
| Final modularity | The `_communities.txt` + `snap_results_table.py` |
| Community count | Same |

Only then is "this optimization helped" a defensible statement.

---

## Part 2 — Optimization plan

Estimates are rough and stack non-linearly; expect some combinations to interfere. Each row: implementation effort (S/M/L), standalone speedup estimate, and whether it changes algorithmic output.

| # | Fix | Effort | Speedup | Changes output? |
|---|---|---|---|---|
| 1 | Same-community early-exit before locks | S | **3–10×** on late iters | No |
| 2 | Precompute `vertex_weight[v]` once, read O(1) for `k_i` | S | 1.5–2× | No |
| 3 | Process only edges with `src < dest` (eliminate duplicate work) | S | **2×** | Slight — needs dual-direction move in one visit |
| 4 | Raise move threshold from `1e-12` to `1/m` | S | 1.5× (fewer iters) | Minor quality improvement on ER-like graphs |
| 5 | Per-vertex lock only (drop community locks, atomic `community_degree`) | M | **2–4×** | Slight — `community_degree` sees stale reads |
| 6 | Warp-cooperative `calculate_modularity_change` (all 32 threads scan) | M | **5–15×** | No |
| 7 | Increase launch config: `<<<256, 128>>>` (verify with `cudaOccupancyMax…`) | S | 1.5–3× | No |
| 8 | Shared-memory cache of `community_degree[]` per block | M | 1.3–2× | No |
| 9 | Lock-free asynchronous moves (`atomicCAS` on `community[]`, `atomicAdd` on degree) | L | **3–8×** | Yes — non-deterministic, ~2–5% Q variance |
| 10 | Graph coloring + lock-free moves within a color | L | **5–10×** | Slight — different convergence path |
| 11 | Kill the warp-leader guard + use warp-cooperative everywhere | L | **5–15×** | No |
| 12 | Drop per-pass host↔device memcpy for modularity print | S | 1.2–2× | No |
| 13 | SoA memory layout instead of `Edge{src,dest,weight}` | M | 1.3–1.8× memory-bound parts | No |
| 14 | Precompute graph coloring once; reuse across passes | M | Enabler for #10 | No |
| 15 | Fast-path singleton-init pass (degree-1 / no-conflict moves) | M | 2–3× on pass 1 | No |

### Drastic rewrite candidates (pick ONE — do not combine)

- **A. Asynchronous lock-free kernel** (#9 + #6 + #11). What cuGraph does. Threads propose moves via `atomicCAS(&community[u], old, new)` and update `community_degree` with `atomicAdd`. No spin-locks. All 32 threads in a warp participate. Expected: **10–30× speedup**, small quality loss. Benchmark against NetworkX for quality delta.

- **B. Graph-coloring kernel** (#10 + #14 + #6 + #11). Precompute colors once. Inner loop: `for each color c: kernel processes only vertices with color c, using all threads, no locks`. Deterministic, no quality loss, avoids thrashing. Expected: **10–20× speedup**. Adds a coloring pass (Jones-Plassmann parallel coloring, ~2–3% of total time).

- **C. Hybrid CPU–GPU (cuVite style)**. Modularity optimization on GPU, aggregation on CPU. Less drastic algorithmically; each phase uses its strong hand. Expected: **3–5× on medium graphs**, scales to multi-GPU.

### Recommended sequence

**Phase I — hours, correctness-preserving** (expected 5–10× total)
1. #1 — same-community skip (one line).
2. #2 — precompute `vertex_weight`.
3. #3 — only process `src < dest`.
4. #4 — raise threshold.
5. #7 — larger launch config (after verifying cooperative-kernel block limit with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`).

Expected: ER(10K, 0.005) static drops from 34 s to ~3–5 s. Same modularity output.

**Phase II — 1–2 days, warp-cooperative** (expected another 3–10×)
6. #11 + #6 — kill warp-leader guard, rewrite `calculate_modularity_change` and the edge loop to use all 32 threads of a warp; warp-level reductions for `k_i`, `k_i_in_old`, `k_i_in_new`.
7. #12 — remove per-pass modularity prints / memcpys.

Expected: ER(10K) down to ~500 ms; ca-AstroPh static down from 54 s to ~5 s. Modularity unchanged.

**Phase III — 1 week, drastic rewrite** (expected another 2–5×)
Pick A, B, or C.

Recommendation: **B (graph coloring + warp-cooperative)**. Deterministic output matches sequential Louvain quality — no "justify non-determinism" section in the paper — and coloring is a well-studied GPU primitive.

---

## Decisions that change priorities

Before committing to a phase plan, nail down:

1. **Which GPU?** — SM count determines the block-count ceiling for cooperative kernels. RTX 3090 has 82 SMs; T4 has 40 SMs; older cards fewer.
2. **Is determinism a hard requirement for the BTP?** If yes, rule out option A.
3. **Real graphs or synthetic graphs as the critical path?** — different fixes matter. Real graphs benefit more from #3, #5, #6; synthetic from #1, #4.

---

## Minimal profiling harness (run once before touching code)

Drop-in shell wrapper:

```bash
#!/bin/bash
# profile_edge_kernel.sh — run once to capture baseline metrics.
INPUT="../generate/graphs/er_n10000_p0.02_b5_bp0.05.txt"
OUT="profile_baseline"
mkdir -p "$OUT"

# Timeline
nsys profile --stats=true --output="$OUT/nsys" \
    ../algorithm/dynamic_louvain "$OUT/communities.txt" < "$INPUT"

# Per-kernel deep dive (slow; use a smaller graph or limit kernel count)
ncu --set full --target-processes all --export "$OUT/ncu" \
    ../algorithm/dynamic_louvain "$OUT/comms2.txt" < "$INPUT"

echo "Results in $OUT/"
echo "Open nsys in Nsight Systems UI; open ncu with: ncu-ui $OUT/ncu.ncu-rep"
```

Run this **before** any optimization; keep the resulting report as the baseline against which every change is compared.
