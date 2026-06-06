# Background — Section Plan

Planning document for the Background chapter of the BTP report. Lists every subsection to discuss with a one-paragraph brief. Much of this can be ported from UGRC-I §2 and `notes/explanation.md`.

Parallel-programming-level design decisions (locking, coloring, edge- vs vertex-parallel) are **not** in Background — they belong in `work_theory.md`. Background is foundational concepts only.

---

## 1. Graph-Theoretic Preliminaries

### 1.1 Graphs, vertices, edges, weights
Define $G = (V, E)$, undirected vs directed, weighted vs unweighted, adjacency representation. Brief (~1 paragraph) — not novel, just notation.

### 1.2 Real-world graph properties
Power-law degree distribution, sparsity ($|E| \ll |V|^2$), clustering coefficient. Motivates CSR representation choice and why load-balancing matters on GPU.

### 1.3 Dynamic / temporal graphs
Define edge insertions and deletions as the basic update primitive. Motivates why a dynamic algorithm is needed: real networks evolve (social graphs, citation graphs, biological interaction networks).

---

## 2. Community Detection

### 2.1 Definition of a community
Intuitive: densely-connected internally, sparsely-connected externally. Mention there is no single universal definition; different algorithms optimize different objectives.

### 2.2 Modularity
The fitness metric. Give the formula (already in UGRC-I §2.1.1, reuse):
$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
Range $[-0.5, 1]$; higher is better. Per-community decomposition $Q_c = \Sigma_{in}(c)/2m - (\Sigma_{deg}(c)/2m)^2$. Cite **Brandes et al. 2007** for modularity analysis.

### 2.3 Modularity gain ($\Delta Q$)
The key incremental quantity — moving vertex $i$ from community $a$ to $b$:
$$\Delta Q_{i:a\to b} = \frac{1}{m}\left[(k_{i,b} - k_{i,a}) - \frac{k_i}{2m}(k_i + \Sigma_b - \Sigma_a)\right]$$
Already in UGRC-I §2.1.3. **Note for the writer**: include a subsection on why self-loops must be excluded from $k_{i,a}$ / $k_{i,b}$ — this is the bug the BTP fixes. Good place to plant the flag.

### 2.4 Limitations of modularity
Resolution limit (Fortunato & Barthélemy 2007). Motivates Leiden and alternatives mentioned in Related Work.

---

## 3. The Louvain Algorithm

### 3.1 Two-phase structure
Phase 1: local moving. Phase 2: aggregation. Repeat until no further reduction. Include the UGRC-I pseudocode (already written, port it).

### 3.2 Convergence and local optima
Louvain is greedy — converges to a local modularity optimum, not global. Mention order-sensitivity (Blondel's empirical observation that order affects time but not quality on their test cases; later work shows order *does* affect quality).

### 3.3 Louvain vs alternatives (brief)
One paragraph + small table: Louvain / Leiden / Label Propagation / Infomap / SLM. Reference to Related Work chapter for details.

---

## 4. Dynamic Louvain — Problem Statement

### 4.1 Why not just re-run static?
Naive re-running wastes work that was already done. Cost grows with graph size, not with batch size. Real-time analysis becomes impossible on large evolving graphs.

### 4.2 Three categories of dynamic approaches
- **Naive dynamic (warm-start)** — keep prior partition, run Louvain from there on *all* vertices.
- **Frontier-based** — only process vertices "affected" by structural changes, propagate affected status during moves.
- **Delta-screening** — additionally filter affected vertices by modularity gain from newly inserted edges.

One paragraph per approach. Cite Aynaud & Guillaume 2010 (naive), Sahu 2024 (DF / frontier), Zarayeneh & Kalyanaraman 2019/2021 (delta-screening).

### 4.3 Batch updates
Define the batch: a set of insertions and deletions applied atomically. Justify the batch model — most real systems buffer updates rather than processing them one-by-one. Introduces the `n_del, n_ins` per-batch parameters used in the BTP evaluation.

---

## 5. GPU Architecture (NVIDIA)

### 5.1 Streaming multiprocessor model
SMs, CUDA cores per SM, warps (32 threads), blocks, grid. Reuse Figure 1 from UGRC-I. One paragraph per level.

### 5.2 Memory hierarchy
Global memory (high capacity, high latency), shared memory (per-SM, low latency), registers (per-thread). Motivates CSR layout and why repeated access patterns matter.

### 5.3 Throughput-oriented design
Contrast with CPU latency-oriented design. Why GPUs dominate on dense, data-parallel workloads but struggle with irregular memory access — the central tension for graph algorithms.

### 5.4 Test hardware
Specify the GPU used for benchmarks (UGRC-I used T4 on Colab — state whatever the BTP uses, with SM count, compute capability, memory size). Mandatory for reproducibility.

---

## 6. CUDA Programming Model

### 6.1 Kernels and launch configuration
Define `__global__` kernel, `<<<blocks, threads>>>` launch syntax. Brief example.

### 6.2 Thread hierarchy and indexing
`threadIdx`, `blockIdx`, `blockDim`, `gridDim`. How threads map to data indices.

### 6.3 Synchronization primitives
- `__syncthreads()` — intra-block.
- Atomic operations (`atomicAdd`, `atomicCAS`, `atomicExch`) — lock-free primitives used everywhere in the Louvain kernel.
- `__threadfence()` — cross-SM visibility. Explain why it appears in the move-commit path.

### 6.4 Cooperative groups (grid-wide sync)
Why `__syncthreads()` isn't enough (only intra-block). `cg::grid_group` + `cudaLaunchCooperativeKernel` enable grid-wide `grid.sync()`. Requires `-rdc=true` and compute capability ≥ 6.0. Disadvantage: all blocks must fit on the GPU concurrently.

### 6.5 Thrust library
Productivity library on top of CUDA — `thrust::sort`, `thrust::reduce_by_key`, `thrust::inclusive_scan`, `thrust::for_each`. Used heavily in the BTP for aggregation. Cite Hoberock & Bell.

---

## 7. Graph Representations on GPU

### 7.1 Adjacency matrix
$O(n^2)$ memory — infeasible for large sparse graphs. Useful only for small dense examples.

### 7.2 Compressed Sparse Row (CSR)
The standard representation. Two arrays: edge list (sorted by source), offset array. $O(n+m)$ memory. Supports $O(1)$ degree lookup via offset differences. Brief discussion of why CSR is the input format for cuGraph and most GPU graph libraries.

### 7.3 Edge-struct representation (BTP choice)
The BTP uses an `Edge{src, dest, weight}` struct array rather than parallel arrays. Discuss memory layout tradeoff (struct-of-arrays vs array-of-structs, coalesced access).

### 7.4 Dynamic CSR — challenges
CSR was designed for static graphs. Insertions and deletions require either rebuilding or maintaining a secondary structure. This is a real design decision in the BTP — rebuild-from-adjacency-list-each-batch is what's currently implemented. Details of the alternatives and the choice go in `work_theory.md`.

---

## 8. Summary

One paragraph closing the chapter — "With these preliminaries we can now discuss prior work (Chapter 3), our design choices (Chapter 4), and the implementation itself (Chapter 5)." Soft handoff to Related Work and Work Theory.

---

## Notes for the writer

- Reuse every figure from UGRC-I you can. GPU hierarchy, algorithm pseudocode, CSR diagram.
- Keep each subsection short (½–1 page). Background should be ~6–8 pages of a 15–20 page report, not more.
- If a reviewer already knows graph theory / GPU basics, they will skim this — don't pad. But it must be complete enough for a non-GPU reader.
- Cite at the first mention of any concept, not at the end of a paragraph.
- **Do not put parallel-programming design decisions here** (locking, coloring, edge-vs-vertex). Those belong in `work_theory.md`.
