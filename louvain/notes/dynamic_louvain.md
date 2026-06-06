# Dynamic Louvain Algorithm — Implementation & Considerations

## 1. Overview

The file `cuda_dynamic_louvain.cu` implements four GPU-accelerated variants of the Louvain community detection algorithm:

| Variant | Entry Point | Description |
|---------|-------------|-------------|
| **Static** | `louvain_static_cuda()` | Standard Louvain from scratch (all vertices start in their own community) |
| **Naive Dynamic** | `louvain_dynamic_naive_cuda()` | Warm-started from previous communities; processes *all* vertices |
| **Frontier Dynamic** | `louvain_dynamic_frontier_cuda()` | Warm-started; processes only *affected* vertices in the first pass |
| **Delta-Screening Dynamic** | `louvain_dynamic_delta_screening_cuda()` | Warm-started; screens insertions for positive modularity gain before marking affected |

All four share the same core multi-pass loop (`run_louvain_phases`) consisting of alternating **local-moving** and **graph aggregation** phases.

---

## 2. Algorithm Design

### 2.1 Static Louvain (Baseline)

The standard Louvain method proceeds in two alternating steps:

1. **Local Moving** — Each vertex is considered for moving to a neighboring community. A move is accepted if it increases modularity ($\Delta Q > 0$). Iterations continue until no vertex moves.
2. **Aggregation** — Communities are collapsed into super-nodes, producing a smaller graph. Edges between communities become edges between super-nodes, with weights summed.

These two steps repeat until the graph can no longer be reduced (i.e., the number of communities equals the number of super-nodes).

**Modularity gain** for moving vertex $i$ from community $A$ to community $B$:

$$
\Delta Q = \frac{2(k_{i \to B} - k_{i \to A})}{m} + \frac{2 k_i (\Sigma_A - \Sigma_B - k_i)}{m^2}
$$

where:
- $k_{i \to C}$ = sum of edge weights from $i$ to vertices in community $C$
- $k_i$ = total degree (edge weight sum) of vertex $i$
- $\Sigma_C$ = total degree of all vertices in community $C$
- $m$ = total edge weight of the graph

### 2.2 Naive Dynamic Louvain

When the graph changes (edges added/removed), re-running static Louvain from scratch is wasteful if only a small portion of the graph is affected.

**Naive Dynamic** addresses this by:
- **Warm-starting** community assignments from the previous run rather than initializing each vertex in its own community.
- Recomputing `community_degree` (i.e., $\Sigma_C$) based on the *new* graph topology combined with the *old* community labels.
- Running the full local-moving + aggregation loop on all vertices.

This converges faster because the starting partition is already close to optimal — most vertices do not need to move.

### 2.3 Frontier Dynamic Louvain

**Frontier Dynamic** goes further by identifying which vertices are actually *affected* by the batch update and restricting processing to those vertices:

**Affected vertex identification** (`mark_affected_frontier` kernel):
- **Deleted edges**: If both endpoints were in the *same* community, they are marked affected (removing an intra-community edge may weaken the community).
- **Inserted edges**: If endpoints are in *different* communities, they are marked affected (a new inter-community edge may pull vertices to a different community).

**Frontier propagation** (inside `louvain_kernel`):
- When a vertex moves to a new community, all its neighbors are marked as affected, ensuring the change can propagate outward.

**Scope**: The affected-vertex filter is only applied in the **first pass** (on the original graph). Subsequent passes on the aggregated graph process all super-nodes, since the aggregated graph is already much smaller.

### 2.4 Delta-Screening Dynamic Louvain

**Delta-Screening** is the most selective of the three dynamic approaches. While Frontier marks *all* vertices with inter-community insertions as affected, Delta-Screening applies a modularity-based screening test to decide whether each vertex truly needs re-evaluation.

**Affected vertex identification** (`compute_affected_delta_screening` host function):

**Phase 1 — Deletions** (same as Frontier):
- If both endpoints of a deleted edge were in the same community, mark them as affected.
- Additionally mark the community itself as affected.

**Phase 2 — Insertions** (the key difference from Frontier):
- Group all inserted edges by their source vertex $u$.
- For each source vertex $u$ with inter-community insertions, compute the total weight of newly inserted edges to each target community $c$:
  $$w_{u \to c} = \sum_{(u,v,w) \in \text{insertions},\; \text{comm}(v)=c} w$$
- Compute the delta modularity for moving $u$ from its current community $d$ to the best target community $c$, using **only the newly inserted edge weights**:
  $$\Delta Q_{\text{screen}} = \frac{w_{u \to c}}{m} - \frac{k_u (k_u + \Sigma_c - \Sigma_d)}{2m^2}$$
- Only mark $u$ as affected if $\Delta Q_{\text{screen}} > 0$ for at least one target community.

**Phase 3 — Propagation:**
- All neighbors of directly-marked vertices are marked affected.
- All vertices belonging to a marked community are marked affected.

**Why this is more selective**: Consider a vertex $u$ that receives a new edge to a vertex in another community. Frontier would always mark $u$ as affected. Delta-Screening first checks: given $u$'s degree and the community weights, does this new edge actually provide enough "pull" to justify re-evaluation? If the new edge weight is small relative to $u$'s total degree and the target community is already large, the screening test will be negative and $u$ will be skipped.

**Trade-off**: Delta-Screening requires more computation upfront (host-side per-vertex modularity calculation) but reduces the number of vertices processed by the GPU kernel, leading to faster convergence for small batches.

---

## 3. Data Structures

### 3.1 Graph Representation — CSR

The graph is stored in **Compressed Sparse Row** format:

| Array | Type | Description |
|-------|------|-------------|
| `d_csr_adj` | `Edge[]` | Sorted edge list; each entry has `src`, `dest`, `weight` |
| `d_csr_node_offset` | `int[]` | Offset into `d_csr_adj` for each vertex's adjacency list; size $n+1$ |

Edges are stored as directed (each undirected edge produces two directed entries). During aggregation, duplicate edges are combined using Thrust's `reduce_by_key`.

### 3.2 LouvainState

A persistent state object returned after each run, enabling warm-starting:

```cpp
struct LouvainState {
    int n_nodes;                       // original vertex count
    vector<int> community;             // community[v] for each original vertex
    vector<double> vertex_weight;      // sum of edge weights incident to v
    vector<double> community_weight;   // sum of vertex_weight for all vertices in each community
    double total_weight;               // sum of all directed edge weights
};
```

### 3.3 Community Tracking Through Aggregation

A device array `d_original_community[v]` tracks the mapping from original vertices to final communities across multiple aggregation phases. Before each aggregation step resets `d_community` to the identity, `update_original_communities` composes the current mapping:

```
d_original_community[v] = renumber(d_community[d_original_community[v]])
```

This avoids losing the original-vertex-to-community mapping when the graph is collapsed.

---

## 4. GPU Implementation Details

### 4.1 Cooperative Groups for Grid-Wide Synchronization

The Louvain kernel requires all threads to synchronize between iterations (to check whether any vertex moved). Standard `__syncthreads()` only synchronizes within a block. We use **CUDA Cooperative Groups** (`cg::grid_group`) for grid-wide barriers:

```cpp
cg::grid_group grid = cg::this_grid();
grid.sync();  // all threads across all blocks synchronize here
```

This requires:
- Compile flag: `-rdc=true`
- Launch via: `cudaLaunchCooperativeKernel()`
- Compute capability $\geq$ 6.0

### 4.2 Warp-Level Thread Assignment

Only one thread per warp (the warp leader: `threadIdx.x % 32 == 0`) participates in the Louvain computation. This simplifies locking and avoids intra-warp divergence issues, though it trades off occupancy.

### 4.3 Lock-Based Concurrency

Since multiple threads may try to move vertices simultaneously, two levels of locking are used:

| Lock | Purpose |
|------|---------|
| `d_vertex_locks[v]` | Protects reads of vertex `v`'s community assignment |
| `d_community_locks[c]` | Protects updates to `community_degree[c]` |

**Deadlock prevention**: Locks are always acquired in **sorted order** (lower index first):

```cpp
int first = min(node_a, node_b);
int second = max(node_a, node_b);
while (atomicCAS(&lock[first], 0, 1) != 0) {}
while (atomicCAS(&lock[second], 0, 1) != 0) {}
// ... critical section ...
atomicExch(&lock[second], 0);
atomicExch(&lock[first], 0);
```

A `__threadfence()` is issued after modifying `community_degree` to ensure writes are visible across SMs before releasing locks.

### 4.4 Edge-Parallel Work Distribution

Work is distributed across threads by assigning each thread a contiguous range of edges (not vertices). This provides better load balancing for graphs with skewed degree distributions:

```cpp
long long start = ((long long)tid * m_edges) / nthreads;
long long end   = ((long long)(tid + 1) * m_edges) / nthreads;
```

The `long long` cast prevents integer overflow when `tid * m_edges` exceeds $2^{31}$.

### 4.5 Graph Aggregation on GPU

After local moving converges, the graph is aggregated:

1. **Renumber communities** — `count_communities` + inclusive prefix scan assigns contiguous IDs.
2. **Remap edges** — `aggregate_graph` kernel rewrites `src`/`dest` in `d_csr_adj` to super-node IDs and resets `d_community` to identity.
3. **Combine duplicate edges** — `combine_edges()` uses Thrust `sort` + `reduce_by_key` to merge edges between the same pair of super-nodes, summing their weights.
4. **Rebuild CSR offsets** — `rebuild_csr_offsets()` uses atomic counting + exclusive scan.

### 4.6 Affected Vertex Marking (Frontier)

The `mark_affected_frontier` kernel runs once before the first Louvain pass:

```
Deletions:  if community[u] == community[v]  →  affected[u] = affected[v] = 1
Insertions: if community[u] != community[v]  →  affected[u] = affected[v] = 1
```

During local moving, when a vertex moves:
```
for each neighbor j of moved_vertex:
    affected[j] = 1
```

This propagates the frontier outward from the initial affected set, allowing cascading community restructuring.

### 4.7 Affected Vertex Marking (Delta-Screening)

Unlike the Frontier approach which runs entirely on the GPU, Delta-Screening computes the affected set on the **host** via `compute_affected_delta_screening()`, then uploads the result to device memory. This is because the screening computation requires per-vertex iteration over grouped insertions with per-community weight accumulation — a pattern that maps more naturally to sequential/CPU processing given the typically small batch sizes.

The host function performs three phases:

1. **Deletion scan** — Mark vertices at endpoints of deleted intra-community edges; flag their community.
2. **Insertion screening** — Sort insertions by source vertex. For each source $u$, accumulate inserted edge weights per target community using a `std::map`. Evaluate $\Delta Q$ for each candidate community. Only mark $u$ if the best $\Delta Q > 0$.
3. **Propagation** — Expand the affected set to neighbors and community members.

The precomputed `h_affected` array is uploaded to `d_affected` via `cudaMemcpy` before launching the Louvain kernel.

---

## 5. Batch Update Pipeline

The `main()` function orchestrates the full pipeline:

```
1. Read initial graph → build adjacency list
2. Run static Louvain → get LouvainState
3. For each batch:
   a. Read deletions and insertions
   b. Apply updates to host adjacency list (apply_batch_updates)
   c. Run naive dynamic Louvain → LouvainState
   d. Run frontier dynamic Louvain → LouvainState
   e. Run delta-screening dynamic Louvain → LouvainState
   f. Print modularity + timing comparison of all four approaches
   g. Use delta-screening result as starting state for next batch
```

Edge updates are applied on the host via `apply_batch_updates()`, which modifies the adjacency list in-place. The CSR is rebuilt from scratch for each run since the topology change may affect all offsets.

---

## 6. Design Considerations & Trade-offs

### 6.1 Correctness vs. Performance

| Decision | Trade-off |
|----------|-----------|
| **AtomicCAS locking** | Guarantees correctness for concurrent moves but introduces serialization. Spin-waiting can waste cycles. |
| **Warp-leader-only execution** | Simplifies locking (no intra-warp conflicts) but uses only 1/32 of available threads. |
| **Grid-wide sync via cooperative groups** | Enables iterative convergence in a single kernel launch but limits the number of blocks to hardware occupancy. |
| **Edge-parallel distribution** | Better load balance than vertex-parallel for power-law graphs, but a vertex's edges may span multiple threads, requiring locks. |

### 6.2 Memory Management

- All device memory is allocated per-run and freed afterward. This avoids persistent memory leaks but introduces allocation overhead per batch.
- The `d_csr_adj` pointer is passed **by reference** (`struct Edge*& d_csr_adj`) to `run_louvain_phases` because `combine_edges()` frees and reallocates the array. Passing by value would cause a double-free.
- Host arrays (`h_csr_adj`, etc.) are allocated with `new[]` and freed with `delete[]`. Using RAII wrappers or `std::vector` would be safer.

### 6.3 Affected-Vertex Scope

Both the Frontier and Delta-Screening approaches only apply the affected filter in the **first pass**. Rationale:
- After aggregation, the graph is much smaller, so processing all super-nodes is cheap.
- The community structure of the aggregated graph may differ substantially from the original, making the original affected set meaningless.

### 6.4 Delta-Screening: Host vs. Device

The delta-screening computation runs on the host rather than the GPU. This is a deliberate trade-off:
- **Batch sizes are typically small** (tens to hundreds of edges), so the overhead of GPU kernel launch and synchronization would outweigh the parallelism benefit.
- **Per-vertex grouping** of insertions with per-community accumulation is naturally sequential and would require complex atomic operations on the GPU.
- **The LouvainState** (community weights, vertex weights) is already stored on the host, avoiding extra device-to-host transfers.

For very large batches (millions of edges), a GPU-based delta-screening implementation could be beneficial.

### 6.4 Modularity Computation

Modularity is computed **on the host** for verification purposes. It is not used to drive algorithmic decisions on the GPU. The GPU kernel uses the modularity *change* formula ($\Delta Q$) to decide vertex moves.

### 6.5 Convergence Control

- Local moving terminates when no vertex moves in an iteration (tracked via `volatile int changed`).
- A hard cap of **100 iterations** per pass prevents infinite loops in pathological cases.
- The outer loop terminates when aggregation produces no reduction in vertex count.

### 6.6 Numerical Precision

- All weights and modularity values use `double` precision.
- The threshold for accepting a move is $\Delta Q > 10^{-12}$, avoiding moves driven purely by floating-point noise.
- `atomicAdd` for `double` requires compute capability $\geq$ 6.0.

---

## 7. Known Limitations

1. **Fixed grid dimensions**: The cooperative kernel is launched with 32 blocks × 512 threads. For large graphs, occupancy-based calculation should be used to maximize parallelism.
2. **No self-loop handling**: Self-loops in the aggregated graph are not explicitly removed, which can affect the modularity computation.
3. **Host-side CSR rebuild**: The CSR is fully reconstructed from the adjacency list for each dynamic run. Incremental CSR updates could reduce overhead for small batches.
4. **No vertex addition/removal**: The dynamic algorithm assumes the vertex set is fixed; only edge insertions and deletions are supported.
5. **Single GPU**: No multi-GPU or distributed support.
6. **Determinism**: Due to concurrent atomic operations, results may differ across runs. Deterministic ordering of vertex processing would require sorting or sequential execution.

---

## 8. Relationship to Reference Implementation

This CUDA implementation draws algorithmic inspiration from the OpenMP-based dynamic Louvain community detection framework (see `louvain-communities-openmp-dynamic/`). Key differences:

| Aspect | OpenMP Reference | CUDA Implementation |
|--------|------------------|---------------------|
| Parallelism model | Thread-parallel (OpenMP pragmas) | Warp-parallel (CUDA cooperative groups) |
| Synchronization | OpenMP barriers | `cg::grid_group::sync()` + atomicCAS locks |
| Dynamic approaches | Naive, Delta-Screening, Frontier | Naive, Frontier, Delta-Screening |
| Delta-Screening execution | Full OpenMP parallel (with per-thread buffers) | Host-side sequential (batch sizes typically small) |
| Graph storage | CSR with separate weight arrays | CSR with `Edge` structs (src, dest, weight) |
| Aggregation | CPU-side | GPU-side (Thrust sort + reduce_by_key) |

---

## 9. File Dependencies

```
cuda_dynamic_louvain.cu
├── CUDA Runtime (cuda_runtime.h)
├── Thrust (thrust/sort.h, thrust/reduce_by_key, thrust/scan, ...)
├── Cooperative Groups (cooperative_groups.h)
└── C++ STL (vector, map, tuple, chrono, algorithm)
```

Compile command:
```bash
nvcc -rdc=true -arch=sm_60 cuda_dynamic_louvain.cu -o dynamic_louvain
```

Minimum requirements:
- CUDA Toolkit with `nvcc`
- GPU with compute capability $\geq$ 6.0
- `-rdc=true` for cooperative kernel support
