# Explanation of CUDA Louvain Community Detection Code

## Project Structure

```
louvain/
├── cuda_static_louvain.cu        # GPU static Louvain (standalone)
├── cuda_dynamic_louvain.cu       # GPU static + 3 dynamic Louvain variants
├── test_dynamic_input.txt        # Sample input with batch updates
├── README.md                     # Build & run instructions
├── dynamic_louvain.md            # Algorithm design document
├── explanation.md                # This file
└── generate/
    └── graphgen.py               # Random graph + batch generator
```

---

## 1. The Louvain Algorithm

Louvain is a greedy method for maximizing **modularity** — a scalar in [-0.5, 1] measuring how well a graph is partitioned into communities compared to a random graph.

$$Q = \sum_c \left[\frac{L_c}{m} - \left(\frac{k_c}{m}\right)^2\right]$$

where $L_c$ = sum of internal edge weights in community $c$, $k_c$ = sum of degrees of vertices in $c$, $m$ = total edge weight.

The algorithm alternates two phases until convergence:

1. **Local Moving**: For each vertex, try moving it to each neighbor's community. Accept the move that gives the largest modularity gain $\Delta Q > 0$. Repeat until no vertex moves.
2. **Aggregation**: Collapse each community into a single super-node. Edges between communities become weighted edges between super-nodes (weights summed). Reset community assignments to identity.

Repeat phases 1–2 on the progressively smaller graph until no further reduction occurs.

---

## 2. cuda_static_louvain.cu — Static Louvain

This file implements the basic Louvain algorithm on GPU as a standalone program.

### Data Structures

| Name | Type | Purpose |
|------|------|---------|
| `Edge` | struct `{src, dest, weight}` | Single directed edge |
| `h_csr_adj` / `d_csr_adj` | `Edge[]` | All edges sorted by source (CSR format) |
| `h_csr_node_offset` / `d_csr_node_offset` | `int[n+1]` | Vertex $v$'s edges span indices `[offset[v], offset[v+1])` |
| `community` | `int[n]` | `community[v]` = community ID of vertex $v$ |
| `community_degree` | `double[n]` | Sum of all edge weights of vertices in community $c$ |
| `d_vertex_locks` | `int[n]` | Per-vertex spinlocks (0=free, 1=held) |
| `d_community_locks` | `int[n]` | Per-community spinlocks |
| `d_total_weight` | `double` | Sum of all directed edge weights ($m$) |

### Key Functions

**`louvain_kernel`** — The main GPU kernel (cooperative launch, grid-wide sync):
- Only warp leaders (`threadIdx.x % 32 == 0`) participate — avoids intra-warp deadlocks.
- Each warp leader processes a contiguous range of edges (edge-parallel, not vertex-parallel).
- For each edge $(u, v)$: if $u$ and $v$ are in different communities, compute $\Delta Q$ for moving $u$ to $v$'s community. If $\Delta Q > 0$, move $u$.
- Locking order: vertex locks acquired in sorted ID order, then community locks in sorted ID order → **prevents deadlocks**.
- Iterates (with `grid.sync()` barriers) until no vertex moves, capped at 100 iterations.

**`calculate_modularity_change`** — Device function computing:
$$\Delta Q = \frac{2(k_{i \to B} - k_{i \to A})}{m} + \frac{2 k_i (\Sigma_A - \Sigma_B - k_i)}{m^2}$$

**`update_community_degree`** — When vertex $i$ moves from community $A$ to $B$, subtracts $i$'s edge weights from $\Sigma_A$ and adds them to $\Sigma_B$.

**`count_communities`** — Marks which community IDs are in use, counts them using shared memory reduction + atomic add.

**`aggregate_graph`** — Cooperative kernel that:
1. Rewrites every edge's `src`/`dest` from vertex IDs → renumbered community IDs (via prefix-sum mapping).
2. Resets `community[i] = i` and recomputes `community_degree`.

**`combine_edges`** — Host function using Thrust to merge duplicate edges:
1. Sort edges by `(src, dest)` using `EdgeComparator`.
2. `reduce_by_key` groups edges with same `(src, dest)` pair and sums their weights.
3. Frees old `d_csr_adj`, replaces with the compacted array.

**`calculate_modularity`** — Host-side modularity computation for verification.

### Execution Flow

```
main() → read graph → louvain_cuda():
  Build CSR on host → upload to GPU
  Loop until n_nodes stops shrinking:
    1. louvain_kernel (cooperative launch, 32 blocks × 512 threads)
    2. count_communities → get new community count
    3. inclusive_scan → renumber communities
    4. aggregate_graph → collapse to super-nodes
    5. combine_edges → merge duplicate edges (Thrust)
    6. Rebuild CSR offsets (atomic count + exclusive_scan)
    7. Print modularity
```

### Known Issues in This File
- Integer overflow: `tid * m_edges` can overflow `int` for large graphs (no `long long` cast).
- Missing `__threadfence()` after `community_degree` updates — writes may not be visible across SMs.
- Missing `CUDA_CHECK` on cooperative kernel launch.
- `d_csr_adj` passed by value to functions that free/reallocate it — potential double-free.

---

## 3. cuda_dynamic_louvain.cu — Dynamic Louvain

This file fixes the issues above and adds three dynamic approaches. It shares the same kernels but introduces:

### New Structures

**`LouvainState`** — Returned after each run, stores everything needed to warm-start:
```cpp
struct LouvainState {
    int n_nodes;
    vector<int> community;          // community[v] for each original vertex
    vector<double> vertex_weight;   // degree of v (sum of edge weights)
    vector<double> community_weight;// degree of community c
    double total_weight;            // m
};
```

### Modified Kernel: `louvain_kernel` with `d_affected`

An extra parameter `int* d_affected` enables dynamic Louvain:
- If `nullptr` → all vertices processed (static / naive behavior).
- If non-null → skip vertex $u$ if `d_affected[u] == 0`.
- When a vertex moves, **all its neighbors** are marked affected (frontier propagation).

### Bug Fixes Over Static Version
1. `d_csr_adj` passed **by reference** (`Edge*&`) to `run_louvain_phases` → prevents double-free in `combine_edges`.
2. `long long` cast: `((long long)tid * m_edges) / nthreads` → prevents overflow.
3. `__threadfence()` after modifying `community_degree` → ensures cross-SM visibility before releasing locks.
4. `CUDA_CHECK` on all `cudaLaunchCooperativeKernel` calls.

### `run_louvain_phases` — Shared Core Loop

Factored out of the static flow and reused by all four approaches:
```
Loop until n_nodes stops shrinking:
  1. louvain_kernel (with d_affected on first pass only)
  2. count_communities
  3. inclusive_scan (renumber)
  4. update_original_communities ← BEFORE aggregation
  5. aggregate_graph
  6. combine_edges
  7. rebuild_csr_offsets
```

**`update_original_communities`** — Composes the mapping: for each original vertex $v$, traces through the current aggregation's community assignments to get $v$'s final community ID. Called before `aggregate_graph` resets `d_community` to identity.

### Four Approaches

#### 3a. Static (`louvain_static_cuda`)
- Each vertex starts in its own community.
- `d_affected = nullptr` → all vertices processed.
- Returns `LouvainState` for warm-starting dynamic runs.

#### 3b. Naive Dynamic (`louvain_dynamic_naive_cuda`)
- Starts from previous `LouvainState.community` assignments.
- Recomputes `community_degree` for the new graph with old labels.
- `d_affected = nullptr` → still processes all vertices.
- Converges faster than static because the starting partition is already close to optimal.

#### 3c. Frontier Dynamic (`louvain_dynamic_frontier_cuda`)
- Starts from previous community assignments.
- **`mark_affected_frontier` kernel** marks affected vertices on GPU:
  - Deletions: if `community[u] == community[v]` → mark both (intra-community edge removed).
  - Insertions: if `community[u] != community[v]` → mark both (inter-community edge added).
- Passes `d_affected` to `run_louvain_phases` → first-pass kernel skips unaffected vertices.
- Neighbors of moved vertices become affected (frontier propagation in kernel).

#### 3d. Delta-Screening Dynamic (`louvain_dynamic_delta_screening_cuda`)
- Most selective approach. Affected set computed **on the host** in `compute_affected_delta_screening`:

  **Phase 1 — Deletions** (same as Frontier): mark endpoints of deleted intra-community edges. Flag their community.

  **Phase 2 — Insertions** (the key difference):
  - Sort insertions by source vertex.
  - For each source $u$ with inter-community insertions, accumulate weight per target community.
  - Compute delta modularity using only newly inserted edge weights:
    $$\Delta Q_{\text{screen}} = \frac{w_{u \to c}}{m} - \frac{k_u(k_u + \Sigma_c - \Sigma_d)}{2m^2}$$
  - Only mark $u$ if best $\Delta Q > 0$.

  **Phase 3 — Propagation**: neighbors of marked vertices + all vertices in flagged communities.

- Uploads `h_affected` to GPU, passes to `run_louvain_phases`.
- Typically marks **fewer vertices** than Frontier → faster first pass.

### `main()` Pipeline

```
1. Read initial graph → adjacency list
2. Static Louvain → LouvainState
3. For each batch:
   a. Read deletions + insertions
   b. apply_batch_updates (host-side adjacency list modification)
   c. Naive Dynamic → LouvainState
   d. Frontier Dynamic → LouvainState
   e. Delta-Screening Dynamic → LouvainState
   f. Print all three modularities + execution times
   g. Use delta-screening result as next batch's starting state
```

---

## 4. Host Utility Functions

| Function | Purpose |
|----------|---------|
| `build_csr` | Converts `vector<vector<pair<int,double>>>` adjacency list to CSR arrays |
| `apply_batch_updates` | Removes deleted edges, adds inserted edges to adjacency list. Returns new directed edge count |
| `rebuild_csr_offsets` | Builds `d_csr_node_offset` from sorted `d_csr_adj` using atomic counting + exclusive scan (on GPU via Thrust) |
| `calculate_modularity` | Host-side modularity computation for verification |
| `combine_edges` | Thrust sort + reduce_by_key to merge duplicate edges after aggregation |
| `compute_affected_delta_screening` | Host-side 3-phase affected-vertex computation for delta-screening |

---

## 5. GPU Execution Model

### Cooperative Groups
The Louvain kernel needs **grid-wide synchronization** (all threads wait between iterations). Standard `__syncthreads()` only works within a block. CUDA Cooperative Groups (`cg::grid_group::sync()`) provides this.

Requirements: `-rdc=true` flag, `cudaLaunchCooperativeKernel`, compute capability ≥ 6.0.

### Warp-Leader Execution
Only `threadIdx.x % 32 == 0` participates. This wastes 31/32 threads but avoids complex intra-warp synchronization for the locking protocol.

### Locking Protocol
Two-level, sorted-order locking prevents deadlocks:
```
Lock vertex min(u, v), then max(u, v)
  Lock community min(A, B), then max(A, B)
    Read + write community assignments and degrees
    __threadfence()
  Unlock community max, then min
Unlock vertex max, then min
```

### Edge-Parallel Distribution
Each warp leader gets a contiguous slice of the edge array: `[tid*m/nthreads, (tid+1)*m/nthreads)`. This provides better load balance than vertex-parallel for power-law degree distributions.

---

## 6. graphgen.py — Graph Generator

Python script using NetworkX + argparse to generate test inputs.

| Function | Purpose |
|----------|---------|
| `create_weighted_random_graph` | Erdos-Renyi graph with random integer edge weights |
| `format_initial_graph` | Formats as `n m` followed by edge lines |
| `generate_batch` | Randomly deletes existing edges and inserts non-edges |
| `generate_dynamic_graph` | Initial graph + $k$ batch updates |
| `generate_static_graph` | Initial graph only (no batches) |

**Key CLI flags**: `--nodes`, `--prob`, `--batches`, `--deletions`, `--insertions`, `--static`, `--seed`, `-o`.

---

## 7. Input Format

```
10 18                    ← n_nodes, n_undirected_edges
0 1 1.0                  ← edge (u, v, weight) — stored as both u→v and v→u internally
0 2 1.0
...
2                        ← n_batches
1 2                      ← batch 1: 1 deletion, 2 insertions
3 7 0.1                  ← deletion: remove edge (3,7) with weight 0.1
0 8 0.5                  ← insertion: add edge (0,8) with weight 0.5
1 9 0.5                  ← insertion: add edge (1,9) with weight 0.5
...
```

---

## 8. Compilation & Execution

```bash
# Compile
nvcc -rdc=true -arch=sm_60 cuda_dynamic_louvain.cu -o dynamic_louvain

# Run
./dynamic_louvain < test_dynamic_input.txt

# Debug mode (verbose per-node logging)
nvcc -rdc=true -arch=sm_60 -DDEBUG cuda_dynamic_louvain.cu -o dynamic_louvain_debug

# Generate test input
python generate/graphgen.py --nodes 500 --prob 0.02 --batches 3 --deletions 10 --insertions 15 -o test_dynamic_input.txt
```

Adjust `-arch=sm_XX` for your GPU: `sm_75` (Turing), `sm_86` (Ampere), `sm_89` (Ada Lovelace).

---

## 9. Summary of Differences Between Approaches

| | Initialization | Vertices Processed (1st pass) | Affected Marking |
|---|---|---|---|
| **Static** | Each vertex in own community | All | N/A |
| **Naive Dynamic** | Previous communities | All | None |
| **Frontier** | Previous communities | Affected only | GPU kernel: all inter-community insertions + intra-community deletions |
| **Delta-Screening** | Previous communities | Affected only | Host: deletions same as Frontier; insertions screened by $\Delta Q > 0$; + community & neighbor propagation |

All subsequent aggregation passes (on smaller graphs) process all super-nodes regardless.
