# Explanation of CUDA Louvain Community Detection Code

## Project Structure

```
louvain/
├── algorithm/
│   ├── cuda_static_louvain.cu              # GPU static Louvain (standalone)
│   ├── cuda_dynamic_louvain.cu             # Static + 3 dynamic variants (edge-based)
│   ├── cuda_dynamic_louvain_nodebased.cu   # Same as above, node-based kernel
│   ├── nx_louvain.py                       # NetworkX CPU baseline
│   └── compile_dyn.sh
├── real_graphs/                            # SNAP / KONECT benchmark graphs + outputs
├── generate/graphgen.py                    # Random graph + batch generator
├── notes/                                  # Documentation
└── test_dynamic_input.txt
```

---

## 1. The Louvain Algorithm

Louvain is a greedy method for maximizing **modularity** — a scalar in [-0.5, 1] measuring how well a graph is partitioned into communities compared to a random graph.

$$Q = \sum_c \left[\frac{L_c}{m} - \left(\frac{k_c}{m}\right)^2\right]$$

where $L_c$ = sum of internal edge weights in community $c$, $k_c$ = sum of degrees of vertices in $c$, $m$ = total edge weight.

The algorithm alternates two phases until convergence:

1. **Local Moving**: For each vertex, try moving it to each neighbor's community. Accept the move that gives the largest modularity gain $\Delta Q > 0$. Repeat until no vertex moves.
2. **Aggregation**: Collapse each community into a single super-node. Edges between communities become weighted edges between super-nodes (weights summed). Intra-community edges become **self-loops** on the super-node. Reset community assignments to identity.

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
| `d_original_community` | `int[n]` | Maps each original vertex to its current super-node ID across passes |

### Key Functions

**`louvain_kernel`** — The main GPU kernel (cooperative launch, grid-wide sync):
- Only warp leaders (`threadIdx.x % 32 == 0`) participate — avoids intra-warp deadlocks.
- Each warp leader processes a contiguous range of edges (edge-parallel).
- For each edge $(u, v)$: if $u$ and $v$ are in different communities, compute $\Delta Q$ for moving $u$ to $v$'s community. If $\Delta Q > 0$, move $u$.
- Locking order: vertex locks acquired in sorted ID order, then community locks in sorted ID order → **prevents deadlocks**.
- Iterates (with `grid.sync()` barriers) until no vertex moves, capped at 100 iterations.

**`calculate_modularity_change`** — Device function computing:
$$\Delta Q = \frac{2(k_{i \to B} - k_{i \to A})}{m} + \frac{2 k_i (\Sigma_A - \Sigma_B - k_i)}{m^2}$$

**Important:** `k_{i \to A}` and `k_{i \to B}` exclude self-loops. The inner loop has `if (dest == node_to_move) continue;` after `k_i` accumulation. Self-loops are community-invariant — they contribute to modularity regardless of which community the node sits in, so they must not appear in `k_i_in_C`.

**`update_community_degree`** — When vertex $i$ moves from community $A$ to $B$, subtracts $i$'s edge weights from $\Sigma_A$ and adds them to $\Sigma_B$.

**`count_communities`** — Marks which community IDs are in use, counts them using shared memory reduction + atomic add.

**`aggregate_graph`** — Cooperative kernel that:
1. Rewrites every edge's `src`/`dest` from vertex IDs → renumbered community IDs (via prefix-sum mapping).
2. Resets `community[i] = i` and recomputes `community_degree`.

**`update_original_communities`** — Composes the mapping: for each original vertex $v$, traces through the current aggregation's community assignments to get $v$'s final community ID. Called before `aggregate_graph` resets `d_community` to identity.

**`combine_edges`** — Host function using Thrust to merge duplicate edges:
1. Sort edges by `(src, dest)` using `EdgeComparator`.
2. `reduce_by_key` groups edges with same `(src, dest)` pair and sums their weights.
3. Frees old `d_csr_adj`, replaces with the compacted array.

**`calculate_modularity`** — Host-side modularity computation for verification.

### Execution Flow

```
main() → read graph → louvain_cuda():
  Build CSR on host → upload to GPU
  Initialize d_original_community = identity
  Loop until n_nodes stops shrinking:
    1. louvain_kernel (cooperative launch, 32 blocks × 512 threads)
    2. count_communities → get new community count
    3. inclusive_scan → renumber communities
    4. update_original_communities ← BEFORE aggregation
    5. aggregate_graph → collapse to super-nodes
    6. combine_edges → merge duplicate edges (Thrust)
    7. Rebuild CSR offsets (atomic count + exclusive_scan)
    8. Print modularity
  Download d_original_community → write to output file
```

### Output

The static binary takes an optional command-line argument: a path to write the final community assignments. Format:
```
<total community count>
<vertex_id> <community_id>
...
```

---

## 3. cuda_dynamic_louvain.cu — Dynamic Louvain

This file builds on static Louvain and adds three dynamic approaches. It shares the same kernels but introduces:

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

### Improvements Over Static Version
1. `d_csr_adj` passed **by reference** (`Edge*&`) to `run_louvain_phases` → prevents double-free in `combine_edges`.
2. `long long` cast: `((long long)tid * m_edges) / nthreads` → prevents overflow on large graphs.
3. `__threadfence()` after modifying `community_degree` → ensures cross-SM visibility before releasing locks.
4. `CUDA_CHECK` on all `cudaLaunchCooperativeKernel` calls.
5. `MAX_PASSES = 20` and `delta_Q_sum < 1e-6` early-exit added (see known issues — the early-exit causes premature termination on some graphs).

### `run_louvain_phases` — Shared Core Loop

Factored out of the static flow and reused by all four approaches:
```
Loop until n_nodes stops shrinking (or MAX_PASSES):
  1. louvain_kernel (with d_affected on first pass only)
  2. Read delta_Q_sum from device — early-exit if too small
  3. count_communities
  4. inclusive_scan (renumber)
  5. update_original_communities ← BEFORE aggregation
  6. aggregate_graph
  7. combine_edges
  8. rebuild_csr_offsets
```

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

- Currently runs on the host; moving it to GPU would be a real algorithmic contribution.
- Uploads `h_affected` to GPU, passes to `run_louvain_phases`.
- Typically marks **fewer vertices** than Frontier → faster first pass.

### `main()` Pipeline

```
1. Read initial graph → adjacency list
2. Static Louvain → LouvainState → write communities file
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

## 4. cuda_dynamic_louvain_nodebased.cu — Node-Based Variant

Identical to `cuda_dynamic_louvain.cu` except for the `louvain_kernel`. Instead of edge-parallel iteration, each thread processes a contiguous chunk of nodes:

```cpp
for (long long node = chunk_start; node < chunk_end; node++) {
    // For each neighbor of node, compute ΔQ and pick the best target community
    for (int e = ...; e < ...; e++) {
        ...
        if (dQ > best_delta_Q) best_community = target_comm;
    }
    if (best_community != initial_community) {
        // Acquire vertex lock + community locks, re-verify, move
    }
}
```

**Tradeoff:** node-based picks the best target before acquiring locks, then re-verifies under lock. This can miss better moves but avoids the cascading propagation effect of the edge-based version. In practice it produces slightly lower modularity but is more robust (the edge-based version sometimes triggers an illegal memory access during aggregation that the node-based version avoids).

---

## 5. Host Utility Functions

| Function | Purpose |
|----------|---------|
| `build_csr` | Converts `vector<vector<pair<int,double>>>` adjacency list to CSR arrays |
| `apply_batch_updates` | Removes deleted edges, adds inserted edges to adjacency list. Returns new directed edge count |
| `rebuild_csr_offsets` | Builds `d_csr_node_offset` from sorted `d_csr_adj` using atomic counting + exclusive scan |
| `calculate_modularity` | Host-side modularity computation for verification |
| `combine_edges` | Thrust sort + reduce_by_key to merge duplicate edges after aggregation |
| `compute_affected_delta_screening` | Host-side 3-phase affected-vertex computation for delta-screening |
| `write_communities` | Writes the final `LouvainState.community` to a file with the count header |

---

## 6. GPU Execution Model

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

### Edge-Parallel vs Node-Parallel
- **Edge-parallel (default)**: each warp leader gets a contiguous slice `[tid*m/nthreads, (tid+1)*m/nthreads)`. Better load balance for power-law degree distributions.
- **Node-parallel (nodebased file)**: each warp leader processes a contiguous range of nodes. Simpler, more robust, slightly lower quality.

---

## 7. graphgen.py — Graph Generator

Python script using NetworkX + argparse to generate test inputs.

| Function | Purpose |
|----------|---------|
| `create_weighted_random_graph` | Erdos-Renyi graph with random integer edge weights |
| `format_initial_graph` | Formats as `n m` followed by edge lines |
| `generate_batch` | Randomly deletes existing edges and inserts non-edges |
| `generate_dynamic_graph` | Initial graph + $k$ batch updates |
| `generate_static_graph` | Initial graph only (no batches) |

---

## 8. Real-Graph Benchmarking

The `real_graphs/` directory contains SNAP and KONECT benchmark graphs:

| Graph | Nodes | Edges |
|-------|-------|-------|
| ca-GrQc | 5,241 | 14,484 |
| facebook | 4,039 | 88,234 |
| ca-HepTh | 9,875 | 25,973 |
| ca-HepPh | 12,006 | 118,489 |
| ca-AstroPh | 18,771 | 198,050 |
| email-Enron | 36,692 | 183,831 |
| com-amazon | 334,863 | 925,872 |
| com-dblp | 317,080 | 1,049,866 |

### Workflow

```bash
# Convert SNAP files to our input format
python real_graphs/convert_snap_to_dynamic.py snap/<graph>.txt snap/<graph>_converted.txt

# Run all 3 implementations on all graphs
cd real_graphs && bash run_benchmarks.sh

# Print results table (parses output logs + computes true modularity from communities files)
python results_table.py

# CPU baseline for comparison
python ../algorithm/nx_louvain.py
```

---

## 9. Input Format

```
10 18                    ← n_nodes, n_undirected_edges
0 1                      ← edge (u, v) — stored as both u→v and v→u; weight = 1.0
0 2
...
2                        ← n_batches (0 if static-only)
1 2                      ← batch 1: 1 deletion, 2 insertions
3 7 0.1                  ← deletion: remove edge (3,7) with weight 0.1
0 8 0.5                  ← insertion: add edge (0,8) with weight 0.5
1 9 0.5                  ← insertion: add edge (1,9) with weight 0.5
...
```

**Note**: initial edges read as `u v` only (weight hardcoded to 1.0); batch edges read as `u v w`. This inconsistency is documented as a bug.

---

## 10. Output Format

When given an output filename argument, all three executables write community assignments:

```
<n_communities>
<vertex_id> <community_id>
<vertex_id> <community_id>
...
```

The first line is the total number of distinct communities. Subsequent lines map each original vertex to its final community.

---

## 11. Compilation & Execution

```bash
cd algorithm

# Compile
nvcc -rdc=true -arch=sm_60 cuda_static_louvain.cu -o static_louvain
nvcc -rdc=true -arch=sm_60 cuda_dynamic_louvain.cu -o dynamic_louvain
nvcc -rdc=true -arch=sm_60 cuda_dynamic_louvain_nodebased.cu -o dynamic_louvain_nodebased

# Run with output file
./dynamic_louvain communities.txt < ../test_dynamic_input.txt

# Debug mode
nvcc -rdc=true -arch=sm_60 -DDEBUG cuda_dynamic_louvain.cu -o dynamic_louvain_debug

# Generate test input
python ../generate/graphgen.py --nodes 500 --prob 0.02 --batches 3 --deletions 10 --insertions 15 -o ../test_dynamic_input.txt
```

Adjust `-arch=sm_XX` for your GPU: `sm_75` (Turing), `sm_86` (Ampere), `sm_89` (Ada Lovelace).

---

## 12. Summary of Differences Between Approaches

| | Initialization | Vertices Processed (1st pass) | Affected Marking |
|---|---|---|---|
| **Static** | Each vertex in own community | All | N/A |
| **Naive Dynamic** | Previous communities | All | None |
| **Frontier** | Previous communities | Affected only | GPU kernel: all inter-community insertions + intra-community deletions |
| **Delta-Screening** | Previous communities | Affected only | Host: deletions same as Frontier; insertions screened by $\Delta Q > 0$; + community & neighbor propagation |

All subsequent aggregation passes (on smaller graphs) process all super-nodes regardless.

---

## 13. Known Bugs

1. **Self-loop in `k_i_in_old`** *(fixed)*: the inner loop in `calculate_modularity_change` was including self-loop weight in `k_i_in_old`, killing all merges after the first aggregation pass. Fix: `if (dest == node_to_move) continue;` after accumulating `k_i`.
2. **Edge-based kernel illegal memory access** during aggregation on denser graphs (ca-HepPh, ca-AstroPh, email-Enron). Nondeterministic. Node-based kernel does not exhibit this.
3. **Input format inconsistency**: initial graph reads `u v` (weight=1.0), batches read `u v w`.
4. **Delta-screening uses stale weights**: `prev_state.vertex_weight/community_weight/total_weight` reflect the pre-batch graph, but `adj` is post-batch. Affected vertex selection is slightly inaccurate.
5. **Delta-screening runs on the host**, not the GPU. Moving it to a kernel would be a real contribution.
