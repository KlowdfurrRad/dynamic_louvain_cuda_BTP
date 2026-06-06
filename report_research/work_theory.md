# Work Theory — Section Plan

Planning document for the Work Theory chapter. This is the chapter that explains **design decisions** between background and implementation — "here's what we chose, and why". Modeled after UGRC-I §5.

Foundational concepts (CSR format, CUDA primitives, modularity definition) already live in `background.md`. This chapter discusses the design trade-offs *on top of* those primitives.

---

## 1. Graph Representation Choices

### 1.1 Edge-struct array vs structure-of-arrays
The BTP uses `Edge{src, dest, weight}` struct array (array-of-structs). Alternative is three parallel arrays (structure-of-arrays). Discuss coalesced-access tradeoff, cache line utilization, and why AoS was chosen.

### 1.2 Unweighted input
Current code hardcodes weight = 1.0 for the initial graph while batch edges are read as `u v w`. Document this as an implementation choice (+ known bug).

### 1.3 Handling dynamic updates on a CSR
Three strategies:
- **Rebuild from scratch** after every batch (current approach; simple, high overhead).
- **Lazy deletion with tombstones** — mark edges dead, periodic compaction.
- **Hybrid CSR + overlay** — keep dynamic updates in a separate delta structure; merge periodically.
Discuss the tradeoff and justify the current choice. If a future change moves to a different representation, Work Theory is where that gets re-explained.

---

## 2. Community Tracking Across Aggregation

### 2.1 The problem
After aggregation, community IDs are re-indexed. To report "vertex 42 is in community 7", we need to trace 42 through every aggregation level. Naive approach: store full history of community assignments per iteration.

### 2.2 Community-centric storage
`vector<vector<int>> community_nodes` — each community holds the list of its member vertices. Allows fast "who is in community C?" but merging multiple lists during aggregation is expensive and hard to parallelize.

### 2.3 Vertex-centric storage (BTP choice)
`int[] d_original_community` — each original vertex maps to its current super-node. On every aggregation, update via composition: `original[v] = new_id_of(old_id_of(original[v]))`. This is what `update_original_communities` kernel does. $O(n)$ per pass, fully parallel.

### 2.4 Why vertex-centric wins on GPU
Parallelism: trivial to update all vertices in parallel with no synchronization. No variable-length data structures. Cache-friendly linear scan. Cite UGRC-I §5.1.2 where this was already established.

---

## 3. Stopping Criteria

### 3.1 Move-based (no modularity gain)
Phase 1 inner loop exits when no vertex finds a beneficial move. The "safe" criterion; guarantees local optimum.

### 3.2 Modularity threshold
Exit when the modularity gain across an entire pass is below $\epsilon$. Used by NetworkX. The BTP uses this for the outer loop (`delta_Q_sum < 1e-6`).

### 3.3 Loop count caps (thrashing prevention)
Max iterations on the inner loop. Prevents pathological "A moves to B's community, B moves to A's" oscillations in asynchronous parallel execution. The BTP uses `louvain_kernel_iteration < 100` and `MAX_PASSES = 20`.

### 3.4 Which criteria are used where
Summarize which criterion applies at which level in the BTP:
- Inner phase-1 loop: move-based + iteration cap.
- Outer multi-pass loop: $n_{nodes}$ unchanged OR modularity threshold OR MAX_PASSES.

---

## 4. Modularity Gain Calculation

### 4.1 Naive full recomputation
Recompute $Q$ before and after each candidate move. $O(E + C)$ per move. Infeasible.

### 4.2 Direct incremental gain calculation
The formula from Background §2.3 computes $\Delta Q$ directly in $O(\deg(v))$. Pre-compute and maintain `community_degree[]` and update it incrementally when moves happen. This is what `calculate_modularity_change` + `update_community_degree` implement.

### 4.3 Self-loop correction
**Important subsection — the headline bug-fix of the BTP.** When working on the aggregated graph, super-nodes have self-loops representing intra-community edges of the original partition. These must be **excluded** from $k_{i,a}$ and $k_{i,b}$ because self-loops are community-invariant under the modularity definition. Including them (as the unfixed BTP code did) inflates the perceived cost of leaving the current community and kills all merges in pass 2+. The fix is a single `continue` after accumulating $k_i$.

Present this as a narrative: observation (pass 2+ stalls), diagnosis (delta_Q ≈ 0 on aggregated graph), root cause (self-loop in $k_{i,a}$), fix, empirical confirmation (table showing before/after modularity and community counts).

---

## 5. Parallel Programming Considerations

### 5.1 Heuristics for parallel-safe moves

#### 5.1.1 Graph coloring
Color vertices such that no two adjacent vertices share a color. Process one color at a time; within a color, all moves are independent. Cost: upfront coloring overhead. Used in Grappolo, Naim et al.

#### 5.1.2 Asynchronous updates (lock-free)
Accept that threads may read inconsistent community state. Rely on eventual convergence. Pros: near-linear speedup, simplicity, no locks. Cons: thrashing (two nodes swapping indefinitely), higher variance in final quality, non-determinism.

#### 5.1.3 Locking (BTP choice)
Take per-vertex locks and per-community locks before any move. Two-level locking.

### 5.2 "Slight Change" — move on any positive gain
Standard Louvain: move a vertex to the community giving the **best** $\Delta Q$. BTP's edge-based variant: move if **any** $\Delta Q > \epsilon$ (simpler, more parallelism-friendly, but non-optimal).

Discuss tradeoff: more work in critical sections vs simpler per-edge processing. Mention the BTP's node-based variant as the "classical best-neighbor" alternative.

### 5.3 Synchronization within the Louvain kernel

#### 5.3.1 Why cooperative kernels
Phase-1 inner loop needs grid-wide synchronization between iterations. Standard kernel boundaries require relaunch (high overhead). Cooperative kernels + `grid.sync()` enable single-launch multi-iteration kernels.

#### 5.3.2 The BTP locking protocol
Reuse the algorithm box from UGRC-I §5.2.3.2:
```
lock min(u, v), lock max(u, v)       // vertex locks, sorted to prevent deadlock
  if community[u] == community[v] continue  // stale check
  lock min(C_u, C_v), lock max(C_u, C_v)    // community locks
    compute delta_Q
    if delta_Q > epsilon:
      community[u] = target
      update community_degree atomically
      __threadfence()                      // cross-SM visibility
  unlock community locks
unlock vertex locks
```
Key correctness argument: lock acquisition in sorted ID order prevents circular wait → no deadlock.

#### 5.3.3 Why this upholds correctness
Port the UGRC-I correctness argument (§5.2.3.2). Nodes A (community 1→2) and B (community 3→4) can safely run in parallel; the only intersection is when A is a neighbor of B, and in that case neither possible read value of `community[A]` changes the gain calculation outcome.

#### 5.3.4 Role of `__threadfence()`
Ensures writes to `community_degree` are visible across SMs before locks are released. Without it, a downstream reader on a different SM may see stale data.

### 5.4 Edge-parallel vs vertex-parallel

#### 5.4.1 Edge-parallel kernel (BTP default)
Each warp leader processes a contiguous slice of the edge array. For each edge $(u, v)$: consider moving $u$ into $\text{community}(v)$. Multiple edges per vertex mean the same vertex may be considered for moves multiple times per pass — cascading propagation.

#### 5.4.2 Node-parallel kernel (BTP variant)
Each warp leader processes a contiguous slice of vertices. For each vertex: scan its neighborhood, pick the best community, attempt the move. One move per vertex per pass.

#### 5.4.3 Tradeoff
- **Quality**: edge-parallel generally wins because of cascading propagation (multiple moves per pass per vertex).
- **Robustness**: node-parallel has not exhibited the illegal-memory-access crash the edge-parallel kernel sometimes shows.
- **Speed**: node-parallel is typically 1.5–3× faster per pass but may need more passes.

This is a genuine finding of the BTP — discuss quantitatively in Results.

---

## 6. Graph Aggregation on GPU

### 6.1 Community indexing (prefix-sum renumbering)
After Phase 1, some community IDs are "gone" (every vertex that was in community 17 has moved out). Use `count_communities` + `inclusive_scan` to compute a renumbering map from old community IDs to contiguous new IDs `[0, C)`.

Algorithm box (port from UGRC-I §5.2.4.1).

### 6.2 Modifying data structures
Rewrite every edge's `src` and `dest` via the renumbering map. Reset `d_community` to identity (each new super-node is its own community). Recompute `d_community_degree` via atomic adds. Algorithm box from UGRC-I §5.2.4.2.

### 6.3 Edge aggregation with Thrust
Sort edges by `(src, dest)`, use `reduce_by_key` to collapse duplicates summing weights, rebuild the offsets array with `thrust::for_each` + `exclusive_scan`. Cite Hoberock & Bell.

### 6.4 Self-loops appear here
Intra-community edges become self-loops on super-nodes after remapping. This is the input state to the next pass's Phase 1 and is the reason §4.3's self-loop correction matters.

### 6.5 Memory management during aggregation
`combine_edges` frees the old `d_csr_adj` and replaces it with a smaller compacted array. Must pass the pointer by reference from the caller to avoid dangling references. The BTP does this; UGRC-I's static code had a potential double-free documented as a known issue.

---

## 7. Dynamic Louvain Algorithm — Design

The dynamic Louvain algorithm is the BTP's core contribution. This is the longest and most important section of Work Theory.

### 7.1 The `LouvainState` abstraction

#### 7.1.1 What must be carried between batches
Static Louvain produces a partition and discards all intermediate state. For dynamic use we need to preserve enough to warm-start the next run:
- `community[v]` for every original vertex $v$.
- `vertex_weight[v]` = degree of $v$ in the original graph.
- `community_weight[c]` = sum of degrees of all vertices in community $c$.
- `total_weight` = $2m$.

Justification per field: `community[]` is the warm-start partition; `vertex_weight` and `community_weight` are needed by the host-side delta-screening heuristic; `total_weight` appears in every modularity gain formula.

#### 7.1.2 Chaining batches
Each call returns a new `LouvainState`; the caller feeds it back on the next batch. The BTP's `main()` uses the delta-screening result as the next batch's starting state (other choices — naive result, frontier result — are equally valid; this is a design decision worth justifying).

#### 7.1.3 State consistency invariants
The invariants every `LouvainState` must satisfy:
- `community[v] < n_nodes` for all $v$.
- $\sum_v \text{vertex\_weight}[v] = \text{total\_weight}$.
- $\sum_c \text{community\_weight}[c] = \text{total\_weight}$.
- `community_weight[c] = Σ vertex_weight[v] for v in community c`.

If any of these break, subsequent runs silently produce wrong modularity. Worth asserting in debug builds.

### 7.2 Applying batch updates

#### 7.2.1 Host-side adjacency list modification
- Deletions: linear scan through `adj[u]` to remove $v$, and vice versa. $O(\deg(u) + \deg(v))$ per edge.
- Insertions: append $(v, w)$ to `adj[u]` and $(u, w)$ to `adj[v]`. $O(1)$ per edge.
- Cost model: $O(|\Delta E| \cdot \bar{d})$ for a batch of size $|\Delta E|$ with average degree $\bar{d}$. On high-degree hubs this can dominate; hash-based adjacency or a secondary edge-set structure would help.

#### 7.2.2 CSR rebuild from adjacency list
Current approach: after all batch updates are applied on the host, the entire CSR is rebuilt from scratch and re-uploaded to device. $O(|V| + |E|)$ per batch regardless of batch size. Simple but does not scale when $|\Delta E| \ll |E|$.

#### 7.2.3 Alternative: incremental CSR updates on device
For small batches this would save an order of magnitude. Sketch of the design: use a delta-CSR structure containing only new / removed edges; the Louvain kernel reads main-CSR ∪ delta-CSR with deletion tombstones. Periodically compact. Leave as future work; explain clearly why it was not implemented (complexity budget).

#### 7.2.4 Recomputing `community_degree` after batch
A batch changes edge weights: community degrees on the new graph with old labels are not the same as they were before. The BTP re-initializes `community_degree[]` with a host-side scan over the new CSR grouping by old `community[]`. This is $O(|V| + |E|)$ and cannot be avoided unless the batch is applied incrementally to `community_degree` (possible but fiddly with deletions).

### 7.3 Naive dynamic variant

#### 7.3.1 Algorithm
Warm-start from `prev_state.community`, process *all* vertices in Phase 1 every pass. Functionally identical to static Louvain except for the initialization.

#### 7.3.2 Why it's still faster than static
The starting partition is already close to an optimum for the updated graph (for small batches, the optimum barely moves). Fewer passes are needed, and each pass finds fewer moves.

#### 7.3.3 Role as a baseline
The simplest dynamic variant. Any selective-processing variant (frontier, delta-screening) must beat naive dynamic to justify its additional complexity. Mandatory comparison point.

### 7.4 Frontier dynamic variant

#### 7.4.1 Initial affected set — marking rules
- **Deletion of $(u, v)$**: if $\text{community}[u] = \text{community}[v]$, mark both $u$ and $v$. Rationale: removing an intra-community edge weakens the community; its members may want to leave.
- **Insertion of $(u, v)$**: if $\text{community}[u] \neq \text{community}[v]$, mark both $u$ and $v$. Rationale: adding an inter-community edge creates pull between the two communities.
- **Intra-community insertions** and **inter-community deletions** are ignored — they only reinforce existing community structure and rarely change partitions.

#### 7.4.2 GPU implementation — `mark_affected_frontier` kernel
Simple: one thread per deletion and one per insertion, each writes atomically into `d_affected[]`. No cooperative kernel needed.

#### 7.4.3 Frontier propagation inside the Louvain kernel
When an affected vertex moves, its neighbors become affected (via `d_affected[neighbor] = 1` inside the kernel after the move commits). No separate pass. The affected set grows dynamically as moves propagate — hence "frontier".

#### 7.4.4 Why this is on GPU (unlike delta-screening)
The marking rules are constant-time per edge and need no cross-vertex state. Perfect fit for a trivially parallel kernel. The delta-screening rules, by contrast, require accumulating per-community weights per source vertex — less trivially parallelizable, which is why the BTP currently keeps it on host.

### 7.5 Delta-screening dynamic variant

#### 7.5.1 Intuition
Frontier is pessimistic — it marks every endpoint of every qualifying edge change. Delta-screening further filters insertions: a vertex is only affected if there exists *some* community it could profitably move to **using only the weights of the newly inserted edges**.

#### 7.5.2 The screening formula
For source vertex $u$ with community $d = \text{community}[u]$, for each target community $c$ with inserted-edge weight $w_{u \to c}$:
$$\Delta Q_{\text{screen}}(u, c) = \frac{w_{u \to c}}{M} - \frac{k_u (k_u + \Sigma_c - \Sigma_d)}{2 M^2}$$
Mark $u$ only if $\max_c \Delta Q_{\text{screen}} > 0$.

Derivation (port from Zarayeneh & Kalyanaraman 2019/2021): this is the standard modularity gain formula with $k_{u \to c}$ replaced by the *incremental* weight from newly inserted edges.

#### 7.5.3 Three phases (host-side, current)
- **Phase 1 — Deletions**: same as Frontier (mark intra-community deletion endpoints + flag their community).
- **Phase 2 — Insertions**: sort insertions by source, iterate per source, accumulate weight per target community, apply screening formula, mark if positive.
- **Phase 3 — Propagation**: neighbors of marked vertices + all vertices in flagged communities.

#### 7.5.4 Why this is more selective than Frontier
Phase 2 screens out insertions that are not individually strong enough to pull a vertex across. On graphs with many weak random insertions, delta-screening marks a much smaller affected set.

#### 7.5.5 Current implementation lives on host
The BTP implements Phases 1–3 on the CPU using `std::map` and `std::sort`. The affected boolean array is then uploaded once. This leaves GPU compute idle during screening and incurs host-device transfer overhead.

### 7.6 Porting delta-screening to GPU (novelty hook)

This subsection is the core algorithmic contribution the BTP can claim. Expand in depth.

#### 7.6.1 Why it's not a trivial port
Each source vertex $u$ needs to aggregate inserted-edge weights *per target community*. The aggregation is a sparse reduction keyed on $(u, c)$ — conceptually a histogram-per-source. Doing this on GPU requires either:
- Sorting insertions by $(src, \text{community}[dest])$ then reducing by key; or
- A two-pass scheme with atomic adds into a thread-local sparse hash.

#### 7.6.2 Sort-based GPU design
Use `thrust::sort_by_key` to group insertions by $(src, \text{community}[dest])$, then `thrust::reduce_by_key` to accumulate per-pair weight. For each source, find the argmax target community and compare against zero. Straightforward extension of the aggregation infrastructure already in use.

#### 7.6.3 Hash-based GPU design
Warp-per-source, each warp maintains a shared-memory hash table of target-community weights while scanning its source's insertions. Faster for sources with many insertions but harder to bound memory per source.

#### 7.6.4 Propagation phase
Phase 3 (neighbor + community propagation) can be done as two additional trivial kernels: one for neighbor expansion (one thread per marked vertex), one for community expansion (one thread per vertex, check `communities_flag[community[v]]`).

#### 7.6.5 Tradeoff analysis (to include in Results)
- How much host-side time does the current delta-screening consume?
- How much transfer time?
- What fraction of total batch time?
- Do small batches benefit less because of GPU-kernel-launch overhead?
Discuss all of these, with actual numbers when available.

### 7.7 Using affected vertices in the Louvain kernel

#### 7.7.1 Kernel parameter `int* d_affected`
- `nullptr` → process all vertices (static / naive behavior).
- Non-null → skip vertex $u$ if `d_affected[u] == 0`.

Unified kernel across all four approaches; only the caller decides whether to pass `d_affected`.

#### 7.7.2 First-pass-only affectedness
`d_affected` is used only on the first pass. Subsequent passes (on aggregated graphs) process all super-nodes. Rationale: aggregated graphs are small (1–2 orders of magnitude smaller); per-vertex filtering offers little and the affected-vertex→super-node mapping is non-trivial.

#### 7.7.3 Composition with frontier propagation
Even after the first pass, the kernel maintains the affected array (marking neighbors of moved vertices during the inner local-moving loop). This lets the second inner iteration of pass 1 still use the affected optimization. This is distinct from "use affected only on the first outer pass" — the outer pass decision is cleaner and simpler.

### 7.8 Cost model and when each variant wins

Present a conceptual cost model (not full analysis):

$$T_{\text{static}} \approx c_1 \cdot (|V| + |E|) \cdot P$$

$$T_{\text{naive}} \approx c_1 \cdot (|V| + |E|) \cdot P'$$, with $P' < P$ (fewer passes)

$$T_{\text{frontier}} \approx c_1 \cdot (|A| + |E|) + T_{\text{mark}}$$, where $|A|$ is the affected set size

$$T_{\text{delta-screen}} \approx c_1 \cdot (|A'| + |E|) + T_{\text{screen}}$$, where $|A'| \leq |A|$

Discuss when each wins:
- Very small batches → delta-screening wins (smallest $|A'|$, screening cost amortizes).
- Medium batches → frontier wins (smaller $|A|$, cheap marking).
- Very large batches → naive wins (affected set approaches $|V|$, marking cost not recouped).
- Batch size approaching $|E|$ → static wins (warm-start gives no benefit).

This sets up the batch-size scaling experiment in Results.

### 7.9 State chaining across many batches

#### 7.9.1 Quality degradation over long batch sequences
Warm-started Louvain can get stuck in progressively worse local optima over many batches. Discuss whether the BTP exhibits this, and mitigations (periodic static restart, restart on detected quality drop).

#### 7.9.2 State persistence
Currently `LouvainState` lives in host memory across batches. For very large graphs this is fine (state is $O(|V|)$), but a fully device-resident pipeline would avoid round-trips. Design sketch, leave as future work.

### 7.10 Correctness arguments

Same correctness argument as static Louvain (§5.3.3) applies — the kernel is identical. Additional concerns specific to dynamic paths:
- `community_degree` must be recomputed consistently after a batch before any kernel uses it. Document where this happens in each variant.
- The `d_affected` array starts clean every batch. Verify this is cleared (it is — `cudaMemset` in each variant).
- `LouvainState.community_weight` must match the reducible sum from `LouvainState.vertex_weight` — invariant checkable cheaply in debug builds.

---

## 8. Summary

One paragraph closing: "With these design choices fixed, Chapter 5 describes the implementation in detail." Or merge Work Theory and Implementation if the report runs short.

---

## Notes for the writer

- Use pseudocode boxes liberally. UGRC-I has 8 of them — reuse.
- Every design choice should have a "why" paragraph. The worst report anti-pattern is "we did X" without "instead of Y, because Z".
- When a choice is tied to a known bug or limitation, **say so**. "This approach cannot handle pathological case X; see Future Work §N" is better writing than pretending the limitation doesn't exist.
- Figure candidates for this chapter:
  - Locking protocol state diagram.
  - Aggregation before/after CSR visualization.
  - Edge-parallel vs vertex-parallel thread-to-data mapping.
  - Affected-set timeline diagram for frontier vs delta-screening.
  - Cost-model crossover plot (when each variant wins).
  - `LouvainState` chaining diagram across successive batches.
- §7 (Dynamic Louvain) is the chapter's centrepiece — budget 40–50% of the page count for it.
