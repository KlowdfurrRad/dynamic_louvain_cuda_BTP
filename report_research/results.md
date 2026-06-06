# Results — Section Plan

Planning document for the Results chapter. Lists every benchmark to run, every chart to produce, every baseline to compare against. The goal is to leave no obvious question unanswered when a reviewer (IJPP / JPDC / TPDS) reads the chapter.

**Every chart and table must report three columns/axes minimum: modularity (Q), number of communities, time. Modularity alone is insufficient — community count reveals whether the algorithm is stuck (see the nx-vs-cuda gap on com-amazon).**

---

## 1. Experimental Setup (Section 7.1 of the report)

### 1.1 Hardware
State the GPU used (model, SMs, CUDA cores, memory, compute capability, driver, CUDA version). CPU (for baselines): model, cores, clock, memory. OS. Mandatory for reproducibility.

### 1.2 Software
- CUDA toolkit version.
- Thrust version (bundled with CUDA).
- cuGraph version.
- NetworkX version.
- GVE-Louvain build hash.
- DF-Louvain OpenMP build hash (`puzzlef/louvain-communities-openmp-dynamic`).
- Compiler flags (`-rdc=true -arch=sm_XX -O3`).

### 1.3 Datasets (static benchmarks)
Reuse the SNAP 8-graph set from `notes/real_small_graph_datasets.md`. Report $|V|$, $|E|$, density, max degree, reference ground-truth communities (where available). Standard table:

| Graph | Nodes | Undir. Edges | Type |
|---|---|---|---|
| ca-GrQc | 5,241 | 14,484 | Co-authorship |
| facebook (ego) | 4,039 | 88,234 | Social |
| ca-HepTh | 9,875 | 25,973 | Co-authorship |
| ca-HepPh | 12,006 | 118,489 | Co-authorship |
| ca-AstroPh | 18,771 | 198,050 | Co-authorship |
| email-Enron | 36,692 | 183,831 | Email |
| com-amazon | 334,863 | 925,872 | Product co-purchase |
| com-dblp | 317,080 | 1,049,866 | Co-authorship |

Add at least 2 larger graphs (com-LiveJournal or com-Orkut, ~1M+ nodes) if hardware allows — journal reviewers expect scaling beyond 1M vertices.

### 1.4 Datasets (dynamic benchmarks)
- **Synthetic batches** on the above static graphs (random insertions/deletions).
- **Real temporal graphs** if available — e.g., sx-stackoverflow, wiki-talk-temporal. Argue why temporal data matters (insertion/deletion distribution not uniform). In DF-Louvain, sx-mathoverflow, sx-askubuntu, sx-superuser, wiki-talk-temporal, sx-stackoverflow are used from SNAP for temporal graphs.

### 1.5 Methodology
- 5 runs per configuration, report mean ± stdev.
- Fixed random seed for deterministic ordering where applicable.
- Warm-up run discarded.
- Modularity measured on the **original graph** with final community assignments (not the aggregated graph — which is the UGRC-I reporting error this BTP fixes).
- Number of communities = number of distinct values in the final `original_community[]` array.
- Time = end-to-end wall-clock of the Louvain call (host-to-device upload + kernel + device-to-host download).

---

## 2. Baselines (Section 7.2)

Every reviewer will ask "what did you compare against?". Answer preemptively:

### 2.1 Mandatory
- **cuGraph Louvain (GPU, RAPIDS)** — *compulsory per author's requirement*. The de-facto GPU Louvain baseline. Used in UGRC-I. Install via RAPIDS conda package or docker.
- **NetworkX Louvain (CPU, single-thread)** — sanity / quality ceiling. Already wired up (`algorithm/nx_louvain.py`). Slow but produces the highest-quality partitions.
- **GVE-Louvain (CPU, OpenMP, shared memory)** — current state of the art on CPU. Mandatory after Sahu 2023. Build from `puzzlef/GVE-Louvain`.
- **DF-Louvain OpenMP (CPU, dynamic)** — the direct dynamic-Louvain competitor. Already cloned as `louvain-communities-openmp-dynamic`. Used for dynamic benchmarks.

### 2.2 Strongly recommended
- **Static-rerun-from-scratch on the updated graph** — the dynamic-Louvain speedup is measured against this. Without it, "dynamic is fast" is meaningless.
- **UGRC-I predecessor** — since this BTP extends UGRC-I, include those numbers in the headline comparison so the improvement is visible.
- **ν-Louvain (Sahu 2025, arXiv 2501.19004)** — the newest GPU Louvain. Difficult to beat; worth comparing at least on a subset of graphs.

### 2.3 Optional (mention even if not run)
- **Naim et al. 2017** — classic GPU Louvain. No public implementation but authors' results are in the paper.
- **ACLM / Mohammadi-Fazlali 2020** — no widely available implementation.
- **cuGraph Leiden** — different algorithm, but a fair "modern GPU community detection" reference.

### 2.4 Baseline selection matrix
For each experiment type, state explicitly which baselines are included:

| Experiment | cuGraph | NetworkX | GVE-Louvain | DF-Louvain | Static-rerun | UGRC-I | ν-Louvain |
|---|---|---|---|---|---|---|---|
| Static quality & time | ✓ | ✓ | ✓ | — | — | ✓ | ✓ |
| Dynamic vs batch size | ✓ static | — | ✓ static | ✓ | ✓ | ✓ | — |
| Dynamic vs graph size | ✓ static | — | — | ✓ | ✓ | ✓ | — |
| Scaling (GPU util) | — | — | — | — | — | — | — |

Explain each ✓ and — choice in the caption.

---

## 3. Charts and Tables — Static Experiments (Section 7.3)

### 3.1 Headline table: Q, communities, time across all 8 static graphs
Three implementations from this BTP (static, dynamic-edge, dynamic-node), side by side with NetworkX, cuGraph Louvain, GVE-Louvain. 8 graphs × 6 implementations × 3 metrics.

This is the **single most important table in the paper**. Reuse `real_graphs/results_table.py` output and add the external baselines in the same script.

### 3.2 Quality chart: modularity across graphs
Grouped bar chart. X-axis: graph (sorted by size). Y-axis: modularity. One bar group per implementation.

Expected: NetworkX and GVE-Louvain at the top; this BTP close but below on some graphs; cuGraph middle; node-based slightly below edge-based.

### 3.3 Community count chart
Same layout, Y-axis: community count (log scale). The "NetworkX 231 vs CUDA 52,286 on com-amazon" gap is the most striking chart in the whole report. After the self-loop bug fix, show the new numbers and whether the gap closes.

### 3.4 Time chart
Same layout, Y-axis: wall-clock seconds (log scale). Expected: GPU wins on large graphs, CPU is competitive on small ones.

### 3.5 Speedup chart
Y-axis: speedup over NetworkX (log scale). One line per implementation. Makes the GPU advantage on large graphs visible.

---

## 4. Charts and Tables — Dynamic vs Batch Size (Section 7.4) **(mandatory)**

This is the core dynamic experiment. Fix one graph (or a small set — pick 3: one small dense, one medium, one large sparse — e.g., facebook, ca-AstroPh, com-amazon). Vary batch size.

### 4.1 Setup
- Generate synthetic batches with sizes $|\Delta E| \in \{0.1\%, 0.5\%, 1\%, 5\%, 10\%, 20\%, 50\%\}$ of $|E|$.
- 50% insertions, 50% deletions. (Also report a second suite with all-insertions and all-deletions.)
- 3 batches per size (or more if time permits); report averages.

### 4.2 Implementations compared (on each batch size point)
- **BTP static rerun** — re-run static Louvain from scratch on the updated graph. Upper bound on time, upper bound on quality.
- **BTP naive dynamic** — warm-start, all vertices.
- **BTP frontier dynamic** — selective.
- **BTP delta-screening dynamic** — most selective.
- **DF-Louvain OpenMP** — *direct competitor baseline*.
- **cuGraph static rerun** — cuGraph has no dynamic mode, so this is the "what you get if you use a mainstream GPU library on dynamic workloads" baseline.

### 4.3 Charts
- **Chart 4.3a (modularity vs batch size)** — one line per implementation. Expected: all converge toward the same Q as batch grows; naive/frontier/delta should match static-rerun.
- **Chart 4.3b (communities vs batch size)** — catches quality degradation in warm-started variants.
- **Chart 4.3c (time vs batch size)** — log-log. The core speedup story. Expected: dynamic variants are orders of magnitude faster on small batches; crossover with static-rerun somewhere around 10–20% batch size.
- **Chart 4.3d (speedup vs static-rerun)** — ratio, one line per dynamic variant. Makes "how much does dynamic help?" explicit.

### 4.4 Table
For each batch size × implementation: (Q, communities, time, speedup over static-rerun, speedup over DF-Louvain). Long table — go to landscape orientation if needed.

### 4.5 Narrative expected
- Small batches: BTP delta-screening beats frontier beats naive beats static-rerun.
- Medium batches: crossover; frontier may overtake delta-screening as screening overhead stops paying for itself.
- Large batches: static-rerun catches up; warm-start doesn't help when the graph has fundamentally changed.
- Quality: does the dynamic variants' Q match the static-rerun Q? If not, by how much, and is the time saving worth the quality loss?

---

## 5. Charts and Tables — Dynamic vs Graph Size (Section 7.5) **(mandatory)**

Complementary to §4. Fix batch size (e.g., 1% of edges) and vary graph size.

### 5.1 Graph size axis
Two versions of this experiment:
- **By $|V|$**: order the 8 SNAP graphs by $|V|$, plot time / Q / communities. Linear or log-scale on X depending on spread.
- **By $|E|$**: same graphs ordered by $|E|$ instead. Reveals whether the BTP is edge-bound or vertex-bound.

Consider also a **synthetic scaling** suite: Erdős-Rényi graphs at fixed density with $|V| \in \{10^4, 10^5, 10^6\}$, to get clean scaling curves free of graph-specific artifacts.

### 5.2 Implementations compared
Same set as §4.2.

### 5.3 Charts
- **Chart 5.3a (time vs $|V|$, fixed batch size)** — log-log. Slope reveals asymptotic complexity. Expected: dynamic variants grow slower than $|V|$, static-rerun grows linearly with $|V|$.
- **Chart 5.3b (time vs $|E|$, fixed batch size)** — same. Expected: similar pattern, slope closer to 1.
- **Chart 5.3c (modularity vs $|V|$)** — ensures quality does not degrade as graphs grow.
- **Chart 5.3d (communities vs $|V|$)** — catches the self-loop-bug class of regressions where community count blows up on large graphs.

### 5.4 Table
One row per graph × implementation with (Q, communities, time). Gets long; consider splitting into one subtable per implementation with a common leftmost graph column.

### 5.5 Narrative expected
- Do the dynamic variants maintain their advantage on large graphs, or does static-rerun become competitive because of GPU pipeline efficiency?
- Does edge-based vs node-based scale differently?
- Is DF-Louvain (CPU) ever competitive on small graphs (where GPU launch overhead dominates)?

---

## 6. Ablation and Diagnostic Studies (Section 7.6)

Not mandatory, but strong submissions include at least one or two.

### 6.1 Edge-based vs node-based kernel
On the same 8 graphs: Q, communities, time, robustness (crash / no-crash). Small dedicated table and bar chart. One-paragraph discussion — what does the tradeoff look like in practice?

### 6.2 Self-loop-bug before/after
Two columns of the headline table: before the fix vs after the fix. Communities and Q. Direct demonstration that the bug caused the nx-vs-cuda gap. Candidate for the report's most compelling figure.

### 6.3 Impact of the `delta_Q_sum < 1e-6` early-exit threshold
Vary the threshold: $\{10^{-3}, 10^{-4}, 10^{-5}, 10^{-6}, 10^{-8}, 0\}$. Plot Q and time. Justifies the chosen default.

### 6.4 Affected-vertex count over time
For one dynamic benchmark, plot `|affected|` over kernel iterations. Shows frontier propagation behavior. Diagnostic figure; easier to discuss in Work Theory.

### 6.5 Kernel time breakdown (`nsys` / `ncu`)
Use NVIDIA profiler. Report: fraction of time in `louvain_kernel` vs `aggregate_graph` vs `combine_edges` vs host-device transfers. Justifies optimization priorities and exposes whether delta-screening host-side work is the bottleneck.

### 6.6 Multi-batch quality stability
Run 10 consecutive batches on the same graph. Does Q monotonically improve, degrade, or oscillate? Does it track a periodic static-restart? Informs future-work on when to restart from scratch.

---

## 7. Patterns, Anomalies, and Analysis (Section 7.7)

This section is where raw numbers become insights. Every experiment section above produces data; this section interprets it. **A reviewer will trust the paper much more if surprising results are acknowledged and explained rather than hidden.**

### 7.1 Cross-graph patterns to call out

Look for patterns that hold (or fail) across the full 8-graph suite. Concrete candidates based on preliminary data:

- **Sparse vs dense behavior**: dense graphs (facebook) converge in 1–2 aggregation passes, sparse graphs (com-amazon, com-dblp) need many more. The dynamic variants' benefit therefore varies by density — frontier propagation helps more on dense graphs where one move cascades to many neighbors.
- **Graph family (co-authorship vs social vs communication)**: co-authorship graphs (ca-GrQc, ca-HepTh, ca-HepPh, ca-AstroPh) show consistently lower modularity than community-structure graphs (com-amazon, com-dblp) because the internal communities are less well-defined. Report this as a family-effect rather than an implementation quality effect.
- **Ground-truth-rich graphs behave differently**: com-amazon and com-dblp have ground-truth communities and very high NetworkX modularity (0.92, 0.82). The CUDA implementations consistently leave more communities than NetworkX on these graphs — a pattern worth noting.
- **GPU launch overhead dominates small graphs**: on graphs $< 10^4$ vertices, CPU baselines (NetworkX, GVE-Louvain) beat GPU Louvain. The crossover is somewhere around $|V| = 10^4$–$10^5$. State it explicitly.

### 7.2 Graph-specific anomalies

For each graph that behaves unexpectedly, give a paragraph. Based on the current data:

- **facebook (4K nodes, 88K edges)**: unusually dense (avg degree ≈ 44) and produces by far the highest modularity (≈0.83) of any graph in the suite. Edge-based dynamic achieves 0.814 — within 0.02 of NetworkX — but node-based drops to 0.806 and static struggles (0.710 in the old runs). Explanation candidate: dense graphs reward cascading edge-based propagation; the greedy first-positive-move heuristic works well when most neighbors are in the target community already.

- **ca-HepTh (10K nodes, 26K edges)**: large modularity gap (CUDA 0.60 vs NetworkX 0.77). Suspected cause: high clustering coefficient with many small triangular motifs. The edge-greedy approach makes irreversibly suboptimal first moves that trap the partition in a local optimum. Suggest graph-coloring-based parallel safe moves as mitigation — write this as future work.

- **ca-AstroPh, ca-HepPh, email-Enron — edge-based crashes**: nondeterministic illegal memory access during aggregation. Acknowledge openly. Hypothesis: a race condition between `aggregate_graph` and `combine_edges` on denser aggregated graphs; the node-based kernel does not trigger the race because of its different access pattern. Node-based is therefore the "production-safe" variant. Include in Limitations.

- **com-amazon (335K nodes, 926K edges)**: NetworkX reaches Q = 0.926 with 231 communities; CUDA before self-loop fix produced 52K communities with Q = 0.677. After the fix, report the new numbers. This is the single graph where the bug-fix impact is clearest — use it as the Figure for the self-loop ablation (§6.2).

- **com-dblp (317K nodes, 1M edges)**: similar pattern to com-amazon but lower baseline. The CPU GVE-Louvain time is ≈20s, cuGraph is faster, but the quality gap persists.

### 7.3 Why specific graphs trigger specific behaviors

Be explicit about the causal mechanisms. Paragraph structure:

- **Density → convergence speed**: denser graphs give each vertex more information per pass (more neighbors to evaluate), so fewer passes needed.
- **Clustering coefficient → local-optimum risk**: high clustering = many near-optimal neighbor communities = higher chance of parallel greedy picking a non-global-optimal one.
- **Power-law degree distribution → load imbalance**: graphs with hubs (social, web) exhibit warp-level load imbalance in the node-parallel kernel. Edge-parallel fixes it but introduces the aggregation race.
- **Ground-truth structure → quality ceiling**: product co-purchase and citation networks have genuine modular structure, so the Q ceiling is high; communication graphs have noisier structure and lower ceiling regardless of algorithm.

### 7.4 Speedup pattern analysis

- Where does dynamic win and where does it lose? Expect a clean "small batches → dynamic wins, large batches → static wins" crossover; if observed, plot it; if not, investigate why.
- Where does GPU win and where does it lose? Small graphs lose due to launch overhead; very large graphs win. Report the crossover.
- Where does edge-based beat node-based, and vice versa? Expect edge-based to win on dense graphs (cascading), node-based on sparse graphs (lower overhead).

### 7.5 Quality gap analysis vs GVE-Louvain / NetworkX

Quantify and explain the gap, don't hide it. Candidate explanations to test:
- Parallel edge-greedy first-positive-move strategy (vs sequential best-move). Test by running a node-based variant that uses best-move — does the gap close?
- Lack of Leiden-style refinement — Louvain is known to produce sometimes-disconnected communities.
- Aggressive early-exit threshold. Vary per §6.3.
- Batch-size artifacts specific to dynamic runs (gap may only appear after several chained batches).

### 7.6 Reporting surprises honestly

If an experiment produces a counter-intuitive result (e.g., frontier slower than naive on some graph, or delta-screening quality worse than frontier on another), **do not hide it**. Add a subsection explaining what happened. Reviewers trust papers that report both the wins and the losses.

---

## 8. Limitations and Positives — Notes for Future Researchers (Section 7.8)

Framed for a student who picks up this codebase in 2027. Every limitation deserves a workaround note; every positive deserves a "keep doing this" tag.

### 8.1 What worked well (positives — keep)

- **Cooperative-kernel grid synchronization**: enables multi-iteration Phase 1 in a single kernel launch without per-iteration relaunch overhead. Keep this; it is the reason the GPU times are competitive on large graphs.
- **Two-level locking (vertex + community, sorted order)**: produces correct results without deadlocks. Keep the locking protocol — the correctness argument is well-understood.
- **Thrust for aggregation**: `sort` + `reduce_by_key` + `exclusive_scan` is concise, correct, and near-optimal. No reason to reimplement.
- **Vertex-centric community tracking (`d_original_community`)**: a single $O(n)$ update per pass, fully parallel. Significantly simpler than the community-centric alternative.
- **Unified `louvain_kernel` with optional `d_affected`**: keeps the four variants (static, naive, frontier, delta-screening) on a common code path. Easier to maintain than four separate kernels.
- **`LouvainState` as the warm-start interface**: clean separation between static and dynamic. Makes chaining batches trivial. Future Leiden extension can reuse this struct.

### 8.2 What didn't work / known limitations

- **Edge-based kernel crashes during aggregation on denser graphs** (ca-HepPh, ca-AstroPh, email-Enron). Nondeterministic. Workaround: use the node-based kernel in production. Root-cause fix is future work; suspected aggregation race.
- **Delta-screening runs on the host**. The ~20–30% of batch time spent in CPU `std::map` + `std::sort` is wasted. GPU port design is sketched in `work_theory.md §7.6`; implement it.
- **CSR rebuild per batch**: $O(|V| + |E|)$ cost regardless of batch size. A delta-CSR or tombstone scheme would save an order of magnitude on small batches; left as future work.
- **Input format inconsistency**: initial edges read `u v` (weight = 1.0 hardcoded) while batches read `u v w`. Fix before any paper submission.
- **Quality gap vs CPU baselines on high-clustering graphs** (ca-HepTh, ca-HepPh): ~0.1 modularity gap persists even after self-loop fix. Attributed to parallel-greedy local-optimum trapping. Future work: graph-coloring-based safe moves, or Leiden refinement.
- **No multi-GPU scaling**: the implementation is single-GPU. Gawande et al. cuVite or Distributed Multi-GPU Community Detection (OSTI) would be the reference for extending.
- **Host-device transfer on every batch**: the adjacency list lives on host, gets re-uploaded per batch. A fully device-resident pipeline would eliminate this round trip.
- **No resilience to very long batch sequences**: quality may degrade over many chained batches without a periodic static restart. Detection mechanism (quality-drop threshold → auto-restart) is future work.

### 8.3 Gotchas that cost us time (warnings)

- **`volatile __device__` + `atomicAdd` + `cudaMemcpyFromSymbol`** interaction appeared to return stale values in early debugging; root cause turned out to be algorithmic (self-loop bug), not memory-model. Future debuggers: verify the kernel's own `printf` matches `cudaMemcpyFromSymbol` output before blaming the memory model.
- **Reported modularity ≠ true modularity** when measured on the aggregated graph. Always recompute modularity on the original graph with `d_original_community` for paper numbers.
- **`cudaLaunchCooperativeKernel` requires all blocks to fit concurrently** on the GPU. With 32 blocks × 512 threads, this is fine on consumer GPUs. If scaling up, check `cudaOccupancyMaxPotentialBlockSize`.
- **The `-rdc=true` flag is mandatory** for cooperative kernels. Without it, you get a cryptic runtime error.
- **`combine_edges` frees and reallocates `d_csr_adj`** — must pass by reference from the caller. This caused a subtle double-free in an early version.

### 8.4 Reproducibility notes for future runs

- Pin the RAPIDS / cuGraph version; API changes between 24.x and 25.x silently change behavior.
- GPU time is hardware-dependent; always report the GPU model used.
- Random seeds matter for dynamic experiments — a small batch on a lucky seed can give very different numbers.
- The `results_table.py` script computes true modularity independently of the CUDA run; always validate the CUDA-reported modularity against it before trusting a number.

---

## 9. Reproducibility Appendix (Section 7.9)

### 9.1 Exact commands used
For each experiment, the command that produced the numbers. Shell script dropped into `real_graphs/` if not already.

### 9.2 Output manifest
Which CSV / log files produced each chart. Archived somewhere checked in.

### 9.3 Hardware details and CUDA profile
`nvidia-smi` output, `cudaDeviceProperties` dump. Short.

---

## 10. Expected Narrative (to inform writing)

After all experiments are in:

1. **We implemented three dynamic GPU Louvain variants — naive, frontier, delta-screening — on top of an improved static CUDA Louvain.**
2. **The fixed self-loop bug alone brings modularity up from $X$ to $Y$ on large sparse graphs and community count down by $\sim Z\times$, matching NetworkX within $\epsilon$ on most graphs.**
3. **On batch sizes below $\sim 5\%$, the delta-screening variant achieves $N\times$ speedup over static-rerun and $M\times$ over DF-Louvain OpenMP, with no significant quality loss.**
4. **Our node-based kernel is more robust than edge-based (no aggregation crashes) at the cost of $\sim K\%$ modularity — the tradeoff to apply in production.**
5. **The remaining quality gap vs NetworkX / GVE-Louvain on $G$ is $\sim \delta$ — attributed to parallel edge-greedy conflict resolution; future work on graph coloring may close it.**
6. **Patterns observed across graph families (co-authorship, social, communication, co-purchase) reveal when each dynamic variant wins; density and clustering coefficient are the dominant predictors.**
7. **Known limitations (aggregation crash, host-side screening, CSR rebuild cost) are documented so future researchers can pick up exactly where this work leaves off.**

These sentences drive the Abstract and Introduction of the paper once numbers are filled in.

---

## Notes for the writer

- Every chart caption must name the graph, the hardware, and the units. Reviewers skim figures; captions have to be standalone.
- Don't report time without reporting Q alongside. Fast + wrong is not a result.
- Error bars or confidence intervals where applicable. Five runs is enough for mean ± stdev.
- Log scales are almost always right for time and community count. Linear is almost always right for modularity.
- If a baseline fails to run on a graph, report that as a data point (put CRASH/OOM in the table). Don't silently omit.
- **Mandatory experiments per user's requirement**: §4 (batch size variation) and §5 (graph size variation), both with Q, communities, time per data point.
