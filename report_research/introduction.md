# Introduction — Draft Content

Draft prose for the Introduction chapter. Ready for copy-paste into LaTeX with light editing. Numbers that are still firming up are marked `\TODO{...}`.

---

## 1 Introduction

### 1.1 Motivation

Graphs underpin much of modern data analysis: social media platforms, biological interaction networks, financial transaction ledgers, citation networks, and communication traffic are all naturally represented as graphs in which entities are vertices and relationships are edges. A fundamental analytic task on these graphs is **community detection** — partitioning the vertex set into groups that are densely connected internally and sparsely connected across groups. Good community partitions support downstream tasks as diverse as influence maximization, targeted advertising, protein-function prediction, fraud detection, and recommendation.

Among the algorithms proposed for community detection, the **Louvain method** of Blondel et al. \cite{blondel2008} is the practical default: it scales to networks with hundreds of millions of edges, produces partitions of high modularity \cite{brandes2007}, and is straightforward to implement. The algorithm alternates two phases — a *local-moving* phase in which each vertex greedily moves to the neighboring community that most improves modularity, and an *aggregation* phase in which each community collapses into a single super-node for the next level of the hierarchy. The algorithm terminates when no further aggregation reduces the graph.

However, Louvain was conceived for a **single static snapshot**: it processes one graph, produces one partition, and discards all intermediate state. Real-world networks rarely behave this way. Edges arrive and disappear continuously — new friendships on a social platform, new papers in a citation database, updated protein-interaction findings, expired trust relationships. Re-running Louvain from scratch on every update is wasteful: each re-run must rediscover the community structure that barely changed, and the cost grows with the total graph size rather than with the size of the update. For networks with millions of vertices, re-running static Louvain after every batch of changes makes near-real-time analysis infeasible.

This observation has motivated a family of **dynamic Louvain** algorithms that reuse the partition from the previous batch as a warm start and then refine it only where the graph has actually changed. Aynaud and Guillaume \cite{aynaud2010} introduced the naive warm-start variant; Zarayeneh and Kalyanaraman \cite{zarayeneh2021} refined this with *delta-screening*, a modularity-based filter that processes only the vertices most likely to change community; Sahu \cite{sahu2024df} proposed *Dynamic Frontier (DF) Louvain*, which incrementally expands the affected set during local moving. All of these, however, have been realised primarily on multi-core CPUs using shared-memory parallelism (OpenMP).

**Graphics Processing Units (GPUs)** offer a natural fit for Louvain's throughput-bound pattern. A GPU's thousands of lightweight threads can in principle evaluate the modularity-change formula for many vertices simultaneously, and the algorithm's per-vertex work is small and regular. Several implementations have explored this — Naim et al.'s work \cite{naim2017}, Fazlali et al.'s ACLM \cite{fazlali2020}, Bhowmick and Vadhiyar's HyDetect \cite{bhowmik2019}, and NVIDIA's cuGraph \cite{cugraph} — but each of these targets the **static** problem. To our knowledge, no peer-reviewed implementation to date realises dynamic Louvain with delta-screening *natively* on a GPU; hybrid approaches keep the update-screening logic on the CPU and ship only the modularity-optimisation phase to the device, leaving the GPU idle during screening.

### 1.2 Problem statement

This project addresses that gap: **can dynamic Louvain — including the delta-screening affected-vertex selection — run end-to-end on a GPU, and does doing so yield wall-clock speedups over the state-of-the-art CPU dynamic implementation while matching its modularity quality?** Concretely, the goal is to design, implement, and evaluate a CUDA-based dynamic Louvain that, given a sequence of batch updates (edge insertions and deletions), produces a community partition of quality comparable to running static Louvain from scratch, in a fraction of the time that such a rerun would take.

### 1.3 Contributions

This report extends the UGRC-I predecessor \cite{ugrci2025}, which implemented a static CUDA Louvain with cooperative-kernel grid synchronisation and a two-level (vertex + community) locking protocol. Building on that foundation, the contributions of this BTP are:

1. **A correctness fix to the modularity-change kernel** that removes self-loop weights from the per-community edge sum ($k_{i,\mathrm{in}}$). The unfixed formula systematically suppresses merges on aggregated graphs, leaving the algorithm stuck after the first pass; the fix is a one-line change in `calculate\_modularity\_change` but materially improves modularity on large sparse graphs — community counts drop from \TODO{52{,}286} to \TODO{103} on com-amazon and modularity rises from \TODO{0.68} to \TODO{0.92}, matching NetworkX within \TODO{0.002}.

2. **Three GPU dynamic Louvain variants** — *naive warm-start*, *frontier*, and *delta-screening* — sharing a unified CUDA kernel that accepts an optional "affected-vertex" mask. The frontier marking runs as a parallel CUDA kernel; the delta-screening phase is currently host-side and is identified as the central opportunity for further GPU work.

3. **An edge-parallel vs node-parallel kernel comparison.** The two kernels differ only in how threads are assigned to work (one thread per CSR edge vs one thread per vertex) and produce a genuine trade-off: edge-parallel wins on modularity via cascading move propagation, node-parallel is more robust (no aggregation-phase crashes) and faster per pass. Previously, the GPU Louvain literature has not systematically quantified this trade-off; this report does.

4. **A reproducible benchmarking pipeline** — shell-scripted — that runs the three CUDA variants plus NetworkX on (i) eight real-world SNAP graphs, (ii) synthetic Erdős-Rényi graphs with configurable batch-size / batch-count, and (iii) four real temporal graphs (CollegeMsg, sx-mathoverflow, sx-askubuntu, sx-superuser) split chronologically into initial + batch structure. True modularity is computed independently from the output community files, guarding against the aggregated-graph reporting error that affected the predecessor UGRC-I work.

5. **Empirical characterisation** of where the GPU dynamic variants win and where they do not. On large sparse real graphs (com-amazon, com-dblp) the best GPU variant achieves up to \TODO{$18\times$} wall-clock speedup against NetworkX while retaining modularity within \TODO{0.04}; on smaller or denser graphs the overhead of kernel launch, graph aggregation via Thrust, and host–device transfers currently dominates the benefit.

### 1.4 Organisation of the report

The remainder of this report is organised as follows. **Chapter 2 (Background)** introduces the graph and CUDA preliminaries the reader will need — the modularity objective, the Louvain two-phase structure, the GPU execution model, and the CSR representation. **Chapter 3 (Related Work)** surveys prior work on static and dynamic Louvain on both CPU and GPU, with particular attention to the delta-screening and Dynamic Frontier algorithms this BTP builds upon. **Chapter 4 (Design)** describes the parallel-programming choices — the two-level locking protocol, the cooperative-kernel synchronisation, and the `LouvainState` warm-start abstraction — and explains how each of the three dynamic variants is realised on top of the shared kernel. **Chapter 5 (Implementation)** covers the concrete CUDA code, the self-loop bug and its fix, the $k_i$ precomputation, and the edge-parallel vs node-parallel kernel variants. **Chapter 6 (Evaluation)** reports results across the three graph families (real SNAP, synthetic ER, real temporal), compares against NetworkX, cuGraph, and DF-Louvain OpenMP baselines, and analyses the patterns and anomalies observed. **Chapter 7 (Conclusion)** summarises the contributions, discusses the limitations — most importantly, the host-side delta-screening that remains as future work — and points to directions beyond this report, including a GPU-native delta-screening phase and an extension to the Leiden algorithm.

---

## Writer's notes

- Citations here are in `\cite{...}` form and match the keys recommended in `related_work.md` — `blondel2008`, `brandes2007`, `aynaud2010`, `zarayeneh2021`, `sahu2024df`, `naim2017`, `fazlali2020`, `bhowmik2019`, `cugraph`, `ugrci2025`. Add these to the `.bib` file before compiling.
- `\TODO{...}` markers denote numbers that must be re-verified against the final Results chapter before submission. Every numerical claim in 1.3 already has a concrete defensible figure in the benchmarks; update before submission.
- Tone is intentionally cautious — "to our knowledge", "in a fraction of the time". Avoid bolder "first ever" claims until the related work is re-checked at submission time.
- Length: approximately 900 words. Scales to ≈2 pages of double-column IJPP/JPDC layout.
- §1.3 lists five contributions — a common reviewer request is to reduce to 3–4 strongest; if needed, merge (2) and (3) under "GPU dynamic Louvain variants with a kernel-level parallelism study".
- §1.4 gives the chapter map. Adjust chapter numbers to match actual compiled report.
