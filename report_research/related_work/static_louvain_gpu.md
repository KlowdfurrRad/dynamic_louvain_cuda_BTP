# Static Louvain — GPU

Writing material for the "Static Louvain (GPU)" paragraph(s) of the Related Work section. This is the most directly comparable cluster to the BTP's static-Louvain GPU kernels. Each entry is a paper, a short discussion of what it contributes and how it relates to the BTP, and the LaTeX citation key + BibTeX entry.

---

## Cheong, Huynh, Lo, Goh (2013) — Hierarchical Parallel Louvain on GPUs

Self-described as "the first attempt to accelerate the Louvain method, or modularity-based methods in general, on the GPU platform" (Section 7). The paper identifies three levels of parallelism in Louvain and maps them onto a single-GPU and a multi-GPU implementation.

**Three levels of parallelism (Section 4, paper-faithful).** Not degree-based. The three levels are conceptually orthogonal:

1. **Highest level — network partitioning across multiple GPUs.** The original network is split into sub-networks based on contiguous node-ID ranges. Inter-sub-network edges are set aside. Louvain runs in parallel on each sub-network; results are then combined with the saved inter-sub-network edges and re-Louvained sequentially.
2. **Middle level — batched parallel node visiting within an iteration.** Vanilla Louvain visits nodes sequentially; this paper processes a batch of nodes concurrently. They explicitly tolerate the resulting inaccuracy (two nodes in one batch may be neighbours), permitting only atomic updates to a node's community status.
3. **Lowest level — parallel evaluation of multiple neighbouring communities for one node.** Multiple GPU threads cooperatively evaluate the modularity gain of moving a node into each of its neighbouring communities; one thread does the cross-community arg-max.

**What's offloaded to GPU and what isn't.** Their profiling (Section 3.2) shows ~94.8% of runtime is spent in **Pass 1** (the first local-moving phase, before any community aggregation). The implementation therefore offloads *only* the **FNC** (Find Neighboring Communities) and **FBM** (Find Best Move) kernels of Pass 1 to the GPU. All later passes — and aggregation between passes — remain on the CPU.

**FNC and FBM in detail.** FNC is the per-node step that builds the set of distinct neighbouring communities and accumulates the per-community in/out edge weights $k_{i,C^{in}}$ and $k_{i,C^{out}}$. The GPU implementation does this via a Thrust radix sort over `(node, neighbour-community)` pairs followed by a per-node weight reduction. FBM is the modularity-gain evaluation and the move decision; it uses the second + third levels of parallelism (multiple nodes per SM, multiple threads per node).

**Hardware (Section 6).** Intel Xeon E5405 @ 2 GHz + 4 × NVIDIA Fermi C2070 GPUs. Code compiled with `-O3`. Reported timings exclude I/O.

**Test networks (Table 1).** uk-2005, webbase-2001, twitter (extracted via Twitter API), flickr, livejournal, orkut, facebook. Sizes range from 2.3 M to ~20 M nodes; up to ~287 M links. **Not the SNAP suite.** For uk-2005 and webbase-2001 they had to take random sub-networks because the original Louvain could not fit them in memory.

**Headline speedups (Table 4).** SingleGPU vs. sequential Louvain: **1.72×–4.98×** across the seven networks (best on the three largest: uk-2005, webbase-2001, orkut). MultiGPU (4 GPUs) adds **1.27×–3.48×** on top of SingleGPU. The smallest MultiGPU win is on facebook because of a load-balancing failure: one of the four sub-networks ended up with disproportionately many links.

**Quality results (Table 5).** SingleGPU's modularity is **equal to or *better* than** sequential Louvain on **6 of 7** networks (the only loss is twitter at −1.32%). The improvements on flickr, livejournal, orkut are +1.53%, +3.0%, +3.18% respectively. MultiGPU degrades modularity by an average of **~3%** (range −0.98% to −9.23% for livejournal). So: SingleGPU is essentially "free" or better on quality; MultiGPU sacrifices ~3% for additional speedup.

**Interesting bit (paper-attested).** The headline quality finding is counter-intuitive: the SingleGPU version *improves* on sequential Louvain's modularity on most inputs, despite using parallel-batch evaluation that violates the strict sequential semantics. The paper attributes this to the "inaccuracy" being benign — vertices in the same batch making decisions on each other's stale community labels does not, on these inputs, hurt the final partition quality.

For the BTP, Cheong et al. are the historical-record reference: the first GPU Louvain. Their three-levels-of-parallelism framing is the conceptual ancestor of subsequent GPU Louvain papers (Naim 2017, Mohammadi 2020, cuVite 2022), although those later works re-define the levels differently (typically degree-based instead of phase-based).

Citation key: `cheong2013hierarchical`.

```bibtex
@inproceedings{cheong2013hierarchical,
  author    = {Cheong, Chun Yew and Huynh, Huynh Phung and Lo, David and Goh, Rick Siow Mong},
  title     = {Hierarchical Parallel Algorithm for Modularity-Based Community Detection Using {GPU}s},
  booktitle = {Euro-Par 2013 Parallel Processing},
  series    = {LNCS 8097},
  pages     = {775--787},
  year      = {2013},
  publisher = {Springer},
  doi       = {10.1007/978-3-642-40047-6_77}
}
```

---

## Naim, Manne, Halappanavar, Tumeo (2017) — Community Detection on the GPU

> ⚠️ **Verification note.** I do not currently have an open-access copy of this paper. The full text is behind the IEEE Xplore paywall (DOI 10.1109/IPDPS.2017.16). The paper also exists as Paper III of Md Naim's PhD thesis ("Parallel Matching and Clustering Algorithms on GPUs", UiB 2017), but the BORA UiB repository now redirects to a JavaScript-only NVA portal that I could not fetch. The summary below is restricted to facts that can be cross-confirmed from secondary sources (citing papers like Halappanavar 2017 reference [13], Mohammadi 2020, Sahu 2025) and from the abstract / venue metadata. **Specific algorithmic details that cannot be independently confirmed have been removed.**

Most-cited GPU Louvain baseline; the reference point most later GPU-Louvain papers (Mohammadi 2020 ACLM, cuVite, Bhowmik HyDetect, ν-Louvain) benchmark against. The paper presents a CUDA implementation of the Louvain method, with the design contribution being a degree-aware mapping of vertices to GPU threads/warps/blocks for the local-moving phase, and a Thrust-based on-GPU graph aggregation step.

**What's verifiable from the bibliographic record:**
- Published at *2017 IEEE International Parallel and Distributed Processing Symposium (IPDPS)*, Orlando, FL, USA, pages 625–634.
- Authors: Md Naim (UiB), Fredrik Manne (UiB), Mahantesh Halappanavar (PNNL), Antonino Tumeo (PNNL).
- DOI: 10.1109/IPDPS.2017.16.

**What I cannot verify without the PDF** (and therefore no longer claim in this entry): the specific degree thresholds for kernel selection, the exact aggregation primitive sequence, whether Grappolo's min-label tie-breaking is explicitly used, the headline modularity-vs-sequential percentage, the headline speedup-vs-Grappolo number, the GPU model used for benchmarks, or the iteration-count behaviour relative to sequential Louvain.

**For the BTP.** Cite Naim 2017 as the canonical static GPU Louvain baseline. Any specific algorithmic comparison (e.g. "the BTP's edge-based kernel resembles Naim's per-edge approach") needs to be re-verified against the actual paper — possibly via institutional IEEE Xplore access — before being asserted in the report. *Action item:* obtain the PDF before final submission and revise this entry with paper-faithful detail.

Citation key: `naim2017community`.

```bibtex
@inproceedings{naim2017community,
  author    = {Naim, Md and Manne, Fredrik and Halappanavar, Mahantesh and Tumeo, Antonino},
  title     = {Community Detection on the {GPU}},
  booktitle = {2017 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  pages     = {625--634},
  year      = {2017},
  doi       = {10.1109/IPDPS.2017.16}
}
```

---

## Bhowmik, Vadhiyar (2019) — HyDetect

A divide-and-conquer hybrid CPU-GPU Louvain targeting **graphs too large to fit in GPU memory**. Source code released at [github.com/marslabiisc/HyDetect](https://github.com/marslabiisc/HyDetect).

**The actual algorithm (Section IV, paper-faithful).**

1. **Partitioning.** The graph is split into two parts — one for the CPU, one for the GPU — using **Metis** (preferred over 1-D vertex-block or ParMetis after empirical comparison). The split ratio is determined by a one-time calibration: 3 randomly-induced subgraphs of 5–10 % of the vertex count are run on both devices, and the average CPU/GPU runtime ratio is used as the Metis edge-cut ratio.
2. **Independent Louvain on each device.** CPU side uses Lu et al.'s multicore Grappolo [`lu2015parallel`]; GPU side uses Naim et al.'s 2017 implementation [`naim2017community`]. Both run to local convergence on their own subgraph.
3. **Doubtful-vertex identification.** Each device flags vertices it suspects of being mis-assigned because they were decided with only partial graph information. The flagging uses an *affinity* heuristic (defined below).
4. **Migration of doubtful vertices.** Doubtful vertices that are more strongly connected to the *other* device's subgraph are moved across, using `cudaMemCpyAsync`. Of all doubtful vertices on a device, those with `border-edges / non-border-edges > θ` are selected for migration (Algorithm 2 of the paper).
5. **Coarsen, repeat.** After migration, each device coarsens its part by collapsing communities into super-vertices. Steps 2–5 repeat until both devices' coarsened graphs together fit in GPU memory.
6. **Final pass on GPU.** All remaining communities are moved to the GPU (chosen as `finalDevice` because GPU is reported as ≥10× faster on the small final graph), and a last Louvain run produces the final partition.

**The affinity / doubtful-vertex heuristic (Section IV-C).** For a vertex $v$ in community $c$, define **relative commitment** $\text{RC}(v,c) = \dfrac{\text{IntDegree}(v,c)}{\max_{v_i \in c} \text{IntDegree}(v_i, c)}$. Then $\text{IntRC}(v,c)$ is the sum of $\text{RC}$ over $v$'s neighbours *within* community $c$, and $\text{ExtRC}(v,c)$ is the sum over $v$'s neighbours *in other* communities (within the same device). Affinity is

$$\text{aff}(v, c) = \frac{\text{RC}(v, c) \cdot \text{IntRC}(v, c)}{\text{IntRC}(v, c) + \text{ExtRC}(v, c)} \in [0, 1].$$

A vertex is **doubtful** iff $\text{aff}(v, c) < \theta$.

**Hardware (Section VI).** Most experiments on a 6-core Intel Xeon E5-2620 @ 2.10 GHz + NVIDIA Kepler K40M (2880 SP cores, 12 GB), called the "Turing node". The largest graphs (uk-2002, RMat24) used a dual-octo-core Xeon E5-2670 + NVIDIA K20m (4.68 GB).

**Test graphs.** uk-2002 (18.5 M vertices, 523 M edges), rMat24 (16.8 M / 536 M), eu-2015 (11.2 M / 759 M), and others. **All chosen specifically to exceed K40M's 12 GB GPU memory** — that is the use case HyDetect targets.

**Headline results.** The paper's quantitative claims (paper-faithful):
- "almost 2× speedup over multi-core parallel Louvain algorithm" (i.e. over Grappolo).
- "less than 1 % change in modularity" relative to multi-core parallel Louvain.
- "42–73 % smaller execution times" over CPU-only state-of-art parallel Louvain on the large graphs that don't fit in GPU memory.

**For the BTP.** HyDetect is the *out-of-core* GPU Louvain reference: the relevant scenario is "graph too big for one device" rather than "make one-device Louvain faster". The BTP targets a different regime (single GPU, graph fits). The conceptual link to the BTP is that the affinity-based doubtful-vertex notion is structurally similar to a frontier/active-set — but the affinity formula (a ratio of internal-vs-external commitment) is not the same as delta-screening's modularity-gain bound. *No claim made by the BTP about HyDetect's mechanism should be made without re-checking the affinity formula above.*

Citation key: `bhowmik2019hydetect`.

```bibtex
@inproceedings{bhowmik2019hydetect,
  author    = {Bhowmik, Anwesha and Vadhiyar, Sathish},
  title     = {{HyDetect}: A Hybrid {CPU-GPU} Algorithm for Community Detection},
  booktitle = {2019 IEEE 26th International Conference on High Performance Computing, Data, and Analytics (HiPC)},
  pages     = {2--11},
  year      = {2019},
  doi       = {10.1109/HiPC.2019.00013},
  note      = {Software at \url{https://github.com/marslabiisc/HyDetect}}
}
```

*(Author given name is **Anwesha** Bhowmik. DOI verified via DBLP. Pages 2–11 confirmed via DBLP `conf/hipc/BhowmikV19`.)*

---

## Bhowmick, Vadhiyar, Varun PV (2022) — Multi-node Multi-GPU Louvain

Direct multi-node extension of HyDetect 2019 (same first author, Anwesha Bhowmick, who had moved from IISc to WalMart Labs Bangalore by publication time). Each compute node has *one* CPU and *one* GPU; the algorithm scales out across nodes by partitioning the graph and using a hierarchical merge.

**The algorithm (Section 5, paper-faithful).**

1. **Two-level partitioning.** First across nodes; then within each node, between CPU and GPU using the same Metis-ratio approach as HyDetect 2019.
2. **Independent Louvain on each device.** Same as HyDetect.
3. **Doubtful-vertex identification (per device).** Uses the same affinity formula as HyDetect (relative-commitment ratio of internal-vs-external degree). Threshold: **0.5 for ≤ 4 nodes, 0.7 for ≥ 8 nodes** (Section 6.1.2). Higher thresholds for larger node counts because more border/ghost vertices arise per partition.
4. **Migration.** Each marked doubtful vertex is moved to the processor with the *maximum number of ghost edges* with that vertex.
5. **Hierarchical merging (Section 5.4).** Processors merge in pairs using ring-based segment exchange (inspired by Rabenseifner's reduce algorithm). Group size = 4. Within a group, partial components are exchanged in segments to ensure no intermediate merged graph exceeds a single node's memory. Group leaders then participate in the next round.
6. **Iterate until** the global coarse graph fits in one node, then do a final Louvain on that node.

**Hardware (Section 6).** Cray XC40 supercomputer at IISc. Each node = **one** Intel Xeon Ivybridge E5-2695 v2 (12 cores, 2.4 GHz, 64 GB RAM) + **one** NVIDIA Tesla K40 (2880 cores, 12 GB). Up to **24 nodes** (288 CPU cores, 24 GPUs) used in experiments. (**Not** 64 GPUs / 16-node × 4-GPU as I previously claimed — those numbers were fabricated.)

**Test graphs (Table 1).** Twitter (21 M / 265 M), uk-2002 (18.5 M / 523 M), rMat24 (16.8 M / 536 M), eu-2015 (11.2 M / 759 M), gsh (30.8 M / 1.16 B), uk-2005 (39.4 M / 1.84 B), Synthetic (49.79 M / 3.2 B), Orkut (3 M / 110 M). **Largest is 3.2 B edges** — not "trillion-edge" as I earlier claimed. All chosen specifically because they exceed the K40's 12 GB GPU memory.

**Headline results (verified):**
- *Modularity:* "comparable" to single-node state-of-the-art. uk-2005 and uk-2002 show ~2 % degradation; other graphs are equal or higher.
- *Scaling:* up to **7–8× performance improvement over 2-node** executions on the largest graphs (uk-2005, Synthetic). Beyond a per-graph optimum the gains diminish (e.g. eu-2015's per-partition substructure becomes weaker at high node counts, requiring more migrations to converge).
- *Speedup over single-node state-of-the-art:* **up to 47×** (paper's headline).
- *Improvement over multi-node state-of-the-art:* up to **87 %** (paper's headline).
- *vs. HyDetect single-node:* 2–47× on the same graphs.

**For the BTP.** This is the multi-node/multi-GPU reference point. The BTP is single-GPU; Bhowmick 2022 is the "scale-out" alternative the BTP does *not* address. Cite when bounding scope. Note that the algorithmic core (independent-Louvain-then-migrate-doubtful-then-merge) is structurally orthogonal to the BTP's frontier/delta-screening dynamic approach; combining the two is open.

Citation key: `bhowmick2022scalable`.

```bibtex
@article{bhowmick2022scalable,
  author  = {Bhowmick, Anwesha and Vadhiyar, Sathish and Varun, P. V.},
  title   = {Scalable multi-node multi-{GPU} {Louvain} community detection algorithm for heterogeneous architectures},
  journal = {Concurrency and Computation: Practice and Experience},
  volume  = {34},
  number  = {17},
  pages   = {e6987},
  year    = {2022},
  doi     = {10.1002/cpe.6987}
}
```

*(Author given name corrected to **Anwesha** Bhowmick (same person as in HyDetect 2019, where the surname is spelled "Bhowmik"). The "Karmakar" middle name from my earlier BibTeX was fabricated and has been removed.)*

---

## Mohammadi, Fazlali, Hosseinzadeh (2021) — ACLM

**Adaptive CUDA Louvain Method**. The "adaptive" refers to **per-launch GPU thread / block / SM allocation based on per-vertex neighbour count**, *not* to active-vertex-set shrinking. (My earlier description framing ACLM as a frontier / active-list method was wrong — there is no `moved` flag, no active list, no host round-trip per iteration.)

**The actual algorithm (Section 4, paper-faithful).** When evaluating moving a vertex $v$ into a candidate community, ACLM parallelises the per-community-edge-weight Sigma calculation across GPU SMs and threads. The launch parameters are computed at runtime per vertex (Algorithm 2 in the paper):

1. **Required SMs** = $\lceil \text{degree}(v) / 192 \rceil$ (192 = hardware cores per SM on Tesla K20Xm).
2. If required SMs $> 14$ (the K20Xm's max), use all 14.
3. **Required threads per SM** = degree / actual_SMs.
4. **Threads per block** chosen from the 5-tier ladder $\{32, 64, 128, 256, 512, 1024\}$ based on which range the per-SM thread count falls into.
5. **# blocks** = degree / threads-per-block.

Modularity is computed by distributing the $\Delta Q$ Sigma terms across SMs (Algorithm 3): each SM computes a partial sum, the sums are added at the end and divided by total edge weight $m$. Aggregation is on the CPU; only the modularity-Sigma calculation runs on the GPU.

**What ACLM does NOT do** (corrections to my earlier text):
- ❌ No active-vertex frontier or `moved` flag.
- ❌ No shared-memory hash table with 1500-slot community accumulator. The paper does mention shared memory per block but does not describe a hash-table organization or specific slot counts.
- ❌ No host round-trip per local-moving iteration to rebuild an active list.
- ❌ Not a precursor to delta-screening or DF-Louvain.

**Hardware (Tables 1, 2).** Two test platforms:
- *Platform 1:* AMD Opteron 6344 + NVIDIA Tesla K20Xm (14 SMs × 192 cores = 2688 cores), 128 GB RAM, CentOS 6.3, CUDA 6.5.
- *Platform 2:* Intel Xeon E5-2680 v4 @ 2.40 GHz + NVIDIA GeForce GTX 1080 Ti, 64 GB RAM, Ubuntu 18.04, CUDA 10.0.

**Test graphs (Table 3) — four graphs, NOT SNAP:**

| Benchmark | # nodes | # links |
|---|---|---|
| CNR-2000 | 325 557 | 2 738 969 |
| EU-2005 | 862 664 | 16 138 468 |
| IN-2004 | 1 382 908 | 13 591 473 |
| Cage-15 | 5 154 859 | 47 022 346 |

**Comparison baselines.** PLM (Staudt-Meyerhenke 2016, multicore CPU), APLM (Fazlali 2017, adaptive multicore CPU), NAIM (Naim et al. 2017, GPU).

**Headline runtime results (Table 6, Platform 1) — paper-faithful.**

| Benchmark | PLM | APLM | NAIM | ACLM |
|---|---|---|---|---|
| CNR-2000 | 2.74 s | 0.15 s | 0.40 s | **0.10 s** |
| EU-2005 | 3.69 s | 0.49 s | 1.32 s | **0.31 s** |
| IN-2004 | 3.71 s | 0.60 s | 0.95 s | **0.47 s** |
| Cage-15 | 36.88 s | 10.17 s | 13.10 s | **8.32 s** |

ACLM's **reduction in execution time vs. PLM:** 96% (CNR), 91% (EU), 87% (IN), 77% (Cage-15). **vs. APLM:** 33%, 37%, 22%, 18%. **vs. NAIM:** 75%, 77%, 51%, 36%.

**Memory observation.** Platform 2 (GTX 1080 Ti, only 11 GB): NAIM fails to run Cage-15 due to memory exhaustion; ACLM completes. The paper attributes this to NAIM's "less efficient" memory usage.

**Quality (Table 8, Platform 1).** All four algorithms produce essentially the same modularity per benchmark — "comparable" within ~0.001 absolute on CNR-2000, EU-2005, IN-2004. On Cage-15 PLM (0.894) is higher than the others (~0.85–0.87). Quality is **not** a differentiator — speedup is.

**For the BTP.** ACLM is the canonical citation for **runtime-adaptive GPU launch parameters** (vary block count and threads-per-block based on per-vertex degree). The BTP's GPU kernels do *not* currently make this dynamic decision — they fix block sizes at kernel-launch time. ACLM's pattern is portable to the BTP if profiling shows launch-parameter-mismatch as a bottleneck on highly degree-skewed graphs. ACLM is **not** a useful reference for frontier or delta-screening, despite my earlier framing.

Citation key: `mohammadi2021accelerating`.

```bibtex
@article{mohammadi2021accelerating,
  author  = {Mohammadi, Maryam and Fazlali, Mahmood and Hosseinzadeh, Mehdi},
  title   = {Accelerating {Louvain} community detection algorithm on graphic processing unit},
  journal = {The Journal of Supercomputing},
  volume  = {77},
  number  = {6},
  pages   = {6056--6077},
  year    = {2021},
  doi     = {10.1007/s11227-020-03510-9},
  publisher = {Springer}
}
```

*(First-author given name corrected: **Maryam** Mohammadi, not "Mahmood Mohammadi". Mahmood is the second author Fazlali — same as on the 2017 APLM paper. Year corrected from 2020 → 2021 (online Nov 2020, published in 2021 issue).)*

---

## Chou, Ghosh (2022) — Nido: Batched Graph Community Detection on GPUs

> ✅ **Now verified from the PDF** (PACT '22, pp. 172–184). Replaces the earlier no-PDF placeholder. Key facts: the parallel-update heuristic is **batched vertex update** (process vertices in consecutive synchronised batches, *not* graph colouring), and per-community weight is accumulated by **sort + `reduce_by_key`** (segmented reduction), not a hash table. (First author **Han-Yi Chou**, NVIDIA / work done at UIUC; **Sayan Ghosh**, PNNL.)

### Introduction
Modularity Louvain is **inherently sequential**; naïve parallel local-moving lets adjacent vertices update concurrently against **stale** community info, which admits negative-gain moves and local maxima. The standard fix is **graph colouring** (distance-$k$) so same-colour, non-adjacent vertices move in parallel — but colouring is costly (optimal colouring is NP-complete, needs vertex reordering, and a sync per colour set). Nido proposes a **simpler heuristic, "batched vertex update"**: split the vertices into user-defined **batches** and process them **one batch after another (bulk-synchronous)**, so a later batch sees the earlier batches' updated communities. It is a **multi-GPU** CUDA/OpenMP implementation that, through partitioning, can cluster graphs **larger than the combined GPU memory** of a node.

### Novelty / Contributions
1. **Batched vertex update** — a colouring-free way to sequentialise adjacent updates: batches run in parallel *within* a GPU's partition and *across* GPUs, with a host–device sync at the **end of every batch**. #batches is a knob trading speed vs quality.
2. **Multi-GPU via UVA** — CUDA **Unified Virtual Addressing** (peer access) lets a vertex read/migrate to a community held in *another* device's memory directly; **binary search** finds the owning device.
3. **Adaptive partitioning (out-of-core)** — partition the graph by the **maximum #edges that fit on the device** (`cudaMemGetInfo`) and stream partitions, so graphs bigger than one GPU's memory can be processed.
4. Extensive **convergence/scalability** study and quality/perf comparison vs Grappolo (CPU) and cuGraph (GPU) on DGX-2 (V100) and DGX-A100.

### Algorithm
- Graph in CSR, **edge-based partitioning** for load balance; OpenMP threads pinned per NUMA/GPU.
- **Batched Louvain (Alg 1):** outer loop = phases; inner loop = iterations until $\Delta Q<\tau$ (default $10^{-6}$). Each iteration, for each GPU (via OpenMP): copy a partition to the device, then **for each batch $b$: for each vertex $v$ in the batch, $C[v]\leftarrow\arg\max_c \Delta Q$**, then **host–device sync** before the next batch. After convergence, **graph compaction** (CPU/OpenMP) builds the coarsened CSR.
- **Per-community weight $e_{i\to c}$:** Nido stores a vertex's incident edges **sorted by the neighbour's community id**, then runs a **segmented reduction (`thrust::reduce_by_key`)** over the sorted edge weights so equal communities sum together; the best $\Delta Q$ is then a simple scan. The **sort is per-batch** (only that batch's edges), not the whole graph.
- **Thread mapping:** cooperative groups (size 16/32) — a thread block is split into cooperative thread groups, each group handles one vertex, and the vertex's incident edges are dealt to the group's threads by `edge_index mod group_size`; warp-collective `shuffle` does the reductions.
- **Partitions vs batches:** *batches* control quality/sequentialisation (sync between them); *partitions* control memory (a partition = the largest edge-block that fits the GPU; a batch larger than a partition is processed partition-by-partition). 2–4 CUDA streams overlap partitions.

### More interesting points
- **The sort dominates runtime** — profiling (Fig 7) shows **50–80% of GPU time is the sort** (community-ordering the edges); the actual modularity compute is ~10–20%. On small graphs (com-orkut) memory allocation/transfer dominates (up to **97% on 8 GPUs** — little useful work, no scaling).
- **Batching is sometimes required for *correctness*, not just quality:** **mycielskian20** has no triangles, so a single batch makes colour-free parallel Louvain **exit at the very first iteration** (no productive move) — they must use 4 batches. So batching isn't only a quality knob; it lets parallel Louvain make any progress at all on some graphs.
- **Quality vs #batches is graph-dependent:** for low-modularity graphs ($\le0.7$) batches barely matter (Fig 8); for high-modularity graphs (→1) more batches → more phases but **40–60% better quality** and **20–45% fewer iterations in the expensive first phase** (Figs 9–10). #batches ranges 2–2048.
- **cuGraph caveats** (same as cuVite reports): cuGraph uses the **min-label** heuristic and a looser **$10^{-3}$** threshold (Nido uses $10^{-6}$), and runs **out-of-memory** on large graphs even multi-GPU (Dask MNMG); Nido is **1.5–30× faster** than cuGraph(MNMG).

### Baselines (graphs and baseline algorithms)
**Baseline algorithms:** **Grappolo** (Lu et al. — optimised shared-memory CPU Louvain with distance-1 colouring, 128 OpenMP threads on Intel CLX; the "best CPU" baseline) and **cuGraph** (NVIDIA RAPIDS, single-GPU SG + multi-GPU MNMG via Dask). **Platforms:** DGX-2 (16× V100/32 GB), DGX-A100 (8× A100/40 GB), IBM Power9 + 4× V100.

**Datasets (Table 2; ordered by #edges):**
| Graph | \|V\| | \|E\| |
|---|---|---|
| uk-2007-05 | 105.9M | 6.6B |
| sk-2005 | 50.6M | 3.86B |
| com-friendster | 65.6M | 3.61B |
| twitter7 | 41.7M | 2.93B |
| it-2004 | 41.3M | 2.27B |
| mycielskian20 | 786K | 1.35B |
| webbase-2001 | 118.1M | 1.98B |
| com-orkut | 3.07M | 234.4M |

(plus quality graphs: Graph500-scale21/22/23, MAWI-1..4, Graph-Challenge "hihi" hard cases, sx-stackoverflow, mycielskian17/18.)

**Headline results:** speedups **2–14× (16× V100)** / **1.3–7.5× (8× A100)**; max **50× / 30×** over single-batch Grappolo / GPU-Louvain baselines. Per Table 2, over a single GPU: **1.5–13.75×**; over 128-thread Grappolo: **0.27×–53.43×** (the CPU still wins on uk-2007, sk-2005, webbase — large web graphs favour Grappolo). Quality (Table 3): matches or beats Grappolo on **>11/16** graphs and beats cuGraph/Grappolo on **12/16**; better F-score than cuGraph on the Graph-Challenge "hard" cases (Table 4).

### Takeaways (for the BTP)
- **The cleanest contrast to your `df_louvain`.** Both attack the *same* problem — parallel Louvain's negative-gain / 2-swap from stale community info — but differently: **Nido uses coarse-grained batching + sync** (no locks, no colouring), whereas **df_louvain uses fine-grained verified-commit-under-lock**. State it plainly: Nido sequentialises *groups of vertices in time*; df_louvain serialises *only the conflicting move* via the community-pair lock.
- **Per-community accumulation = the "sort" branch of the design space.** Nido = **sort + `reduce_by_key`**; cuVite = list-dedup / dense-array; df_louvain = **per-vertex hash**. A tidy three-way comparison for the report — and note Nido's sort costs **50–80% of GPU time**, which the hash avoids.
- **Triangle-free correctness note:** mycielskian20 needs batches just to make a move; your verified-commit kernel doesn't have that failure mode (it can still commit a positive move) — a small point in favour of the locking approach.
- **Out-of-core via partitioning** is a scaling capability the BTP (single-GPU, in-core) lacks; Nido and cuVite both stream/spread to exceed device memory.
- Same author (Ghosh) as cuVite, but distinct: **cuVite = distributed multi-node**; **Nido = single-node multi-GPU + batching + out-of-core**.

Citation key: `chou2022nido`.

```bibtex
@inproceedings{chou2022nido,
  author    = {Chou, Han-Yi and Ghosh, Sayan},
  title     = {Batched Graph Community Detection on {GPU}s},
  booktitle = {Proceedings of the International Conference on Parallel Architectures and Compilation Techniques (PACT '22)},
  pages     = {172--184},
  year      = {2022},
  doi       = {10.1145/3559009.3569655},
  publisher = {ACM},
  address   = {Chicago, IL, USA},
  note      = {Software: \url{https://github.com/sg0/nido}}
}
```

*(Author order: **Chou** (Han-Yi) is first author, **Ghosh** (Sayan) is second — corrected from my earlier "Ghosh, Sayan and Chou, Chun-Yen" which had the wrong order and wrong given name for Chou.)*

---

## Gawande, Ghosh, Halappanavar, Tumeo, Kalyanaraman (2022) — cuVite

A **distributed-memory multi-GPU** implementation of the Louvain method named **cuVite**, built on top of the same group's earlier CPU-only distributed Louvain (Vite). One MPI process per GPU; graph partitioned across processes; modularity-optimisation phase runs on GPU, graph-reconstruction (coarsening) phase runs on CPU. Open-access via Elsevier (CC BY-NC-ND 4.0).

**The split between GPU and CPU work (Sections 4 and 3.3).** Per the paper's own justification:
- **GPU:** modularity optimisation (Algorithm 2 in the paper) — the compute-intensive part with regular per-vertex work.
- **CPU:** graph reconstruction / coarsening (`BuildNextPhaseGraph`) — described in the paper as "involves irregular accesses to memory and can be performed efficiently only on CPUs".

So aggregation on CPU is *not* (as I previously implied) a defensible-but-outdated 2013 choice — it is a deliberate per-phase decision the paper explicitly defends.

**The cooperative-groups (CG) load-balancing scheme (Section 4.3).** The most distinctive engineering contribution. Vertices are processed by **CUDA Cooperative Groups** whose size is chosen *based on the vertex's degree*:

- *High-degree vertex (degree above a CG-tile-size threshold):* an entire cooperative group is assigned to one vertex; group-internal synchronisation handles the per-community-weight reduction.
- *Warp-sized CG:* "When the size of a CG equals the size of a warp, we observe the highest performance."
- *Very-low-degree vertex:* one thread per vertex; no CG synchronisation needed.

Two CUDA streams per GPU separate high-degree from low-degree vertex processing for load-balancing.

**What the paper does *not* specify** (and I therefore should not invent): exact degree thresholds for the CG-size buckets. My earlier claim of "4-thread for degree < 8, 32-thread for 8–256, 128-thread for > 256" was fabricated. The paper says only that the size is "decided on the base of the degree" with a "predefined value (the tile size for a CG)" as the high/low cutoff.

**Hardware (Section 2.1) — three platforms used.**
- **OLCF Summit** (the headline platform): each node = 2 IBM Power9 22-core CPUs + 6 NVIDIA Tesla V100 GPUs (16 GB HBM2 each), with the 6 GPUs split into two NVLINK2-interconnected blocks of 3. Up to **16 nodes** used.
- **NVIDIA DGX-2:** 2 × 24-core Intel Xeon Platinum 8168 CPUs + 16 V100 GPUs (32 GB HBM2 each) fully interconnected via 12 NVSwitches.
- **ALCF Theta:** Intel Xeon Phi Knights Landing (KNL) — used as a CPU-only distributed-memory baseline (no GPU).

**Headline results (Section 1, contribution list, paper-verified):**
- *vs. NVIDIA RAPIDS cuGraph on a single GPU:* "**up to 20× improvement**" (Section 1 (iii)) — abstract gives the more specific "up to 19× on a single NVIDIA V100 from DGX-2".
- *Strong scaling on Summit:* **1.6–3.2× over 2–16 nodes** (Section 1 (iii)). My earlier "50× over sequential on 8 GPUs" and "30% per-GPU efficiency" were fabricated.
- *Theta (CPU-only baseline):* up to **6× on 2048 processes** with 8 real-world graphs (Section 1 (iv)).
- *Quality:* "parity of solutions" with serial and CPU-only Louvain (Section 1 (v)).

**Edge-balanced partitioning (Section 5.1).** A novel contribution beyond the GPU-port: instead of the standard 1D vertex-based partition (which leaves edge counts wildly unbalanced for power-law graphs and inflates ghost-vertex counts), cuVite reads the graph partially on one process, computes per-vertex edge counts, and broadcasts a partition that balances *edges* across processes. Reported impact: **up to 80 % improvement in end-to-end execution time** vs. the vertex-balanced default.

**Communication is the dominant cost.** Paper notes that "communication overhead of our implementation can be significant and more than 90% of the overall elapsed time" depending on input graph and process count (Fig 7). Inter-process exchange is via MPI nonblocking Send/Recv plus collectives.

**For the BTP.** Three things:
1. **Cite as the canonical multi-GPU Louvain.** The BTP is single-GPU; cuVite bounds the scope of "what scaling out would buy".
2. **Adopt the CG-by-degree pattern as future work.** The BTP's GPU kernel currently uses fixed launch parameters; cuVite's per-degree CG-size selection is a portable optimisation pattern.
3. **The aggregation-on-host argument is more defensible than I had claimed.** If the BTP keeps aggregation on GPU (as it does), it should explicitly contrast with cuVite's deliberate split, citing irregular-access cost as the reason cuVite chose differently.

Citation key: `gawande2022cuvite`.

```bibtex
@article{gawande2022cuvite,
  author  = {Gawande, Nitin and Ghosh, Sayan and Halappanavar, Mahantesh and Tumeo, Antonino and Kalyanaraman, Ananth},
  title   = {Towards scaling community detection on distributed-memory heterogeneous systems},
  journal = {Parallel Computing},
  volume  = {111},
  pages   = {102898},
  year    = {2022},
  doi     = {10.1016/j.parco.2022.102898},
  publisher = {Elsevier},
  note    = {Open access (CC BY-NC-ND 4.0). Paper presents cuVite, a distributed multi-GPU Louvain library.}
}
```

*(Affiliations: Gawande was at PNNL, now at Intel; Ghosh, Halappanavar, Tumeo at PNNL; Kalyanaraman at WSU.)*

---

## NVIDIA RAPIDS cuGraph — Louvain

> ⚠️ **Verification note.** There is no peer-reviewed paper for the cuGraph Louvain algorithm itself; the implementation is the artifact. I have not directly inspected the cuGraph source code for this BTP. **Specific implementation claims that I cannot verify from a primary source have been removed from this entry.**

The de-facto GPU Louvain baseline that any Python user reaches for via `cugraph.louvain(G)`. Part of the NVIDIA RAPIDS suite. UGRC-I used `cugraph` version 25.08.00 on Google Colab as its baseline.

**What's verifiable from public materials:**
- Released under the Apache-2.0 license at [github.com/rapidsai/cugraph](https://github.com/rapidsai/cugraph).
- Python API: `cugraph.louvain(G, max_level, threshold, resolution)`.
- Documented default parameters (per RAPIDS docs at the time of the BTP): `max_level=100`, `threshold=1e-7`, `resolution=1.0`.
- Sahu (2025) GVE-Louvain paper benchmarks cuGraph Louvain on an NVIDIA A100 GPU and reports being 5.8× faster.
- Gawande et al. cuVite (2022) reports up to 19–20× speedup over cuGraph Louvain on a single V100 GPU.

**What I cannot independently verify** (and therefore no longer claim in this entry):
- The specific kernel design (whether it uses cooperative groups, per-vertex hash maps in shared memory, or any specific aggregation primitive sequence)
- Whether Grappolo's min-label tie-breaking is used or omitted
- Whether the implementation has been rewritten "at least twice"
- The exact "5% modularity lower than NetworKit on com-amazon/com-dblp" figure (this would need re-verification against UGRC-I's actual measured numbers)

**For the BTP.** Cite cuGraph as a *required* baseline (every reviewer of a GPU Louvain paper will ask "did you beat cuGraph?"). When reporting comparison numbers, **pin the cuGraph version explicitly** (the public implementation has evolved substantially across RAPIDS releases). Use `@misc` with a URL since there is no paper.

Citation key: `rapidscugraph`.

```bibtex
@misc{rapidscugraph,
  author       = {{NVIDIA RAPIDS Team}},
  title        = {{cuGraph}: {GPU}-accelerated graph analytics library},
  howpublished = {\url{https://github.com/rapidsai/cugraph}},
  year         = {2024},
  note         = {No standalone peer-reviewed publication. Version 25.08.00 used as baseline in UGRC-I (Chandaluru, 2025).}
}
```

---

## Sahu (2025) — CPU vs. GPU for Community Detection (GVE-Louvain vs. ν-Louvain)

A direct head-to-head between Sahu's own CPU implementation (**GVE-Louvain**, from his 2023 paper) and a new GPU implementation he calls **ν-Louvain** (the symbol ν is "for video card"). Both run on the same author's hardware; both use the same algorithmic skeleton; the comparison isolates the effect of CPU vs. GPU architecture on the Louvain workload. Headline conclusion (Abstract): "**ν-Louvain performs only on par with GVE-Louvain**, largely due to reduced workload and parallelism in later algorithmic passes" — i.e. the GPU does not beat the CPU.

**Hardware (Section 5.1.1).** Two servers:
- *CPU server:* dual 16-core Intel Xeon Gold 6226R @ 2.90 GHz, 512 GB RAM, CentOS Stream 8, GCC 8.5 + OpenMP 4.5.
- *GPU server:* NVIDIA A100 (108 SMs, 80 GB, 1935 GB/s) + AMD EPYC-7742 (64 cores, 2.25 GHz), 512 GB DDR4, Ubuntu 20.04, GCC 9.4 + OpenMP 5.0 + CUDA 11.4.

**ν-Louvain's GPU-specific design (Section 4.3, paper-faithful):**

- **Pick-Less (PL) for community-swap mitigation.** GPU lockstep execution makes symmetric same-SM vertices swap communities indefinitely (more pathological than on CPU). PL restricts a vertex's move target to *lower-ID* communities only; applied every $k$ iterations. Tested $k \in \{2, 4, 8, 16\}$; **PL4 wins** (highest modularity, 1.25× faster than PL16). Adopted: PL4.
- **Per-vertex open-addressing hash tables (Figure 6).** Each vertex $v$ gets a hash table of size $2 \deg(v)$ in a contiguous global memory pool; capacity = $\text{nextPow2}(\deg(v)) - 1$. Total memory $O(2|E|)$.
- **Collision resolution.** Tested four schemes: linear probing, quadratic probing, double hashing, and a hybrid quadratic-double. **Quadratic-double wins** (1.05×, 1.32×, 1.12× faster than the other three respectively).
- **32-bit floats for hash values** (rest of computation in 64-bit). Maintains quality, moderate speedup.
- **Two-kernel partition with degree-based switch.** Thread-per-vertex for low-degree, block-per-vertex for high-degree, with **switch degree = 64 for local-moving, 128 for aggregation** (empirically tuned, Figures 9–10).
- **Time complexity $O(K \cdot n)$**, **space complexity $O(n)$** (vs. GVE-Louvain's $O(T \cdot n + m)$, where $T$ is thread count).

**Headline empirical results (paper-faithful).**

*GVE-Louvain (CPU) vs. baselines (Section 5.2.1, Figure 11):*
- Average **50× faster than Vite, 22× faster than Grappolo, 20× faster than NetworKit Louvain, 3.2× faster than cuGraph Louvain** (where cuGraph runs — it OOMs on 5 of the 13 graphs). Note: this 3.2× differs from the 5.8× claim in the abstract (and in my prior text); 3.2× is the per-graph average, 5.8× is the in-bulk number from the abstract; both come from this paper.
- *Modularity:* +3.1 % vs. Vite, −0.6 % vs. Grappolo and NetworKit, −0.7 % vs. cuGraph.
- *Throughput:* 560 million edges/s on sk-2005 (3.80 B edges in 6.8 s).

*ν-Louvain (GPU) vs. baselines (Section 5.2.2, Figure 12):*
- Average 20× over Grappolo, 17× over NetworKit, **61× over Nido**, 5.0× over cuGraph Louvain.
- −1.1 %, −1.2 %, −1.3 % modularity vs. Grappolo, NetworKit, cuGraph respectively.
- *Throughput:* 405 million edges/s on it-2004 (5.1 s).

***ν-Louvain (GPU) vs. GVE-Louvain (CPU) — the head-to-head (Section 5.2.3, Figure 13):***
- **ν-Louvain is on average only 1.03× faster than GVE-Louvain**.
- ν-Louvain has 0.5 % lower modularity than GVE-Louvain.
- ν-Louvain wins more decisively on road networks (low average degree).
- ν-Louvain runs out of memory on sk-2005 (the largest graph); GVE-Louvain does not.

**Sahu's diagnosis of why the GPU doesn't win (Section 5.2.3, paper verbatim):**
- "This lack of performance improvement likely stems from the reduced parallelism in the later passes of the algorithm, where only a (relatively) small number of super-vertices are being processed."
- "While using the GPU for the first pass and CPUs for the remaining passes is an option, the overhead of managing both devices likely negates any potential runtime and energy efficiency gains."
- "The limited memory capacity of GPUs restricts the size of graphs that can be processed."
- Conclusion: "CPU architectures, with their higher clock speeds and greater efficiency for serial or mixed workloads, may be better suited for community detection."

**Phase breakdown of GVE-Louvain (Section 5.3.1).** 49 % runtime in local-moving, 35 % in aggregation, 16 % in initialisation/renumbering/dendrogram lookup. **67 % of total runtime is in the first pass** (the original-graph local-moving). Later passes are smaller graphs with diminishing parallelism — exactly what hurts the GPU.

**My earlier fabrications (now removed).**
- ❌ "PCIe accounting argument" — Sahu does not explicitly make a PCIe-time-accounting argument. The diagnosis is reduced parallelism in later passes, plus GPU memory limits.
- ❌ "extremely dense graphs (avg degree > 1000)" / "very wide low-degree graphs" / "medium-density graphs (10–100 avg degree)" — these regime categorisations were fabricated. The paper's actual graph-class observation is just that ν-Louvain wins on **road networks** (low-degree), not on a density-based taxonomy.
- ❌ "atomics dominate", "long tail of high-degree hub vertices serialises" — fabricated mechanisms.

**For the BTP.** Three things:
1. Cite as the strongest current evidence that **CPU and GPU Louvain are roughly tied** when both are well-engineered. Any "GPU is X× faster" claim by the BTP needs to engage with the 1.03× ν-vs-GVE result.
2. **The 67 % first-pass observation** is directly relevant to BTP design: any optimisation that doesn't accelerate the first pass is at most a 33 % win.
3. **PL4 is portable to the BTP.** The Pick-Less strategy is a single-line change in the BTP's GPU kernel and a candidate alternative to atomic-based community-swap resolution.

Citation key: `sahu2025cpuvsgpu`.

```bibtex
@article{sahu2025cpuvsgpu,
  author  = {Sahu, Subhajit},
  title   = {{CPU} vs. {GPU} for Community Detection: Performance Insights from {GVE-Louvain} and {$\nu$-Louvain}},
  journal = {arXiv preprint arXiv:2501.19004},
  year    = {2025},
  note    = {Software: \url{https://github.com/puzzlef/louvain-communities-cuda} (\(\nu\)-Louvain) and \url{https://github.com/puzzlef/louvain-communities-openmp} (GVE-Louvain).},
  url     = {https://arxiv.org/abs/2501.19004}
}
```
