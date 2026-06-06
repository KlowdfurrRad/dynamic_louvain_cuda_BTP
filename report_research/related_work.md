# Related Work — GPU Louvain Community Detection

This BTP implements GPU-accelerated static + dynamic Louvain (naive, frontier, delta-screening variants; edge-based and node-based kernels). This document surveys the literature the paper has to engage with: what exists, what doesn't, and where this BTP can claim novelty.

**Legend**: `[UGRC reference]` = cited in the UGRC-I report *Dynamic Louvain on GPUs with CUDA* (Chandaluru, 2025), which is the direct predecessor to this BTP.

---

## Opportunities for This BTP (Key Gaps)

1. **No published pure-GPU delta-screening dynamic Louvain.** CPU (Zarayeneh & Kalyanaraman 2021) and OpenMP (Sahu DF Louvain 2024) versions exist; GPU work stops at "static Louvain + partial screening on host". Moving the screening logic itself onto the GPU is the clearest novelty hook.
2. **Edge-based vs node-based kernel tradeoff** is not systematically explored in the GPU Louvain literature. Current BTP data (quality vs crash-robustness) is genuinely new.
3. **Multi-pass aggregation stability on GPU** — the BTP's edge-based kernel crashes nondeterministically on denser graphs during aggregation. If root-caused and fixed, this is a real contribution of independent interest.
4. **Warm-start + selective recomputation on GPU is thin** — most GPU Louvain papers run static only. A systematic evaluation of naive vs frontier vs delta-screening on GPU across batch sizes would fill this.
5. **Quality gap vs NetworkX / GVE-Louvain on large sparse graphs** (com-amazon, com-dblp) — an open question whether this is inherent to parallel edge-greedy Louvain or fixable via better kernel ordering / Leiden-style refinement.
6. **Resolving the UGRC-I "dense-sparse single-community collapse" bug** — the UGRC report notes modularity drops to 0 on dense-sparse inputs due to a suspected synchronization error. Reproducing and fixing this in the BTP is concrete progress.

---

## Static Louvain — CPU

- **Blondel, Guillaume, Lambiotte, Lefebvre (2008)** — "Fast unfolding of communities in large networks", *J. Stat. Mech.* 2008(10):P10008. *The original Louvain paper; mandatory citation.* `[UGRC reference]` `[Included in BTP report]`
- **Brandes, Delling, Gaertler, Görke, Hoefer, Nikoloski, Wagner (2008)** — "On modularity clustering", *IEEE TKDE* 20(2):172–188. *Foundational modularity analysis; used by UGRC for the modularity definition.* `[UGRC reference]` `[Included in BTP report]`
- **Rotta, Noack (2011)** — "Multilevel local search algorithms for modularity clustering", *ACM Journal of Experimental Algorithmics.* `[Included in BTP report]`
- **Lu, Halappanavar, Kalyanaraman (2015)** — "Parallel heuristics for scalable community detection", *Parallel Computing.* *Grappolo — canonical shared-memory parallel Louvain baseline.*  `[Included in BTP report]`
- **Staudt, Meyerhenke (2016)** — "Engineering parallel algorithms for community detection in massive networks", *IEEE TPDS.* *NetworKit's Louvain.* `[Included in BTP report]`
- **Fazlali et al. (2017)** — "Adaptive Parallel Louvain Community Detection on a Multicore Platform", *Microprocessors and Microsystems.* `[Included in BTP report]`
- **Ghosh, Halappanavar, Tumeo, Kalyanaraman, Lu, Chavarría-Miranda, Khan, Gebremedhin (2018)** — "Distributed Louvain Algorithm for Graph Community Detection", *IPDPS 2018*, pp. 885–895, doi:10.1109/IPDPS.2018.00098. *Vite — the canonical **MPI-distributed-memory** parallel Louvain. Two heuristics: threshold cycling (vary tolerance across phases) and probabilistic early termination (decay each vertex's "active" probability). Headline: 7× over single-node Grappolo at 4 K processes; processes uk-2007 (3.3 B edges) in 32 s on 1 K cores. Companion HPEC 2018 paper "Scalable Distributed Memory Community Detection Using Vite" extends evaluation to streaming graphs.* `[Included in BTP report]`
- **Traag, Waltman, van Eck (2019)** — "From Louvain to Leiden: guaranteeing well-connected communities", *Scientific Reports.* *Contains the analysis of Louvain's disconnected-community failure mode — important for evaluation.*  `[Included in BTP report]`
- **Sahu (2023, arXiv 2312.04876)** — "GVE-Louvain: Fast Louvain Algorithm for Community Detection in Shared Memory Setting". *Currently the fastest published CPU Louvain; mandatory baseline for any Louvain paper after 2024.* `[Included in BTP report]`

## Static Louvain — GPU

- **Cheong, Huynh, Goh (2013)** — "Hierarchical parallel algorithm for modularity-based community detection using GPUs", *Euro-Par.* *Early GPU Louvain; three-level hierarchical partitioning.* `[UGRC reference]` `[Included in BTP report]`
- **Naim, Manne, Halappanavar, Tumeo (2017)** — "Community detection on the GPU", *IPDPS.* *Most-cited GPU Louvain baseline; parallelizes per-edge access with degree-aware load balancing.* `[UGRC reference]`
- **Bhowmik, Vadhiyar (HiPC 2019)** — "HyDetect: A hybrid CPU-GPU algorithm for community detection". *Introduces a "doubtful vertices" heuristic; reports 42–73% improvement over CPU state of the art.* `[UGRC reference]`
- **Bhowmick, Vadhiyar, Varun PV** — "Scalable multi-node multi-GPU Louvain community detection algorithm for heterogeneous architectures", *Concurrency and Computation: Practice and Experience* (doi:10.1002/cpe.6987). *Multi-node/multi-device algorithm with GPU-only or hybrid GPU-CPU HyDetect.* `[UGRC reference]`
- **Mohammadi, Fazlali, Hosseinzadeh (2021)** — "Accelerating Louvain community detection algorithm on graphic processing unit", *Journal of Supercomputing* 77(6):6056–6077, doi:10.1007/s11227-020-03510-9. *The "Adaptive CUDA Louvain Method" (ACLM) — dynamic resource management and block-level shared memory.* `[UGRC reference]`
- **Chou, Ghosh (2022)** — "Batched Graph Community Detection on GPUs", *PACT 2022*, pp. 172–184, doi:10.1145/3559009.3569655. *Introduces "Nido" batched vertex update; divides vertices into batches processed bulk-synchronously.* `[UGRC reference]`
- **Gawande, Ghosh, Halappanavar, Tumeo, Kalyanaraman** — "Towards scaling community detection on distributed-memory heterogeneous systems", *Parallel Computing* (S0167819122000060). *cuVite — distributed multi-GPU C++ Louvain library; uses CUDA cooperative groups (one group per vertex) and adjusts launch parameters by vertex degree.* `[UGRC reference]`
- **NVIDIA RAPIDS cuGraph — Louvain** — de-facto GPU-library baseline. UGRC-I used it as baseline on Google Colab (cugraph 25.08.00). No standalone peer-reviewed paper. `[UGRC reference]`
- **Sahu (2025, arXiv 2501.19004)** — "CPU vs. GPU for Community Detection: Performance Insights from GVE-Louvain and ν-Louvain". *Recent head-to-head; argues well-tuned CPU can beat GPU cuGraph for Louvain.*

## Dynamic Louvain — CPU

- **Aynaud, Guillaume (2010)** — "Static community detection algorithms for evolving networks", *WiOpt.* *Naive-dynamic approach: process all vertices after any graph change; one of the earliest warm-start papers.* `[UGRC reference]`
- **Cordasco, Gargano (2011)** — "Community Detection via Semi-Synchronous Label Propagation Algorithms", *arXiv:1103.4550*. *Semi-synchronous LPA (static-graph; included for the colour-class parallelism pattern).*
- **Shang, Liu, Xie, Chen, Miao, Fang, Wu (2012)** — "A Real-Time Detecting Algorithm for Tracking Community Structure of Dynamic Networks", *SNA-KDD '12 Workshop*, arXiv:1407.2683. *Per-edge incremental Louvain with four edge-type taxonomy.*
- **Cordeiro, Sarmento, Gama (2016)** — "Dynamic community detection in evolving networks using locality modularity optimization", *Social Network Analysis and Mining* 6(1):15, doi:10.1007/s13278-016-0325-1. *Locality-based modularity optimization for dynamic networks.* `[UGRC reference]`
- **Seifikar, Farzi, Barati (2020)** — "C-Blondel: An Efficient Louvain-Based Dynamic Community Detection Algorithm", *IEEE Transactions on Computational Social Systems* 7(2):308–318, doi:10.1109/TCSS.2020.2964197. *Key ideas: "destructive nodes" and community blow-up when a destructive node is removed.* `[UGRC reference]`
- **Zarayeneh, Kalyanaraman (2019)** — "A Fast and Efficient Incremental Approach toward Dynamic Community Detection", *arXiv:1904.08553*. *Earlier Delta-screening paper (insertions only); sequential.* `[UGRC reference]`
- **Zarayeneh, Kalyanaraman (2021)** — "Delta-Screening: A Fast and Efficient Technique to Update Communities in Dynamic Graphs", *IEEE Transactions on Network Science and Engineering* 8(2):1614–1629, doi:10.1109/TNSE.2021.3067665. *Extended journal version of the delta-screening technique — the PDF in the BTP repo root. Adds edge-deletion support.*
- **Sahu (2024, arXiv 2404.19634)** — "DF Louvain: Fast Incrementally Expanding Approach for Community Detection on Dynamic Graphs". *Dynamic Frontier (DF) approach; addresses overestimates in naive-dynamic and delta-screening approaches. The `puzzlef/louvain-communities-openmp-dynamic` repo in this BTP is its reference OpenMP implementation.* `[UGRC reference]`
- **Halappanavar, Lu, Kalyanaraman, Tumeo (2017)** — "Scalable Static and Dynamic Community Detection Using Grappolo", *IEEE HPEC 2017*, pp. 1–6, doi:10.1109/HPEC.2017.8091047. *Direct dynamic extension of the 2015 Grappolo paper. Adds two heuristics ("data caching" and "threshold scaling") on top of static Grappolo, plus two dynamic schemes — *unseeded* (re-run Grappolo per snapshot) and *seeded* (warm-start with previous-snapshot community labels, i.e. naive-dynamic à la Aynaud–Guillaume but parallel). The earliest peer-reviewed parallel dynamic Louvain; pre-dates Zarayeneh delta-screening and Sahu DF Louvain.*
- *(Earlier versions of "Halappanavar Vite streaming Louvain" placeholder retired — that work is the Ghosh et al. cuVite line, already covered under Static Louvain — GPU.)*

## Dynamic Louvain — GPU

**State of the field:** no peer-reviewed paper to date presents a GPU-native delta-screening dynamic Louvain. This scarcity is precisely the niche this BTP targets.

- **Muthavarapu (BTech, IITM)** — "StarPlat dynamic Louvain code" ([github.com/gajendra-iitm/starplat/tree/main/graphcode/dynamic_louvain_community_detection](https://github.com/gajendra-iitm/starplat/tree/main/graphcode/dynamic_louvain_community_detection)). *Sequential predecessor BTech project in the StarPlat DSL that this BTP builds on.* `[UGRC reference]`
- **UGRC-I (Chandaluru, 2025)** — *"Dynamic Louvain on GPUs with CUDA"*, IIT Madras. *The predecessor report for this BTP; introduces the edge-based CUDA kernel, cooperative-kernel synchronization, vertex+community two-level locking, and graph-aggregation via thrust sort/reduce_by_key. Notes an unresolved dense-sparse single-community collapse bug.*

Closest adjacent work:
- **"Breaking the Latency Barrier: Real-Time Incremental Community Detection with Live Graph Data on a Unified Graph Database Framework"** — Springer 2025 chapter. Incremental, database-framework angle (not pure GPU Louvain).
- **Gawande et al. cuVite (2022)** — distributed multi-GPU Louvain; does modularity optimization on GPU and aggregation on CPU. Static only. `[UGRC reference]`
- **Mohammadi/Fazlali ACLM (2020)** — partial-GPU: modularity calculation on GPU, rest on CPU. Closest published "GPU dynamic-ish" work. `[UGRC reference]`
- **Chakaravarthy et al.** — distributed-memory dynamic community detection (not GPU-specific; exact citation to be confirmed).

Commentary: the `nx_vs_cuda_comparison.md` results in this project show meaningful quality gaps vs NetworkX on large sparse graphs. Whether that gap is inherent to GPU edge-greedy or fixable by Leiden-style refinement / graph coloring is open.

## Related but Out-of-Bin

### Leiden and alternative algorithms
- **Traag, Waltman, van Eck (2019)** — Leiden algorithm, *Scientific Reports.* *Fixes Louvain's disconnected-community issue.*
- **Sahu (2023, arXiv 2312.13936)** — "GVE-Leiden: Fast Leiden Algorithm for Community Detection in Shared Memory Setting".
- **gLeiden (Bioinformatics Advances 2025)** — GPU Leiden for directed and undirected graphs. *A direct Leiden analogue of what this BTP does for Louvain.*
- **NVIDIA cuGraph — Leiden** — GPU-powered Leiden in the RAPIDS suite; NVIDIA developer blog discusses it.
- **Sahu (2024, arXiv 2405.11658)** — "A Starting Point for Dynamic Community Detection with Leiden Algorithm".
- **arXiv 2502.18497 (2025)** — "A Parallel Hierarchical Approach for Community Detection on Large-scale Dynamic Networks" (authors to be confirmed).
- **Raghavan, Albert, Kumara (2007)** — "Near linear time algorithm to detect community structures in large-scale networks", *Phys. Rev. E.* *Label Propagation Algorithm (LPA) — often compared with Louvain for parallelism.*
- **Rosvall, Bergstrom (2008)** — "Maps of random walks on complex networks reveal community structure", *PNAS.* *Infomap — non-modularity alternative.*
- **Waltman, van Eck (2013)** — "A smart local moving algorithm for large-scale modularity-based community detection", *European Physical Journal B.* *SLM — Louvain variant with different local-moving strategy.*
- **Hébert-Dufresne et al.** — graph-coloring-based parallel-safe moves (used in some GPU Louvain variants; exact citation to verify).

### Tools, libraries, and DSLs
- **Hagberg, Schult, Swart (2008)** — "Exploring Network Structure, Dynamics, and Function using NetworkX", *Proc. 7th Python in Science Conference (SciPy 2008)*, pp. 11–15. *Provides `networkx.community.louvain_communities` (added in later versions of NetworkX) — the CPU baseline wired up via `nx_louvain.py`.* `[UGRC reference]`
- **Bell, Hoberock (2011)** — "Thrust: A Productivity-Oriented Library for CUDA", in *GPU Computing Gems Jade Edition*, ed. Hwu, pp. 359–371, doi:10.1016/B978-0-12-385963-1.00026-5. *Used heavily in this BTP for sort, reduce_by_key, inclusive_scan.* `[UGRC reference]`
- **Behera, Kumar, Rajadurai T, Nitish, Pandian M, Nasre (2024)** — "StarPlat: A Versatile DSL for Graph Analytics", *Journal of Parallel and Distributed Computing* 194:104967, doi:10.1016/j.jpdc.2024.104967 (arXiv:2305.03317, 2023). *The DSL the project was originally framed around.* `[UGRC reference]`
- **Nasre, R.** — GPU programming course materials, [cse.iitm.ac.in/~rupesh/teaching/](https://cse.iitm.ac.in/~rupesh/teaching/). *Source of the GPU-hierarchy figure in UGRC-I; foundational CUDA reference.* `[UGRC reference]`

---

## How to Use This Document

**Mandatory baselines to run** (a reviewer will expect these side-by-side with the BTP's numbers):
- GVE-Louvain (CPU, shared memory) — Sahu 2023.
- cuGraph Louvain (GPU library) — NVIDIA RAPIDS. *(UGRC-I already uses this baseline.)*
- DF Louvain OpenMP — Sahu 2024 (already cloned as `louvain-communities-openmp-dynamic`).
- NetworkX Louvain — CPU sanity baseline (already wired up via `nx_louvain.py`).

**Mandatory Related Work citations** (any Louvain paper in 2026 must cite):
- Blondel et al. 2008 (original Louvain).
- Traag et al. 2019 (Leiden / Louvain limitations).
- Naim et al. 2017 (first well-known GPU Louvain).
- Zarayeneh & Kalyanaraman 2019/2021 (delta-screening, the BTP's central technique).
- Sahu 2024 DF Louvain (current dynamic Louvain state of the art).
- Sahu 2023 GVE-Louvain (current CPU Louvain state of the art).
- UGRC-I Chandaluru 2025 (direct predecessor).

**Good candidates for a comparison table**:
Naim 2017, Mohammadi/Fazlali ACLM 2020, Bhowmick/Vadhiyar (CPE 2021), Ghosh/Chou Nido, Gawande cuVite, cuGraph Louvain, GVE-Louvain (CPU), ν-Louvain (GPU), UGRC-I, the BTP's three dynamic variants.

---

## Verification Notes

- DF Louvain reference confirmed via `louvain-communities-openmp-dynamic/CITATION.bib` (Sahu 2024, arXiv 2404.19634).
- Delta-screening reference confirmed via `BTP/Delta-Screening_A_Fast_and_Efficient_Technique_to_.pdf` matching the Zarayeneh & Kalyanaraman 2021 IEEE TNSE title.
- UGRC-I references extracted from `Report/UGRC_I_Dynamic_Louvain_on_GPUs.pdf` References section (pages 18–19).
- Entries marked with "uncertain" / "to verify" / "(n.d.)" / "exact citation to be confirmed" must have their exact venue/year looked up in Google Scholar before being cited in the actual paper.
- Some UGRC-referenced works may have incomplete metadata (author order, venue, year) in the UGRC bibliography itself; re-verify each before final citation.

## To Add
- In GVE Louvain but not in above : RUNDEMAN
