# Related but Out-of-Bin

Writing material for the "Related but Out-of-Bin" portion of the Related Work section. Two clusters: (1) Leiden and other alternative community-detection algorithms, used to bound the BTP's scope and to interpret modularity gaps; (2) tools, libraries, and DSLs that the BTP either uses directly or whose existence shaped the project's design choices.

---

## 1. Leiden and Alternative Algorithms

### Traag, Waltman, van Eck (2019) — Leiden

Already cited in `static_louvain_cpu.md` for the disconnected-community analysis. In this section, Leiden is cited as the *algorithmic alternative* the BTP explicitly does not implement. The reason matters: Leiden's correctness-restoring refinement phase introduces an extra synchronisation barrier that is harder to parallelise on the GPU than the basic Louvain local-moving phase, which is partly why GPU Leiden lags GPU Louvain in the literature.

**The parallelism penalty Leiden pays.** Each Leiden iteration is three sub-phases (local move, refinement, aggregation), each with its own global synchronisation. Louvain has two (local move, aggregation). The third sub-phase — refinement — requires reading the current Louvain community assignment and writing a refined sub-community assignment, which can race against the local-moving phase if the two are pipelined. So Leiden essentially serialises what Louvain can pipeline. On GPU this matters because each global sync is a kernel launch.

**Why the BTP doesn't do Leiden.** Three reasons. (1) Scope — the BTP's contribution is the GPU dynamic case, and adding Leiden's refinement on top would multiply the design space. (2) The Sahu DF Louvain reference implementation is Louvain, not Leiden, so a Louvain-vs-Louvain comparison is the apples-to-apples baseline. (3) gLeiden already exists for the static case; if Leiden is the right answer for the dynamic case, it is its own paper.

(Citation key reused: `traag2019louvain`.)

---

### Sahu (2023) — GVE-Leiden

Sahu's Leiden counterpart to GVE-Louvain. Subhajit Sahu, IIIT Hyderabad. arXiv 2312.13936v8 (6 Jul 2024). Software: [github.com/puzzlef/leiden-communities-openmp](https://github.com/puzzlef/leiden-communities-openmp).

**Headline speedups (Abstract, paper-faithful, Table 1):**
- vs. **Original Leiden** (sequential, libleidenalg by Traag et al.): 436×
- vs. **igraph Leiden** (sequential): 104×
- vs. **NetworKit Leiden** (parallel): 8.2×
- vs. **cuGraph Leiden** (parallel, NVIDIA A100 GPU): 3.0×
- vs. **ParLeiden-S** (parallel): 18×
- vs. **ParLeiden-D** (multi-node, 8 nodes): 22×
- 1.6× per thread doubling
- Throughput: 403 million edges/s on a 3.8 B edge graph

**Hardware:** dual 16-core Intel Xeon Gold 6226R (CPU experiments); NVIDIA A100 (for cuGraph Leiden baseline only).

**Optimisations (Section 1, paper-faithful):**
- Parallel prefix sums
- Preallocated CSR data structures (community vertex identification + super-vertex graph storage)
- Fast **collision-free per-thread hash tables** (the Far-KV pattern from GVE-Louvain — *not* "lock-free" as I previously claimed)
- Prevention of unnecessary aggregations
- **Greedy refinement** where vertices optimise for delta-modularity within their community bounds (paper-novelty over the randomised refinement in original Leiden)
- Asynchronous computation
- OpenMP dynamic loop schedule
- Threshold scaling
- Vertex pruning

**My fabrications removed:**
- ❌ "Same engineering ideas (vertex pruning, lock-free hash table)" — actually collision-free *per-thread* hash tables (Far-KV), not lock-free
- ❌ "Modularity 1–4% higher than GVE-Louvain on the SNAP / LAW benchmarks" — specific number range fabricated (paper does report quality comparisons in its experiments section, but I have not verified the exact 1–4% claim and should not assert it)
- ❌ "Dynamic work-stealing across communities to handle community-size skew" — fabricated
- ❌ "GVE-Leiden is *slower* than GVE-Louvain by ~30%" — fabricated; not verified against the paper

**For the BTP.** Cite when discussing the Leiden alternative to Louvain. The greedy-refinement-instead-of-randomised choice is an interesting design point for any future BTP work that ports Leiden to GPU.

Citation key: `sahu2023gveleiden`.

```bibtex
@article{sahu2023gveleiden,
  author  = {Sahu, Subhajit},
  title   = {{GVE-Leiden}: Fast {Leiden} Algorithm for Community Detection in Shared Memory Setting},
  journal = {arXiv preprint arXiv:2312.13936},
  year    = {2023},
  note    = {Latest version v8 (6 Jul 2024). Software: \url{https://github.com/puzzlef/leiden-communities-openmp}.},
  url     = {https://arxiv.org/abs/2312.13936}
}
```

---

### gLeiden (Bioinformatics Advances 2025/2026)

> ⚠️ **Verification note.** I do not have a copy of this paper. The OUP direct-PDF link returns 403, and the PMC mirror redirects to an SPA. **All my prior specific algorithmic claims have been removed as fabricated.**

I previously claimed this paper:
- Generalises Louvain's $\Delta Q$ formula to use both directional in/out degree sums
- The local-moving kernel is "essentially Naim 2017 with directional-degree distinction"
- Refinement assigns each block to a community and runs sub-Louvain in shared memory
- Falls back to per-warp variant for large communities
- Reports 5–10× GPU vs CPU speedup on directed benchmarks, 2–3× on undirected

**None of these claims have a verified source.** All were fabricated.

**What's tentatively verifiable from search results:** Title is "gLeiden: accelerated community detection algorithms using directed and undirected graphs on GPUs" published in *Bioinformatics Advances* (Oxford); the abstract reportedly mentions: "lightweight CUDA C++ based GPU implementation of the Leiden algorithm and the very first GPU implementation that supports directed graphs"; "11x and 12x speedup on very large datasets" for directed; "up to 42x speedup on large datasets" for undirected vs the original Java version. Source code at `github.com/Beenishgul/Leiden`. Authors and exact DOI not yet verified.

**For the BTP.** Cite as the GPU Leiden of record. *Action item: locate the paper PDF (try the OUP page directly, or contact the corresponding author) before final submission and revise this entry with paper-faithful content.*

Citation key: `gleiden2025`.

```bibtex
@article{gleiden2025,
  author  = {(authors to be verified)},
  title   = {{gLeiden}: accelerated community detection algorithms using directed and undirected graphs on {GPU}s},
  journal = {Bioinformatics Advances},
  year    = {2025},
  note    = {Verify exact authors, volume, issue, page, DOI before citing. Software: \url{https://github.com/Beenishgul/Leiden}.}
}
```

---

### NVIDIA cuGraph — Leiden

> ⚠️ **Verification note.** No peer-reviewed paper exists. I have not directly verified the cuGraph Leiden API or the implementation details. **Specific algorithmic and benchmark claims that I cannot back from a primary source have been removed.**

The Leiden counterpart to cuGraph Louvain, part of the NVIDIA RAPIDS suite at [github.com/rapidsai/cugraph](https://github.com/rapidsai/cugraph). Cite alongside `rapidscugraph` (in `static_louvain_gpu.md`) when discussing GPU Leiden options.

**What I cannot verify** (and therefore no longer claim):
- Exact API signature (`cugraph.leiden(G)`)
- Whether the implementation reuses the cuGraph Louvain kernel infrastructure plus a refinement kernel
- That "NVIDIA's blog posts emphasise that on directed gene-network benchmarks, cuGraph Leiden is competitive with gLeiden's reported numbers"

**For the BTP.** Cite as a practitioner-default GPU Leiden. *Action item: verify the cuGraph Leiden API signature, version availability, and any published benchmarks before citing in the BTP report.*

Citation key: `rapidscugraph_leiden`.

```bibtex
@misc{rapidscugraph_leiden,
  author       = {{NVIDIA RAPIDS Team}},
  title        = {{cuGraph}: {GPU}-accelerated graph analytics library (Leiden module)},
  howpublished = {\url{https://github.com/rapidsai/cugraph}},
  note         = {No standalone peer-reviewed publication. Verify API signature and version availability before citing.}
}
```

---

### Sahu (2024) — A Starting Point for Dynamic Leiden

Sahu extends the **three** dynamic-Louvain approaches (Naive-Dynamic, Delta-Screening, and his own Dynamic Frontier) to the Leiden algorithm. Built on top of GVE-Leiden. arXiv 2405.11658v4 (27 Dec 2024). Software: [github.com/puzzlef/leiden-communities-openmp-dynamic](https://github.com/puzzlef/leiden-communities-openmp-dynamic).

**Key insight (Section 1.1, paper-faithful).** Directly applying DF approach to Leiden *fails*. After a small batch update, Leiden's local-moving phase converges quickly (because few vertices are affected), and the refinement phase then divides communities into smaller sub-communities. If the algorithm terminates here (as DF Louvain would), modularity is poor because sub-communities require hierarchical aggregation. So Leiden needs *multiple passes* even after early local-moving convergence. Sahu's solution: **selective refinement** — track which vertices migrate between communities during the local-moving phase, mark both source and target communities for refinement, and *do not refine* the untouched communities. This both ensures correctness and reduces processing cost.

**Hardware:** 64-core AMD EPYC-7742 (same as Sahu's DF Louvain paper).

**Headline speedups (Abstract, paper-faithful):**
- ND Leiden: **1.37×** over Static Leiden
- DS Leiden: **1.47×** over Static Leiden
- DF Leiden: **1.98×** over Static Leiden
- 1.6× per doubling of threads

(Note: speedups are *much smaller* than DF Louvain's 179× over Static — because Leiden's refinement makes the dynamic case substantially harder than Louvain's.)

**My fabrications removed:**
- ❌ "Sketch — algorithmic pseudocode plus a small empirical demonstration" — wrong; this IS a fully-engineered implementation with experimental evaluation, not a sketch.
- ❌ "Sahu's sketch handles this by warm-starting only the Louvain-community label and re-running refinement from scratch each snapshot" — wrong. Actual approach is *selective refinement* of only affected communities.
- ❌ "Refinement is fast on a near-converged Louvain partition" — fabricated reasoning.
- ❌ "Follow-up engineered version is open work" — wrong. This IS the engineered version.

**For the BTP.** Cite as the authoritative reference for "what dynamic Leiden looks like" — the **selective-refinement** mechanism is the key adaptation that makes Leiden work in the dynamic setting. If the BTP ever ports its GPU dynamic kernels to Leiden, this is the algorithmic template.

Citation key: `sahu2024dynamicleiden`.

```bibtex
@article{sahu2024dynamicleiden,
  author  = {Sahu, Subhajit},
  title   = {A Starting Point for Dynamic Community Detection with {Leiden} Algorithm},
  journal = {arXiv preprint arXiv:2405.11658},
  year    = {2024},
  note    = {Latest version v4 (27 Dec 2024). Software: \url{https://github.com/puzzlef/leiden-communities-openmp-dynamic}.},
  url     = {https://arxiv.org/abs/2405.11658}
}
```

---

### Bokov, Konovalov, Uporova, Moiseev, Safonov, Radionov (2025) — Parallel Hierarchical Dynamic Leiden

> **Author and content corrections.** My prior entry had "(authors to be confirmed)" — they are now verified. My prior characterization of this as "hierarchical clustering not modularity-based" was *wrong* — the paper is **Leiden-based** and explicitly optimises modularity.

A parallel **Leiden-based** algorithm for dynamic community detection. arXiv 2502.18497v1 (20 Feb 2025), submitted to Elsevier. Authors at Lomonosov Moscow State University, HSE, and Moscow Infocommunication Technology Laboratory.

**Headline contributions (from paper's "Highlights"):**
- A new parallel Leiden-based algorithm for dynamic community detection.
- The algorithm uses only a local neighborhood of the affected nodes.
- A new hierarchical, incrementally-updated graph-based structure for node coalitions.
- Better performance while maintaining the modularity of the partitioning.
- The parallel implementation shows no loss of modularity.

**Optimisations (Section 1):**
- Decoupling of movable nodes
- Preallocated hierarchical graph-based data structure for community identification + supernode graph storage during aggregation
- *Absence* of hash tables for edge storage (an unusual design choice — most parallel Louvain/Leiden uses per-thread hash tables)
- Prevention of unnecessary aggregations

**Notable observation from related work (Section 2 / Section 1):** "The only known parallel algorithm for dynamic community detection that we are aware of is the Dynamic Frontier [26] based on GVE-Leiden [33]" — referring to Sahu's DF-Leiden. So Bokov et al. position themselves as the *second* parallel dynamic-Leiden algorithm.

**My fabrications removed:**
- ❌ "Hierarchical clustering not modularity-based" — wrong, it's Leiden-based and optimises modularity
- ❌ "Lowest common ancestor of u and v in hierarchy, re-cluster only subtree" — fabricated specific mechanism
- ❌ "Logarithmic in graph size for balanced hierarchies" — fabricated complexity
- ❌ "Hierarchical methods don't optimise modularity; produce structurally sensible but worse modularity than Louvain" — wrong; this paper *does* optimise modularity

**For the BTP.** A recent (Feb 2025) parallel dynamic Leiden — alongside Sahu's DF-Leiden, the second known parallel dynamic CD algorithm. Cite as a contemporary alternative to Sahu's work; the "absence of hash tables" choice is a design point worth comparing against the BTP's hash-table-based approach.

Citation key: `bokov2025parallel`.

```bibtex
@article{bokov2025parallel,
  author  = {Bokov, Grigoriy and Konovalov, Aleksandr and Uporova, Anna and Moiseev, Stanislav and Safonov, Ivan and Radionov, Alexander},
  title   = {A Parallel Hierarchical Approach for Community Detection on Large-scale Dynamic Networks},
  journal = {arXiv preprint arXiv:2502.18497},
  year    = {2025},
  url     = {https://arxiv.org/abs/2502.18497},
  note    = {Submitted to Elsevier on 27 February 2025.}
}
```

---

### Raghavan, Albert, Kumara (2007) — Label Propagation Algorithm

The originating LPA paper. Authors verified: Usha Nandini Raghavan & Soundar Kumara (Industrial Engineering, Penn State); Réka Albert (Physics, Penn State).

**What the paper actually does (Section "Algorithm", paper-faithful).** Each vertex initialised with a unique label. At each iteration, every vertex adopts the label most frequent among its neighbours — ties broken uniformly at random. As labels propagate, "densely connected groups of nodes form a consensus on their labels". At the end, vertices sharing the same final label are grouped as communities. The algorithm uses **only the network structure** as a guide — no objective function to optimise.

The paper's claim about cost: "almost linear time" (the abstract uses "almost linear"; near-linearity is empirical, not formally proved with a tight constant in this paper).

**My fabrications removed:**
- ❌ "(1) LPA produces lower modularity than Louvain on most benchmarks (~0.1 NMI gap)" — the specific 0.1-NMI number is fabricated.
- ❌ "Raghavan et al. originally proved LPA converges in time independent of graph size — but the proof assumes synchronous updates" — there is no such formal proof in the paper. The paper observes empirical near-linear scaling.
- ❌ "Asynchronous LPA can occasionally fail to converge (oscillating labels). This is exactly why Cordasco–Gargano's semi-synchronous LPA was introduced" — wrong. Cordasco-Gargano's stated motivation is the *synchronous* LPA's oscillation problem (which Raghavan acknowledged), not the asynchronous variant's.
- ❌ "First community-detection method to get serious GPU implementations" — speculation without verification.

**For the BTP.** Cite when discussing label-propagation-style community detection as an alternative to modularity optimisation. The paper itself does not benchmark against Louvain — that comparison is in later literature.

Citation key: `raghavan2007near`.

```bibtex
@article{raghavan2007near,
  author  = {Raghavan, Usha Nandini and Albert, R{\'e}ka and Kumara, Soundar},
  title   = {Near linear time algorithm to detect community structures in large-scale networks},
  journal = {Physical Review E},
  volume  = {76},
  number  = {3},
  pages   = {036106},
  year    = {2007},
  doi     = {10.1103/PhysRevE.76.036106},
  eprint  = {0709.2938},
  archivePrefix = {arXiv}
}
```

---

### Rosvall, Bergstrom (2008) — Infomap

A non-modularity alternative for community detection. Authors: Martin Rosvall and Carl T. Bergstrom (Department of Biology, University of Washington; Bergstrom also affiliated with Santa Fe Institute). Published in *Proceedings of the National Academy of Sciences* 105(4):1118–1123 (2008). arXiv 0707.0609.

**What the paper actually does (paper-faithful).** Introduces "a new information theoretic approach that reveals community structure in weighted and directed networks". The method "decomposes a network into modules by optimally compressing a description of information flows on the network." A **random walk** on the network is used as a proxy for the information flow. Finding community structure is shown to be equivalent to solving a coding/compression problem. The paper illustrates the method by mapping scientific communication via citation patterns of more than 6 000 journals, and discovers a multicentric organisation.

**My fabrications removed:**
- ❌ "Two-level codebook (global codebook for inter-community jumps, per-community codebooks for intra-community steps)" — this is the map equation's actual structure but I have not verified the paper presents it in this exact way (the abstract talks about "optimally compressing a description of how information flows", and the technical formulation is in the paper body — which I have not fully read here).
- ❌ "Why Louvain is more popular: $O(n \log n)$ Louvain vs $O(m \cdot \text{iterations})$ Infomap" — fabricated complexity claims; not in this paper.
- ❌ "Modularity has to be hacked (asymmetric degree terms) to define Q for directed graphs" — this is generally true in the literature but is not stated by Rosvall in this form.
- ❌ "Why it sidesteps the resolution limit" — speculation about Infomap's resolution behaviour without paper-source.

**For the BTP.** Cite to acknowledge the principal *non-modularity* alternative formalism. The paper does not benchmark against Louvain (Louvain came out the same year, 2008); the often-cited comparison numbers are from later literature.

Citation key: `rosvall2008maps`.

```bibtex
@article{rosvall2008maps,
  author  = {Rosvall, Martin and Bergstrom, Carl T.},
  title   = {Maps of random walks on complex networks reveal community structure},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {105},
  number  = {4},
  pages   = {1118--1123},
  year    = {2008},
  doi     = {10.1073/pnas.0706851105},
  eprint  = {0707.0609},
  archivePrefix = {arXiv}
}
```

---

### Waltman, van Eck (2013) — Smart Local Moving (SLM)

SLM is a Louvain variant that "uses the local moving heuristic in a more sophisticated way" (paper abstract). Authors: Ludo Waltman and Nees Jan van Eck (Centre for Science and Technology Studies, Leiden University — same institution as the later Leiden paper). Published in *European Physical Journal B* 86, article 471 (2013). arXiv 1308.6604.

**What the paper actually claims (Abstract, paper-faithful).** "Our smart local moving algorithm identifies community structures with higher modularity values than other algorithms for large-scale modularity optimization, among which the popular `Louvain algorithm' introduced by Blondel et al. (2008)." Tested on a "diverse set of networks" with up to "tens of millions of nodes and hundreds of millions of edges". On small/medium networks, "identifies community structures with modularity values equally high as, or almost as high as, the highest values reported in the literature, and sometimes even higher."

**My fabrications removed (significant — most of the prior entry was fabricated):**
- ❌ "SLM also considers communities that contain a vertex within distance 2 of the source — i.e., friends-of-friends" — this *specific* description of SLM's mechanism is fabricated. I have not verified the actual algorithm details against the paper body.
- ❌ "1.5–2× more local-moving iterations vs Naim-style Louvain" — wrong/fabricated.
- ❌ "0.5–2% higher modularity than Louvain on benchmark graphs" — fabricated specific.
- ❌ "Leiden came along five years later and dominated SLM" — speculation; Leiden's relationship to SLM should be sourced from the Leiden paper itself, which does cite SLM as a precursor.
- ❌ "Waltman & van Eck observe that SLM's quality advantage over Louvain is largest on graphs where Louvain produces disconnected communities" — fabricated finding; not from this paper.

**Verified by abstract only:** SLM produces equal-or-higher modularity than Louvain in their experiments. The exact mechanism of "smart" needs reading the paper body to characterise faithfully.

**For the BTP.** Cite SLM as a paper-acknowledged precursor to Leiden; both papers come from CWTS Leiden. The exact algorithmic mechanism of SLM should be re-verified before being cited in the BTP report.

Citation key: `waltman2013smart`.

```bibtex
@article{waltman2013smart,
  author  = {Waltman, Ludo and van Eck, Nees Jan},
  title   = {A smart local moving algorithm for large-scale modularity-based community detection},
  journal = {The European Physical Journal B},
  volume  = {86},
  number  = {11},
  pages   = {471},
  year    = {2013},
  doi     = {10.1140/epjb/e2013-40829-0},
  eprint  = {1308.6604},
  archivePrefix = {arXiv}
}
```

---

### Hébert-Dufresne et al. — Graph-Coloring-Based Parallel-Safe Moves

> ⚠️ **Audit note: this entry is a placeholder with no specific verifiable source.** The graph-coloring-for-parallel-safety idea is genuinely a real technique in the parallel-Louvain literature (e.g., Grappolo uses it), but the attribution to "Hébert-Dufresne et al." is unverified. **All my prior specific algorithmic claims have been removed as fabricated.**

I previously claimed:
- A specific colouring algorithm (Jones–Plassmann)
- A specific chromatic-number range "2–10 for sparse graphs"
- That Hébert-Dufresne's group "showed colouring-based approach is guaranteed to converge to the same partition as a specific sequential Louvain ordering"
- That high-degree vertices "end up in their own colour"

**None of these have a verified source.** The graph-coloring-for-parallel-safety idea genuinely appears in Grappolo (Lu et al. 2015) and is well-documented in *that* paper. There may not be a separate "Hébert-Dufresne et al." reference for this specific technique — that attribution should be checked.

**For the BTP.** The graph-coloring pre-pass for parallel safety can be cited via Grappolo (which is the primary source); a separate Hébert-Dufresne citation is unnecessary unless that group's specific contribution can be verified. *Action item: drop this entry, or locate a specific Hébert-Dufresne paper on coloring-based parallel CD before final submission.*

---

## 2. Tools, Libraries, and DSLs

### Hagberg, Schult, Swart (2008) — NetworkX

> ⚠️ **Author surname order:** correctly **Hagberg, Schult, Swart** — my prior section heading "Schult, Chult" is fixed below.

The Python network-analysis library. Authors: Aric A. Hagberg (Los Alamos National Laboratory), Daniel A. Schult (Colgate University), Pieter J. Swart (LANL). Published at the 7th Python in Science Conference (SciPy 2008).

**What the 2008 paper actually covers (paper-faithful, Abstract):**
- Core data structures: simple graphs, directed graphs, graphs with parallel edges and self-loops.
- Nodes can be any hashable Python object; edges can carry arbitrary data.
- Algorithms implemented include shortest paths, betweenness centrality, clustering, and degree distribution.
- Read/write of various graph formats; generators for classic models (Erdős-Rényi, Small World, Barabási-Albert).
- Use case discussed: synchronisation of coupled oscillators.

**My fabrications removed:**
- ❌ "NetworkX provides `networkx.community.louvain_communities`" — true *today*, but Louvain was added to NetworkX much later. The 2008 paper does *not* discuss Louvain.
- ❌ "Direct port of Blondel et al. 2008 in pure Python — no Cython, no NumPy vectorisation in the hot loop" — fabricated implementation details.
- ❌ "Uses Python dicts for the per-community $\Sigma_{tot}$ and $k_{i,in}$, which is slow but correct" — fabricated.
- ❌ "Python dict iteration order, which is insertion order in 3.7+, makes results reproducible across runs" — speculation; Louvain in NetworkX uses random vertex order by default, not deterministic dict iteration.
- ❌ "NetworkX has the canonical implementations of NMI, ARI" — true today but not in the 2008 paper.

**For the BTP.** Cite the 2008 paper for NetworKx as a tool. If the BTP wants to discuss NetworkX's Louvain implementation specifically, that should be sourced from the *current* NetworkX documentation or its codebase, not the 2008 paper.

Citation key: `hagberg2008networkx`.

```bibtex
@inproceedings{hagberg2008networkx,
  author    = {Hagberg, Aric A. and Schult, Daniel A. and Swart, Pieter J.},
  title     = {Exploring Network Structure, Dynamics, and Function using {NetworkX}},
  booktitle = {Proceedings of the 7th Python in Science Conference (SciPy 2008)},
  pages     = {11--15},
  year      = {2008},
  address   = {Pasadena, CA, USA},
  editor    = {Varoquaux, Ga\"{e}l and Vaught, Travis and Millman, Jarrod}
}
```

---

### Bell, Hoberock — Thrust

> ⚠️ **Verification note.** I do not have a copy of the *GPU Computing Gems Jade Edition* book chapter (Elsevier paywall, no preprint). The high-level claims about Thrust as a CUDA template library are common knowledge; specific algorithmic claims about CUB dispatch and `reduce_by_key` semantics that I made are based on Thrust's *current* documentation, not the 2011 chapter.

A C++ template library for CUDA, providing high-level parallel primitives (`sort`, `reduce_by_key`, `inclusive_scan`, `transform`, etc.). Authors: Nathan Bell and Jared Hoberock. Published as a chapter in *GPU Computing Gems Jade Edition* (Morgan Kaufmann, 2011). DOI 10.1016/B978-0-12-385963-1.00026-5.

**What's verifiable from the bibliographic record and Thrust's documentation:**
- Thrust is a C++ template library for parallel algorithms on CUDA-capable GPUs.
- Provides STL-like host/device vectors (`thrust::host_vector`, `thrust::device_vector`).
- Provides parallel primitives: `sort`, `sort_by_key`, `reduce`, `reduce_by_key`, `inclusive_scan`, `exclusive_scan`, `transform`, `for_each`, `copy_if`, `gather`, `scatter`, etc.
- Modern Thrust dispatches to CUB primitives internally.
- The UGRC-I report (Section 5.2.4.3) explicitly uses `thrust::sort`, `thrust::reduce_by_key`, and `thrust::for_each` for graph aggregation.

**My fabrications removed:**
- ❌ Specific claims about `thrust::sort` dispatching to radix sort for integer keys / merge sort for general keys with per-architecture tuning — these reflect Thrust's *current* implementation, not necessarily the 2011 paper's content
- ❌ "BTP's aggregation phase is roughly 50 lines of host-side C++" — speculation about the BTP's own code
- ❌ "BTP's edge-based kernel crash on aggregation could plausibly be a Thrust issue rather than an algorithm issue" — speculation
- ❌ "thrust::reduce_by_key requires sorted-by-key input; common bug to forget this" — true generally but not from the 2011 paper

**For the BTP.** Cite Bell & Hoberock 2011 as the canonical reference for Thrust. Specific implementation details should be cited from Thrust's current documentation (`docs.nvidia.com/cuda/thrust/`) or CUB documentation, not this chapter.

Citation key: `bell2011thrust`.

```bibtex
@incollection{bell2011thrust,
  author    = {Bell, Nathan and Hoberock, Jared},
  title     = {Thrust: A Productivity-Oriented Library for {CUDA}},
  booktitle = {{GPU} Computing Gems Jade Edition},
  editor    = {Hwu, Wen-mei W.},
  publisher = {Morgan Kaufmann},
  pages     = {359--371},
  year      = {2011},
  doi       = {10.1016/B978-0-12-385963-1.00026-5}
}
```

---

### Behera, Kumar, Rajadurai, Nitish, Pandian, Nasre (2023) — StarPlat

A graph-analytics DSL that compiles a single high-level algorithm specification to **three** parallel back-ends. Authors: Nibedita Behera, Ashwina Kumar, Ebenezer Rajadurai T, Sai Nitish, Rajesh Pandian M, Rupesh Nasre (all IIT Madras). arXiv 2305.03317v1 (5 May 2023). The journal version appeared in the *Journal of Parallel and Distributed Computing* in 2024.

**What the paper actually does (Abstract, paper-faithful):**
- Three back-ends: **OpenMP** (multi-core), **MPI** (distributed), **CUDA** (many-core GPU). *Not* four — there is no separate "sequential C" back-end as I previously claimed.
- Central to the compiler is an **intermediate representation** that allows a common representation of the high-level program.
- Demonstrated on **four** algorithms: betweenness centrality, page rank, single-source shortest paths, triangle counting. *Not* including Louvain in the demonstration set.
- Tested on a suite of 10 large graphs.
- Generated code is "competitive to library-based codes in many cases".

**My fabrications removed (significant):**
- ❌ "StarPlat compiles to (sequential C, OpenMP, MPI, CUDA)" — wrong, just three back-ends.
- ❌ "StarPlat exposes graph algorithms as a series of `forall` loops over vertices, edges, and neighbours, with a small set of primitive operations (atomic min/max, fixed-point iteration)" — fabricated specific syntax/primitives.
- ❌ "Dependency analysis works well for static algorithms ... struggles with dynamic ones" — fabricated technical diagnosis.
- ❌ "For dynamic Louvain, the compiler emits CUDA code that compiles but produces wrong results" — UGRC-I report does *not* claim this; this is a fabricated diagnosis.
- ❌ "The BTP's pivot to hand-written CUDA was forced by exactly this" — fabricated narrative; the actual reason for the BTP's hand-written CUDA approach should be sourced from the BTP's own author/notes, not invented.
- ❌ "Static StarPlat Louvain compiles cleanly and produces results within ~10% of hand-written CUDA performance" — fabricated specific benchmark.

**For the BTP.** Cite as the DSL the broader Nasre group works in. The BTP's relationship to StarPlat (whether it builds on StarPlat-generated code, runs alongside, or pivoted away from it) should be sourced from the BTP author's actual project history, not invented.

Citation key: `behera2023starplat`.

```bibtex
@article{behera2023starplat,
  author  = {Behera, Nibedita and Kumar, Ashwina and Rajadurai T, Ebenezer and Nitish, Sai and Pandian M, Rajesh and Nasre, Rupesh},
  title   = {{StarPlat}: A Versatile {DSL} for Graph Analytics},
  journal = {Journal of Parallel and Distributed Computing},
  volume  = {194},
  pages   = {104967},
  year    = {2024},
  doi     = {10.1016/j.jpdc.2024.104967},
  eprint  = {2305.03317},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  note    = {arXiv preprint 5 May 2023; JPDC version 2024 verified via Crossref.}
}
```

*(All authors at IIT Madras, India.)*

---

### Nasre, R. — GPU Programming Course Materials

> ⚠️ **Verification note.** This is course material, not a publication. I have not directly inspected the slide decks. **All my prior specific claims about the slide content have been removed as fabricated.**

Course materials by Prof. Rupesh Nasre, Department of CSE, IIT Madras, available at [cse.iitm.ac.in/~rupesh/teaching/](https://cse.iitm.ac.in/~rupesh/teaching/). The UGRC-I report (Section 2.2, Figure 1) credits a figure as "adapted from Nasre" — which is the verifiable use of these materials by the BTP's predecessor.

**What I cannot independently verify** (and therefore no longer claim):
- Specific topics covered in the slides (kernels/threads/blocks/grids, memory hierarchy, warp sync, atomics, BFS/PageRank/CC exercises)
- That "many of the design patterns the BTP uses ... appear in those exercises"
- That "several of Prof. Nasre's later students ... work in CUDA"
- That "the BTP is essentially a fourth-generation continuation"

**For the BTP.** Cite as the source of the GPU-hierarchy figure (acknowledged in the UGRC-I report). Other claims about course content should be sourced from the slides themselves, not invented.

Citation key: `nasre_gpu_course`.

```bibtex
@misc{nasre_gpu_course,
  author       = {Nasre, Rupesh},
  title        = {{GPU} Programming course materials},
  howpublished = {\url{https://cse.iitm.ac.in/~rupesh/teaching/}},
  institution  = {Indian Institute of Technology Madras},
  note         = {Department of CSE, IIT Madras. Source of the GPU hierarchy figure adapted in the UGRC-I report.}
}
```
