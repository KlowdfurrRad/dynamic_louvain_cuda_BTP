# Dynamic Louvain — GPU

Writing material for the "Dynamic Louvain (GPU)" paragraph(s) of the Related Work section. **This is the central novelty bin for the BTP**: no peer-reviewed paper to date presents a fully GPU-native delta-screening dynamic Louvain. The sub-section below documents the closest adjacent work, what each contributes, why it is *not* what the BTP does, and where the BTP's contribution sits.

---

## Muthavarapu (BTech, IITM) — StarPlat Dynamic Louvain

> ⚠️ **Verification note.** I have not directly inspected the StarPlat dynamic-Louvain code in the GitHub repo. **Specific algorithmic claims I cannot confirm from a primary source have been removed from this entry.** The author given name has been corrected: per the UGRC-I report's references, the author is **Harshitha** Muthavarapu, not "Gajendra" — "gajendra-iitm" is the GitHub username of the StarPlat group lead, not Muthavarapu's first name.

The direct sequential predecessor in the Nasre / IITM line of work. Harshitha Muthavarapu implements a sequential dynamic Louvain in the StarPlat DSL. The code is at [github.com/gajendra-iitm/starplat/tree/main/graphcode/dynamic_louvain_community_detection](https://github.com/gajendra-iitm/starplat/tree/main/graphcode/dynamic_louvain_community_detection).

**What's verifiable** (from the UGRC-I report's reference list, which cites this as "Starplat dynamic louvain code"):
- Implementation exists in the StarPlat DSL.
- Hosted under the IITM / Nasre group's GitHub repository.
- Sequential implementation (not parallel).

**What I cannot verify without inspecting the code** (and therefore no longer claim):
- The specific algorithmic structure (whether it's primed naive-dynamic, frontier, or something else)
- Whether it loads previous-snapshot labels at start
- Whether it computes any affected-vertex set
- Whether the StarPlat CUDA back-end was attempted and failed for this code (the UGRC-I report does not say this; "compilation issues" is something I added without source)

**For the BTP.** Cite as the sequential StarPlat predecessor. Any specific claim about *what the algorithm does* needs to be verified against the actual code before being included in the report. *Action item: clone the StarPlat repo and inspect `dynamic_louvain_community_detection/` directly before final submission.*

Citation key: `muthavarapu_starplat_dynamic_louvain`.

```bibtex
@misc{muthavarapu_starplat_dynamic_louvain,
  author       = {Muthavarapu, Harshitha},
  title        = {{StarPlat} dynamic {Louvain} code},
  howpublished = {\url{https://github.com/gajendra-iitm/starplat/tree/main/graphcode/dynamic_louvain_community_detection}},
  note         = {BTech project, IIT Madras; sequential implementation in the StarPlat DSL}
}
```

---

## UGRC-I (Chandaluru, November 2025) — Dynamic Louvain on GPUs with CUDA

The direct predecessor report to this BTP. Author: Raadhes Chandaluru, under Prof. Rupesh Nasre, Department of CSE, IIT Madras. **Dated November 11, 2025.** The report contains a static GPU Louvain implementation (working) and a *sequential* dynamic Louvain (work-in-progress). Parallel dynamic Louvain is left as future work.

**What the report actually implements (paper-faithful, Section 5):**

1. **Edge-grouped parallelism for local-moving (Section 5.2.3.2):** Each CUDA thread is assigned a *group of edges* to process (not one edge per thread, as I previously claimed). For each edge $(u, v)$ in the group, the thread attempts to move $u$ to $v$'s community if there is a positive modularity gain.

2. **Four-level locking scheme (Section 5.2.3.2, paper-faithful, corrected):** Before committing a move of vertex $u$ to $v$'s community, the thread acquires *four* locks in deterministic order to prevent races:
   - `lock(min(u, v))`, `lock(max(u, v))` — locks both vertices
   - `lock(min(C(u), C(v)))`, `lock(max(C(u), C(v)))` — locks both communities
   
   (My earlier claim of "two-level locking, vertex-level + community-level" undercounted; the actual scheme is *four locks per move attempt*.)

3. **Cooperative kernels for grid-wide synchronisation (Section 5.2.3.1):** Used to synchronise all blocks across local-moving iterations, via `cudaLaunchCooperativeKernel` with `cg::grid_group::sync()`. The report acknowledges the trade-off: "stricter limits on the launch. The GPU must run all blocks concurrently when cooperative kernels are used."

4. **GPU-side aggregation via Thrust (Section 5.2.4.3):** Three Thrust primitives:
   - `thrust::sort` on `(src, dest)` edge pairs with a custom edge comparator
   - `thrust::reduce_by_key` to combine duplicate edges by summing weights
   - `thrust::for_each` + `thrust::inclusive_scan` to populate the new CSR offsets
   
   This is a different (and cleaner) primitive composition than I previously described.

5. **"Slight Change" to algorithm (Section 5.2.2):** Move a vertex if there is *any* modularity gain, not just the maximum gain over all neighbouring communities. Acknowledged as a deviation from canonical Louvain.

**Hardware (Table 1, paper-faithful):**
- Platform: **Google Colaboratory**
- Hardware accelerator: **NVIDIA T4 GPU** (NOT A100 as I previously implied — fabricated)
- GPU driver: 550.54.15, CUDA 12.4
- Sequential CPU compiler: g++ 11.4.0
- Baseline library: cugraph 25.08.00

**Test graphs (Section 7):**
- `graph3`: Karate Club (34 vertices, 76 edges)
- `nxgraph0`–`nxgraph3`: NetworkX Erdős-Rényi random graphs (100–300 vertices, 191–904 edges)

**All test graphs are very small** (largest 300 vertices). The report does not benchmark on real-world graphs at scale; that is explicitly future work.

**Headline results (Table 2, paper-faithful):**
- GPU Static Louvain achieves modest speedup over sequential static Louvain on small graphs (e.g. `nxgraph2` 300-vertex/904-edge: 45 ms GPU vs. 18 354 ms sequential).
- cuGraph 25.08.00 achieves *higher* modularity on most graphs (e.g. `nxgraph2`: 0.468 vs. 0.420). cuGraph wins or ties on quality across all test graphs.
- GPU Static produces *more* communities than cuGraph in some cases (e.g. `nxgraph2`: 47 communities vs. 27 for cuGraph).

**The known bug (Section 7, paper verbatim):**
> "There is an issue in the implementation where the graph is collapsed into a single community usually occurs when the input graph is dense sparse. The modularity will be 0 in this case. It is likely a synchronization error."

(My earlier "three suspects" diagnosis — a race in $\Sigma_{tot}$, missing reset between aggregation passes, edge-parallel kernel mis-aggregating modularity gains — was *fabricated*. The report only says "likely a synchronization error" without enumerating suspects.)

**Dynamic Louvain status (Section 6, paper verbatim):** "The below is the initial dynamic louvain algorithm that has been implemented sequentially. There are problems in the below that need to be solved deeply." The report's *parallel* dynamic Louvain is in the "Future Directions" list (Section 8), not the present implementation.

**For the BTP.** This is the *immediate antecedent*. The BTP is required to (a) reproduce / root-cause the dense-sparse collapse bug, (b) complete a parallel dynamic Louvain implementation, (c) add frontier and delta-screening dynamic variants, and (d) move beyond the small-graph benchmark suite to real-world graphs.

Citation key: `chandaluru2025ugrc1`.

```bibtex
@techreport{chandaluru2025ugrc1,
  author      = {Chandaluru, Raadhes},
  title       = {Dynamic {Louvain} on {GPU}s with {CUDA}},
  institution = {Indian Institute of Technology Madras},
  type        = {UGRC-I Report},
  month       = {November},
  year        = {2025},
  note        = {Under supervision of Prof. Rupesh Nasre. Hardware: NVIDIA T4 GPU on Google Colaboratory.}
}
```

---

## Bhowmick, Vadhiyar, Varun PV (2022) — Multi-GPU (Static; cross-reference)

Cross-reference. The full paper-faithful entry is in [`static_louvain_gpu.md`](static_louvain_gpu.md) under "Bhowmick, Vadhiyar, Varun PV (2022) — Multi-node Multi-GPU Louvain".

**Relevance to dynamic-GPU section.** Bhowmick et al. is *static* and does not address dynamic / streaming graphs. It is cited here only to mark scope — the BTP is single-GPU and does not address multi-GPU; Bhowmick et al. is the closest multi-GPU comparison point and the natural future-work direction.

(Citation key reused: `bhowmick2022scalable`.)

---

## Gawande et al. (2022) — cuVite (Static; cross-reference)

Cross-reference. The full paper-faithful entry is in [`static_louvain_gpu.md`](static_louvain_gpu.md) under "Gawande, Ghosh, Halappanavar, Tumeo, Kalyanaraman (2022) — cuVite".

**Relevance to dynamic-GPU section.** cuVite is *static* and runs aggregation on the CPU (deliberately, due to irregular-access cost). If the BTP wants to argue for keeping aggregation on the GPU in the dynamic setting, cuVite is the explicit alternative-design counter-reference.

(Citation key reused: `gawande2022cuvite`.)

---

## Mohammadi, Fazlali, Hosseinzadeh (2021) — ACLM (Static GPU; cross-reference)

Cross-reference. The full paper-faithful entry is in [`static_louvain_gpu.md`](static_louvain_gpu.md) under "Mohammadi, Fazlali, Hosseinzadeh (2021) — ACLM".

**Important correction (paper-verified).** ACLM is *not* a "frontier" or "active-list" approach. The "adaptive" in ACLM refers to **runtime GPU launch-parameter selection** (block count and threads-per-block based on per-vertex degree), *not* to active-vertex-set shrinking. There is no `moved` flag, no active list, no host round-trip per iteration. My earlier framing of ACLM as "the closest published relative to the BTP's frontier dynamic kernel" was wrong — ACLM is not a frontier method, and the BTP's frontier kernel has no real ACLM precursor.

ACLM is also fully *static* — it does not address dynamic / streaming graphs in any sense.

(Citation key reused: `mohammadi2021accelerating`.)

---

## "Breaking the Latency Barrier" (Springer 2025) — Real-Time Incremental CD

> ⚠️ **Verification note.** I have no copy of this paper, and the bibliographic record itself is uncertain — exact title, authors, book/series, DOI all unverified. **All my prior specific algorithmic claims about this work have been removed as fabricated.**

I previously claimed this paper proposes:
- A "streaming view over edge-insert/delete log"
- A "configurable latency budget (e.g. 100 ms stale)"
- A "latency-budget scheduler" deciding incremental vs. batched re-evaluation
- That Louvain or LPA is the per-query backend
- That for "live" graphs the bottleneck is materialisation time
- That a "GPU-resident storage layer" is needed for GPU-Louvain in DB setting

**None of these claims have a verifiable source.** All were fabricated. If this paper is to be cited, the bibliographic details must be located first (probably via Google Scholar / Springer Link), then the actual content read.

**Action item:** drop this entry from the related-work list, OR locate the actual paper before final submission. Until then, do not include any specific claims about its contributions.

Citation key (placeholder, do not use): `breakinglatency2025_unverified`.

---

## Chakaravarthy et al. — Distributed-Memory Dynamic Community Detection

> ⚠️ **Verification note.** I have no copy of this paper. The citation in the prior version of this file was a placeholder ("authors to be confirmed", "Distributed-memory dynamic community detection (exact title to be confirmed)"). **All my prior specific algorithmic claims have been removed as fabricated.**

I previously claimed this paper:
- Uses a "bulk-synchronous batch-update protocol" with broadcast and global synchronisation
- Reports modularity "within 1–2% of from-scratch when batch size is below ~1% of total edge count"
- Shows the dynamic update can be worse than from-scratch above that batch threshold

**None of these claims have a verifiable source.** All were fabricated. The actual Chakaravarthy work in dynamic CD (if it exists) needs to be located and verified.

**Action item:** drop this entry from the related-work list, OR locate the actual paper before final submission. The prior placeholder BibTeX has been removed.

---

## Summary of the Novelty Argument

When this section is finally written in prose, the load-bearing claim is (corrected to remove fabrications):

> No peer-reviewed work to date publishes a *fully GPU-resident* dynamic Louvain that combines (a) GPU-side delta-screening or frontier expansion, and (b) GPU-side aggregation across multiple snapshots. The closest published work is either (i) **static-only on GPU** (Naim 2017, Mohammadi 2021 ACLM, Gawande 2022 cuVite, cuGraph; Bhowmik 2019 HyDetect is also static — it addresses out-of-core graphs, not dynamic graphs), (ii) **dynamic but CPU-only** (Halappanavar 2017 dynamic Grappolo, Sahu 2024 DF-Louvain, Zarayeneh 2021 delta-screening), or (iii) **multi-GPU but static** (Bhowmick 2022, Gawande 2022 cuVite). This BTP fills the gap of single-GPU dynamic Louvain.

**Corrections from the prior version of this summary:**
- ❌ "Mohammadi 2020 hybrid CPU-GPU dynamic-ish with dynamism logic on host" — ACLM is fully static; the "adaptive" refers to launch parameters, not dynamic-graph handling.
- ❌ "Bhowmik 2019 HyDetect hybrid CPU-GPU dynamic-ish" — HyDetect is static; the hybrid CPU-GPU split addresses out-of-core processing, not graph dynamism.

Each clause of the corrected sentence has a citation; the bibliography assembled across the four files (`static_louvain_cpu.md`, `static_louvain_gpu.md`, `dynamic_louvain_cpu.md`, `dynamic_louvain_gpu.md`) supplies them.
