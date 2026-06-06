# Abstract and Motivation — Section Plan

Planning document for two early report sections that serve different purposes:
- **Abstract** — 200–250-word standalone summary. Audience: a reviewer deciding whether to read further.
- **Motivation** — narrative opening of the Introduction. Audience: a reader who has committed to the report and wants to understand *why this matters*.

---

## Part 1 — Abstract

### Sentence-by-sentence template

1. **Context** (1 sentence) — community detection is important, real networks are dynamic, static algorithms require re-running from scratch.
2. **Problem** (1 sentence) — efficient dynamic Louvain on GPU is an open area; existing GPU implementations are static-only or keep update logic on the host.
3. **Contribution** (1–2 sentences) — this work presents a GPU-accelerated dynamic Louvain with three variants (naive, frontier, delta-screening) and an edge-based vs node-based kernel comparison.
4. **Methodology** (1 sentence) — CUDA cooperative kernels, two-level locking, Thrust-based aggregation; evaluated on 8 SNAP graphs against cuGraph, NetworkX, GVE-Louvain, and DF-Louvain.
5. **Key finding 1** (1 sentence) — quantitative speedup on dynamic updates (best-variant vs static-rerun and vs DF-Louvain OpenMP).
6. **Key finding 2** (1 sentence) — quality is within $\epsilon$ of the NetworkX reference after fixing the self-loop modularity bug.
7. **Key finding 3 / novelty** (1 sentence) — edge-based vs node-based kernel tradeoff quantified for the first time.
8. **Closing** (1 sentence) — directions for future work (GPU-native delta-screening, Leiden extension).

### Draft abstract (with `\TODO{}` placeholders)

> Community detection on dynamic graphs — networks that evolve through edge insertions and deletions — is a core task in social, biological, and information network analysis. State-of-the-art CPU implementations of dynamic Louvain achieve strong performance through selective vertex processing (delta-screening, Dynamic Frontier), but GPU implementations remain largely static: existing libraries such as NVIDIA cuGraph re-run the full Louvain algorithm on every update, and hybrid approaches keep screening logic on the host.
>
> This work presents a CUDA-based dynamic Louvain algorithm with three variants — naive warm-start, frontier-based, and delta-screening — built on top of a corrected static kernel. We describe the two-level locking protocol, cooperative-kernel synchronization, and Thrust-based graph aggregation underlying the implementation, and identify and fix a self-loop bug in the modularity-gain calculation that suppresses merges after the first aggregation pass.
>
> We evaluate on \TODO{8} real-world SNAP graphs of up to \TODO{$\sim$335K} vertices and \TODO{$\sim$1M} edges, comparing against NetworkX, cuGraph Louvain, GVE-Louvain (CPU), and DF-Louvain OpenMP. Our delta-screening variant achieves a \TODO{$N\times$} speedup over static re-execution and \TODO{$M\times$} over DF-Louvain OpenMP on batch sizes below \TODO{5\%} of edges, while matching NetworkX modularity within \TODO{$\epsilon$}. We further quantify the edge-parallel vs node-parallel kernel tradeoff: node-parallel is \TODO{$K\times$} faster and more robust against aggregation-phase crashes, at a cost of \TODO{$\delta$} in modularity.
>
> We release the implementation and all scripts, and discuss remaining gaps — a GPU-native delta-screening phase and an extension to Leiden-style refinement — as future work.

### Placeholder checklist

Before submission, replace every `\TODO{}`:

- **$N\times$** — speedup of best dynamic variant over static-rerun on the same GPU. Fill from Results §4.
- **$M\times$** — speedup of best dynamic variant over DF-Louvain OpenMP. Fill from Results §4.
- **5%** — the batch-size threshold below which dynamic wins. Verify from the crossover curve in Chart 4.3d.
- **$\epsilon$** — maximum modularity gap to NetworkX across the 8-graph suite. Fill from the headline table in Results §3.1.
- **$K\times$** — node-based speedup over edge-based, averaged across graphs. Fill from Results §6.1.
- **$\delta$** — node-based modularity gap to edge-based. Same source.

Also verify:
- Graph count (8) if more or fewer graphs are used.
- Graph size upper bound (335K / 1M) against the largest graph actually benchmarked.

### Notes for the writer

- No jargon in the first sentence. A reader outside parallel computing must understand the motivation.
- Every number in an abstract **must** appear verbatim in a table in the body. Reviewers spot-check this.
- Avoid "we show", "we demonstrate" — instead use "achieves", "matches", "is $X\times$ faster".
- Do not claim "first" without a literature-survey defense. Safer phrasing: "To our knowledge, the first full-GPU implementation of delta-screening dynamic Louvain". The related-work survey backs this up.
- Keep to $\leq 250$ words for IJPP / JPDC; $\leq 200$ words if submitting to a conference workshop with stricter limits.
- Re-read after writing the Conclusion. The Abstract should promise exactly what the Conclusion delivers — nothing more, nothing less.

---

## Part 2 — Motivation

### What the Motivation section needs to establish

- Community detection is a central analytical task across many domains.
- Louvain is the practical default, but was designed for static graphs.
- Real-world networks are dynamic; re-running static Louvain on every update is wasteful and often infeasible in practice.
- GPUs can accelerate the throughput, but existing GPU Louvain implementations are static-only or hybrid (host-side update logic).
- This BTP targets that specific gap.

### Narrative outline (paragraph-by-paragraph)

**Paragraph 1 — Graphs are everywhere, community detection is core.**
Model real-world systems as graphs: social media, biological interaction networks, financial transactions, citation graphs. Detecting communities — densely-connected subgroups — is a foundational analytic task underlying influence maximization, targeted intervention, fraud detection, and recommendation. Cite **Blondel et al. 2008** for the canonical motivation, **Fortunato 2010** (survey) if accessible.

**Paragraph 2 — Louvain is the workhorse, designed for static graphs.**
Louvain is the most widely adopted algorithm for modularity-based community detection: fast, simple, produces high-quality partitions in near-linear time. But it was designed for a single static snapshot. Cite **Blondel et al. 2008** again here for the static framing; the UGRC-I report makes the same point and is a valid self-citation.

**Paragraph 3 — Real networks are dynamic; re-running static is wasteful.**
Real graphs evolve — new friendships are made, edges are deleted, interactions updated continuously. Applying static Louvain to a dynamic stream means re-running the entire algorithm on every update, which quickly becomes computationally prohibitive and introduces unacceptable latency for near-real-time analysis on large graphs. Cite **Aynaud & Guillaume 2010**, **Zarayeneh & Kalyanaraman 2021**, **Sahu 2024** for the dynamic-Louvain lineage.

**Paragraph 4 — GPUs offer throughput but existing GPU Louvain is static-only.**
GPUs are architecturally suited to the throughput-bound pattern of Louvain's per-vertex modularity optimization. However, mainstream GPU Louvain implementations (cuGraph, Naim et al. 2017, ACLM) are static: they re-run from scratch on every update. Hybrid approaches (HyDetect, cuVite) keep update logic on the CPU, leaving GPU compute idle during the screening/marking phases. Cite **Naim et al. 2017**, **cuGraph**, **Bhowmik & Vadhiyar 2019** (HyDetect), **Gawande et al.** (cuVite), **Mohammadi/Fazlali 2020** (ACLM).

**Paragraph 5 — The specific gap + one-sentence contribution thesis.**
This report builds on the UGRC-I predecessor — which implemented a static CUDA Louvain with cooperative-kernel synchronization — and extends it with three dynamic variants (naive, frontier, delta-screening) running natively on GPU, closing the gap identified above. *Contribution thesis (one sentence):* "We present the first (to our knowledge) GPU-native dynamic Louvain implementation with on-device frontier propagation, benchmarked against cuGraph, NetworkX, GVE-Louvain, and DF-Louvain across 8 real-world graphs."

### Concrete examples to include

Pick 2–3 vivid examples, one per domain, to anchor paragraph 3:
- **Social networks**: Facebook's friendship graph adds millions of edges per day; Twitter follower/unfollow events.
- **Biological / protein interaction networks**: new experimental data incrementally updates known protein interactions (BioGRID, STRING releases).
- **Financial / transaction networks**: near-real-time fraud detection on card transactions — partition drift on the scale of minutes, not hours.
- **Citation / DBLP networks**: new papers and venues added continuously; DBLP releases roughly monthly.

### Objectives (port from UGRC-I §1.2, updated for the BTP scope)

- Understand the Louvain community detection algorithm end-to-end.
- Survey prior work on parallel / GPU Louvain and dynamic Louvain.
- Extend the static CUDA implementation from UGRC-I with correctness fixes (self-loop bug) and a node-parallel variant.
- Implement three dynamic Louvain variants (naive, frontier, delta-screening) on CUDA.
- Benchmark against NetworkX, cuGraph, GVE-Louvain, and DF-Louvain to quantify the speedup and quality tradeoffs.

### Notes for the writer

- Motivation should be **½ to 1 page** — not longer. Readers want to reach the real content fast.
- Every claim in Motivation gets a citation; this is the most-cited section after Related Work.
- Avoid "this work"; prefer "we", "this report", or active voice.
- End Motivation with a one-sentence contribution thesis that cleanly sets up Chapter 2 (Background).
- Do **not** reveal numerical results here — that's the Abstract's job, and repeating them in Motivation is redundant.
- Reuse UGRC-I §1.1 prose where possible, but tighten — UGRC-I's motivation reads slightly verbose and can be compressed.
