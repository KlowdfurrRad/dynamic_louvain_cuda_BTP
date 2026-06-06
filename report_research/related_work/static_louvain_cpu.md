# Static Louvain — CPU

Writing material for the "Static Louvain (CPU)" paragraph(s) of the Related Work section. Each entry is a paper, a short discussion of what it contributes, and the LaTeX citation key + BibTeX entry to drop into `references.bib`.

---

## Blondel, Guillaume, Lambiotte, Lefebvre (2008) — original Louvain

The canonical Louvain paper. Introduces the two-phase heuristic that now bears the authors' town name: (1) a *local-moving* phase in which each vertex is repeatedly moved to the neighbouring community that yields the largest modularity gain, and (2) an *aggregation* phase in which each community is collapsed into a super-vertex and the process is restarted on the resulting coarser graph. The gain of moving a vertex $i$ into community $C$ is derived in closed form so that each candidate move is $O(1)$ given precomputed community sums $\Sigma_{tot}$ and $k_{i,in}$. The paper demonstrates near-linear scaling in practice and reports communities on networks up to 118M nodes that earlier spectral / greedy methods could not touch.

**The closed-form gain — what made Louvain fast.** The pre-Louvain greedy modularity-maximisation algorithms (Newman 2004, Clauset–Newman–Moore 2004) had to recompute modularity from scratch — $O(m)$ — for every candidate merge. Blondel et al.'s key derivation is that for a *single vertex* moving from its current community to a neighbouring community $C$, the modularity change is

$$\Delta Q = \left[\frac{\Sigma_{in} + 2\,k_{i,in}}{2m} - \left(\frac{\Sigma_{tot} + k_i}{2m}\right)^2\right] - \left[\frac{\Sigma_{in}}{2m} - \left(\frac{\Sigma_{tot}}{2m}\right)^2 - \left(\frac{k_i}{2m}\right)^2\right]$$

which simplifies to a constant-time evaluation per candidate community given the running sums. Over the full local-moving phase this drops the per-iteration cost from $O(nm)$ to $O(m)$, which is the source of Louvain's near-linear practical scaling.

**Heuristic choices the paper makes (and that everyone inherits).** Three quietly load-bearing decisions: (i) *vertex iteration order is arbitrary* — the paper says "in a random order" but the reference implementation uses input-file order, and most subsequent implementations use either input order or degree order; this is exactly the freedom that parallel implementations exploit; (ii) *only neighbouring communities are considered as move targets*, not arbitrary communities — this is what makes each candidate $O(\deg(v))$ instead of $O(\#\text{communities})$; (iii) *a vertex moves only if $\Delta Q > 0$* (strictly positive), and ties go to the original community — the strict inequality is what guarantees monotone modularity increase per move and therefore termination.

**Interesting bit.** The paper reports running on a 118-million-node mobile-phone call-graph in 152 minutes on a single core in 2008 — an existence proof that the algorithm scales, and the headline result that established Louvain as the default community-detection method. The graph itself was never released; the result is widely cited but never reproduced.

For this BTP, Blondel et al. is the algorithmic specification being parallelised. Every GPU kernel — whether naive, frontier, or delta-screening — produces exactly this local-moving phase and exactly this aggregation phase, just with vertex iteration reordered for parallelism. Citing this is mandatory.

Citation key: `blondel2008fast`.

```bibtex
@article{blondel2008fast,
  author  = {Blondel, Vincent D. and Guillaume, Jean-Loup and Lambiotte, Renaud and Lefebvre, Etienne},
  title   = {Fast unfolding of communities in large networks},
  journal = {Journal of Statistical Mechanics: Theory and Experiment},
  volume  = {2008},
  number  = {10},
  pages   = {P10008},
  year    = {2008},
  doi     = {10.1088/1742-5468/2008/10/P10008}
}
```

---

## Brandes, Delling, Gaertler, Görke, Hoefer, Nikoloski, Wagner (2007/2008) — On Modularity Clustering

The theoretical companion to Louvain. Brandes et al. formalise modularity, prove that maximising it is NP-hard (via reduction from minimum multiway cut), and analyse the behaviour of several greedy / spectral / ILP approaches. They give the now-standard definition $Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)$, which every Louvain implementation — including this BTP — reproduces.

**The hardness reduction.** Brandes et al. reduce *3-PARTITION* (a strongly-NP-hard variant of bin packing) to modularity maximisation, showing that even deciding whether modularity exceeds a threshold $Q^*$ is NP-complete. Crucially, they also show that the problem remains hard even when restricted to graphs of bounded degree — so the hardness is not just a pathology of dense graphs. The implication for Louvain: any polynomial-time algorithm is necessarily a heuristic, and there is no point chasing exact methods.

**The ILP and what it tells us.** The paper formulates modularity maximisation as an integer linear program with one binary variable $x_{uv}$ per vertex pair (1 if $u$ and $v$ are co-clustered). Solving this exactly is feasible only up to ~30 vertices, but the LP relaxation gives an upper bound on optimal modularity. They use this bound to show that Louvain (and other heuristics) often achieves modularity within 5–10% of the true optimum on small benchmark graphs — the first quantitative evidence that the heuristic gap is small in practice.

**The resolution-limit observation.** The same paper notes (and Fortunato & Barthélemy formalise in 2007) that modularity has a built-in size bias: communities smaller than $\sqrt{2m}$ get merged into larger ones even when the underlying structure is clearly two distinct groups. This is the *resolution limit* and is the deepest known weakness of modularity as an objective. Louvain inherits it; Leiden mitigates but does not eliminate it.

**Interesting bit.** The paper observes that on a complete bipartite graph $K_{n,n}$, optimal modularity is achieved by a *non-bipartite* clustering — a counter-intuitive result that became a teaching example for the resolution limit.

Used in this BTP to justify (i) the modularity equation in the Background section, (ii) the claim that exact maximisation is intractable (so a heuristic is unavoidable), and (iii) the existence of the resolution-limit pathology that motivates later work like Leiden.

Citation key: `brandes2008modularity`. (The final IEEE TKDE version appeared in 2008; the preprint is 2007.)

```bibtex
@article{brandes2008modularity,
  author  = {Brandes, Ulrik and Delling, Daniel and Gaertler, Marco and G\"{o}rke, Robert and Hoefer, Martin and Nikoloski, Zoran and Wagner, Dorothea},
  title   = {On Modularity Clustering},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume  = {20},
  number  = {2},
  pages   = {172--188},
  year    = {2008},
  doi     = {10.1109/TKDE.2007.190689}
}
```

---

## Rotta, Noack (2011) — Multilevel Local Search Algorithms for Modularity Clustering

Rotta and Noack do for modularity clustering what the multilevel-graph-partitioning community had already done for minimum-cut partitioning: they organise the entire space of local-search heuristics into a coherent **five-dimensional design space**, populate it with both existing and new techniques, and run a systematic experimental comparison across 62 real-world graphs. The payoff is concrete: a previously unevaluated combination — single-step cluster joining with their new Z-Score prioritizer, plus multilevel refinement by local vertex moving — turns out to be competitive with the best published modularity-clustering methods at the time.

**The five-dimensional design space.** A multilevel modularity-clustering algorithm in their framework is parameterised by:

1. **Coarsening algorithm** — *Cluster Joining* (CJ), which iteratively joins two clusters, or *Vertex Moving* (VM), which iteratively moves vertices between clusters. Within VM there are two flavours: *Local Moving* (LM, randomised vertex order, identical to Blondel et al.'s Louvain local-moving) and *Global Moving* (GM, repeatedly perform the **globally best** vertex move, i.e. true highest-gain-first). Within CJ: *Single-Step Joining* (CJ0, one pair per iteration) or *Multistep Joining* (CJx, $l$ disjoint pairs per iteration, after Schuetz–Caflisch).
2. **Coarsening prioritizer** — the criterion used to pick the joined pair or the moved vertex/target. Six candidates: Modularity Increase (MI, the "obvious" choice), their new **Z-Score (ZS)**, their new **Weighted Density (WD)**, Graph Conductance (GC, after Danon et al.), and Wakita–Tsurumi's WHN/WHE.
3. **Number of coarsening levels** — controlled by a *reduction factor* $\alpha$: each coarsening level must shrink the cluster count by at least $\alpha\%$ before contracting. Conventional single-level refinement is the special case $\alpha=100\%$.
4. **Refinement algorithm** — LM, GM, *Adapted Kernighan–Lin* (KL, which can accept modularity-decreasing moves to escape local maxima), or none.
5. **Refinement prioritizer** — same menu as (2), but only MI matters in practice for VM-based refinement (see below).

**The two new prioritizers.**
- **Z-Score (ZS)** — defined as $\Delta Q_{C,D} / \sqrt{\deg(C)\deg(D)}$. Motivation: under their null model, the Modularity Increase $\Delta Q_{C,D}$ between two clusters has standard deviation that grows with $\deg(C)\deg(D)$, so MI as a raw priority is biased toward joining large clusters. ZS normalises by the standard deviation, eliminating the size bias and making the priority a "number of standard deviations the actual edge weight exceeds the expected".
- **Weighted Density (WD)** — $f(C,D) / (\deg(C)\deg(D))$, the actual edge weight relative to the null-model expectation. WD is biased toward joining *small* clusters (the opposite of MI's bias).
- **ZS sits between MI and WD** as a balanced compromise; experimentally it is the best overall prioritizer for cluster joining.

**Multilevel Refinement (their key new contribution to refinement).** Conventional refinement runs vertex moving only on the original graph (Single-Level Refinement, SL). Rotta & Noack introduce **Multilevel Refinement (ML)**: run a refiner on *every* coarsening level in reverse order, projecting each level's clustering to the next-finer level as a starting point. The intuition: at coarse levels, an entire group of vertices can be moved together by re-assigning a single super-vertex — moves that are far too expensive to accomplish through single-vertex moves on the original graph. This is what their abstract calls "moving entire clusters from each of these stages, not only individual vertices".

**The surprising LM-vs-GM finding.** A naive reading would predict that GM (always perform the *globally* best move) beats LM (perform the best move *for each vertex in randomised order*) on quality, since GM is greedier. Their experiments (Section 5.4, Figure 3) show the **opposite**: GM is both substantially slower than LM *and* slightly less effective. The cause is that the MI prioritizer (which GM has to use, by definition) creates an *unbalanced cluster growth* pathology — once one cluster gets big, every globally-best move pulls more vertices into it, producing one "absorbing" giant cluster. LM's randomised vertex order naturally interleaves moves to many different growing clusters, breaking this positive-feedback loop. This is a clean empirical justification for Louvain's randomised-order design choice that pre-dates Leiden's similar argument.

**Why prioritizers don't matter much in VM (and do in CJ).** In LM, prioritizers only ever compare moves *of the same vertex* to different target clusters — and for moves of the same vertex, the source cluster is fixed, so all prioritizers reduce to the same ordering. In GM, prioritizers compare moves of *different* vertices, and there only MI is well-defined (the alternatives ignore the current cluster of the moving vertex). In CJ, by contrast, the prioritizer compares *different cluster pairs* and ZS/WD/GC/MI all give substantively different orderings — which is why prioritizer choice matters for CJ but not for VM.

**Headline empirical result (Section 5.6).** Two best combinations:
- **CJ0_ZS+ML50+LM** — single-step joining with ZS prioritizer, multilevel coarsening at 50% reduction factor, multilevel refinement by local vertex moving.
- **LM_MI+ML100+LM** — local moving with MI prioritizer (i.e. Blondel-style coarsening), reduction factor 100%, plus multilevel refinement by LM.

Both beat the published Louvain (LM_MI without refinement) by 1–2% modularity on the benchmark suite. Crucially, **multistep joining (CJx) is no improvement over single-step joining (CJ0)** when paired with a good prioritizer like ZS — the multistep complexity introduced by Schuetz & Caflisch is unnecessary. And **Blondel et al.'s Louvain — algorithmically equivalent to LM_MI with reduction factor 100% and no refinement — is "somewhat faster but slightly less effective with refinement"** than the new combinations.

**Interesting bit #1.** The paper observes that the Adapted Kernighan–Lin refiner (KL) — which can perform modularity-*decreasing* moves to escape local optima — is the *most effective* refiner on quality (Figure 5), but its runtime overhead is so large (5–10× slower than LM) that it is "unsuitable for most applications". This is the cleanest case in the paper of a quality–speed Pareto frontier where the algorithmic improvement is real but the engineering tradeoff doesn't survive scaling.

**Interesting bit #2.** Section 6 benchmarks against published implementations of seven prior algorithms — Newman spectral, Wakita–Tsurumi (HE/HN), Pons–Latapy walktrap, Clauset–Newman–Moore, Schuetz–Caflisch, Blondel-Louvain, Reichardt-spinglass — on graphs up to 75k vertices. CJ0_ZS+ML50+LM is among the fastest *and* among the most effective. Reichardt's spinglass simulated-annealing approach finds slightly higher modularity but is **~2400× slower** than Louvain; Newman spectral is ~150× slower. This is the existence proof that, at the time, the multilevel-local-search family was the best Pareto-optimal point available for modularity clustering.

**Relevance for the BTP.** Two specific things to take from this paper:
1. The vertex-ordering question — randomised LM beats deterministic-order GM not because the randomisation is somehow "better information" but because it *defeats a feedback pathology* (unbalanced cluster growth from MI). This reframes the BTP's quality gap on parallel kernels: when warp-scheduling-driven ordering produces lower modularity than NetworkX's deterministic order, the relevant question is not "is the order good?" but "does the order accidentally re-introduce MI's giant-cluster pathology?".
2. **Multilevel Refinement** — their second-pass refinement at every coarsening level is the cleanest "fix" for Louvain that pre-dates Leiden. It is also natural to port to GPU: each refinement pass is just another local-moving phase on a smaller (coarsened) graph, exactly the kernel the BTP already runs. A future-work direction.

Citation key: `rotta2011multilevel`.

```bibtex
@article{rotta2011multilevel,
  author    = {Rotta, Randolf and Noack, Andreas},
  title     = {Multilevel Local Search Algorithms for Modularity Clustering},
  journal   = {ACM Journal of Experimental Algorithmics},
  volume    = {16},
  number    = {2},
  articleno = {2.3},
  numpages  = {27},
  year      = {2011},
  doi       = {10.1145/1963190.1970376},
  note      = {Earlier version at SEA 2009}
}
```

---

## Lu, Halappanavar, Kalyanaraman (2015) — Grappolo

Lu et al. present the canonical OpenMP-parallel Louvain implementation. The paper makes two distinct contributions: (i) a careful analysis of *why* a naive parallel Louvain breaks down, formalised as two named pathologies (the *negative gain scenario* and the *swap / local-maxima scenario*), each backed by a small lemma; and (ii) three engineering heuristics — **Minimum Label (ML)**, **Vertex Following (VF)**, and **distance-1 graph Coloring** — that mitigate those pathologies in different ways. The implementation is named **Grappolo** ("cluster of grapes" in Italian) and is released under BSD-3 at `hpc.pnl.gov/people/hala/grappolo.html`. Headline result: up to 16× absolute speedup over the original serial Louvain on 32 cores, while *improving* modularity for 7 of 11 benchmark inputs.

**Pathology 1: the negative-gain scenario (Section 4.1).** Two vertices $i$ and $j$, sitting in distinct communities, are both connected to a third vertex $k$ in yet another community $C(k)$. Each independently computes that moving into $C(k)$ would increase modularity. In serial, one moves first and the other re-evaluates and may decide differently. In parallel, both move simultaneously. The actual joint $\Delta Q$ when both vertices land in $C(k)$ at once is

$$\Delta Q_{\{i,j\} \to C(k)} = \Delta Q_{i \to C(k)} + \Delta Q_{j \to C(k)} + \frac{\omega(i,j)}{m} - \frac{2 k_i k_j}{(2m)^2}$$

If $i$ and $j$ are not directly connected ($(i,j) \notin E$), the $\omega(i,j)/m$ term vanishes and the joint move is *strictly worse* than the sum of the individual gains — and can in fact be **negative** even though both individual gains were positive. **Lemma 1** of the paper formalises this: parallel community updates cannot guarantee net positive modularity gain, so the standard termination argument (modularity is monotone increasing → algorithm terminates) breaks. **Corollary 2** extends this: distance-1 colouring does *not* preclude negative gains either, because $i$ and $j$ in this scenario are non-adjacent (different colours allowed) yet still produce the pathology.

**Pathology 2: swap and local-maxima (Section 4.2).** Two named sub-cases. First, the simple swap: two singleton-community vertices $i$ and $j$ connected by an edge, each preferring to join the other's community; in parallel they swap singletons and make no progress (Figure 2, Case 1a). Second, the more general "stuck partial communities": a 4-clique $\{i_4, i_5, i_6, i_7\}$ where pairwise moves create $\{i_4, i_6\}$ and $\{i_5, i_7\}$ as a local maximum, when the global maximum is to merge all four (Figure 2, Case 2a).

**Heuristic 1: the Minimum Label (ML) heuristic.** Two variants. (a) **Singlet minimum label** — when a singleton vertex $i$ wants to move into another singleton $j$'s community, the move is performed *only if* $\ell(C(j)) < \ell(C(i))$, where $\ell(\cdot)$ is the (arbitrary, fixed) numerical community label. This is a globally consistent tie-breaker: of the two would-be swappers, only the one moving to the *smaller-labelled* community actually moves. (b) **Generalised minimum label** — when a vertex has *multiple* equally-good move targets, pick the target with the smallest community label. This breaks the local-maxima pathology by funnelling all four 4-clique vertices into the same target community (Figure 2, Case 2b). Crucially, ML "may delay convergence but can never lead to nontermination" — this is the key correctness property that makes ML the load-bearing heuristic of the three.

**Heuristic 2: Vertex Following (VF) — preprocessing-time reduction.** Lu et al. prove (**Lemma 3**) that any single-degree vertex $i$ — i.e., one whose only edge is $(i, j)$ — always ends up in the same community as $j$ in the final serial Louvain solution. Proof sketch: a counterfactual analysis of $\Delta Q_{i \to C(j)}$ shows it must be positive at any iteration where $i$ and $j$ are separated, because otherwise $a_{C(j)} > 2m$ would be forced — impossible by definition. **The exploitation:** as preprocessing, merge every single-degree vertex into its unique neighbour *a priori*, then run Louvain on the compressed graph. Two benefits beyond the obvious vertex-count reduction: (a) hub vertices become the main drivers of community decisions instead of being "distracted" by spokes; (b) in parallel, hub vertices cannot temporarily migrate to one of their spokes' singleton communities — a subtle source of stuckness. This single-degree case is the version implemented; the paper sketches but does not implement a recursive "single-neighbour" extension based on $k$-core decomposition.

**Heuristic 3: distance-1 graph Coloring.** Vertices of the same colour are non-adjacent, so processing one colour class at a time eliminates the simple swap pathology by construction. The colouring is computed in parallel (the paper cites a separate implementation [12], not Jones–Plassmann specifically); only distance-1 is explored. The cost is a one-time colouring pass plus *reduced parallelism within a colour class*. The paper notes this is an *optional* preprocessing step.

**The full pipeline (Section 5.4).** (1) VF preprocessing (optional). (2) Coloring preprocessing (optional). (3) Phases — within each phase, run iterations of parallel sweeps using ML, terminating when modularity gain falls below threshold. If colouring is on, process colour sets in sequence; once vertex count drops below 100k *or* per-phase modularity gain falls below $10^{-2}$, stop colouring and switch to default $10^{-6}$ termination threshold. (4) Graph rebuilding between phases — partly serial (community renumbering), partly parallel.

**Implementation details worth knowing.** Atomic updates use GCC intrinsics `__sync_fetch_and_add()` and `__sync_fetch_and_sub()`. Per-vertex neighbouring-community accumulator uses `std::map` (they tested `std::unordered_map`, found no improvement). The graph rebuild step's edge-update phase requires *one lock for intra-community edges, two for inter-community* — and Section 6 observes this is a real scaling bottleneck on graphs whose first phase ends in low modularity (Europe-osm, NLPKKT240).

**Empirical findings (Sections 6.2 and 6.2.2).**
- *Speedup:* relative speedup approaches 8× at 32 threads on most inputs but is sub-linear past 8 threads. Absolute speedup over the serial Louvain reference reaches ~16× on the best-case inputs (and the serial implementation simply *fails* to complete on Europe-osm and friendster — the parallel version is the only way to process them at all).
- *Quality:* on 7 of 11 inputs, parallel Grappolo produces **higher** modularity than serial Louvain. On 3 of the 4 where serial wins, the difference is in the third decimal place. The largest gap is +0.1 modularity for coPapersDBLP in favour of parallel — suggesting parallel ordering can *escape* some serial local optima.
- *VF can hurt:* on Europe-osm and Rgg_n_2_24_s0, VF *increases* runtime. Mechanism (Section 6.2): without spoke vertices, the per-iteration modularity gain from compressing chains stays just above the termination cutoff, so phases run longer. Adding the recursive single-neighbour VF extension might fix this.
- *Coloring can hurt:* on uk-2002, distance-1 colouring produces 943 colours with relative-standard-deviation 18.876, meaning many tiny colour classes that under-utilise threads. On most other inputs, colouring gives 3.48–16.52× speedup over baseline+VF.

**Interesting bit #1.** Lemma 1's negative-gain scenario means the parallel algorithm has *no theoretical termination guarantee* — only the empirical "minimum modularity gain threshold" cutoff. The paper accepts this honestly rather than papering over it: "Pessimistically speaking, if the net modularity gain can become negative between consecutive iterations, then there is no theoretical guarantee that the algorithm will terminate." The minimum-required-gain threshold is a band-aid that works in practice because the negative gains are small and rare. This is a useful honest precedent — the BTP can adopt the same framing for any non-monotone behaviour observed in its GPU kernel.

**Interesting bit #2.** The minimum-label heuristic only delays convergence; it does not change correctness. So "what's the right tie-breaking rule?" is a pure quality-vs-speed knob, not a quality-vs-correctness knob. That matters for the BTP, where atomic-based resolution is also "delays convergence but can't break correctness", as long as the algorithm is monotone — which by Lemma 1 it isn't, in parallel. So the precise statement is: ML or atomics both produce correct *terminating* runs in practice, but neither has a formal termination proof.

**Interesting bit #3.** The original serial Louvain reference implementation (Blondel's released code) *fails to complete* on graphs with ~1.8B edges (friendster). Parallelisation is not just a speed-up here; for that graph class it is the difference between "answer in minutes" and "no answer at all".

**Relevance for the BTP.** Three concrete things:
1. The negative-gain scenario (Lemma 1) is the *root explanation* for why GPU kernels can occasionally produce lower modularity than NetworkX even with correct $\Delta Q$ computation — the parallelism is fundamentally non-monotone, and the gap is intrinsic, not an implementation bug. Cite this whenever the BTP's evaluation has to defend a quality gap.
2. **VF preprocessing is essentially free for the GPU kernel** — single-degree vertex compression is a one-pass parallel scan, and removing them shrinks the working set. The BTP currently doesn't do this; it should.
3. The rebuild-phase locking analysis (one lock for intra-community edges, two for inter) is a useful diagnostic for the BTP's edge-based kernel crash on dense graphs: that crash is during graph aggregation, which is exactly the phase Lu et al. flag as the locking bottleneck. The two-lock-per-edge inter-community case is a plausible source of contention or race.

Citation key: `lu2015parallel`.

```bibtex
@article{lu2015parallel,
  author  = {Lu, Hao and Halappanavar, Mahantesh and Kalyanaraman, Ananth},
  title   = {Parallel heuristics for scalable community detection},
  journal = {Parallel Computing},
  volume  = {47},
  pages   = {19--37},
  year    = {2015},
  doi     = {10.1016/j.parco.2015.03.003},
  note    = {Implementation released as Grappolo, BSD-3, hpc.pnl.gov/people/hala/grappolo.html}
}
```

---

## Staudt, Meyerhenke (2016) — Engineering Parallel Algorithms for Community Detection in Massive Networks (NetworKit's PLM)

Staudt and Meyerhenke (KIT) engineer a *family* of four shared-memory community-detection algorithms inside a single coherent framework, all distributed as part of the **NetworKit** open-source toolkit. The framing is explicitly "algorithm engineering": three of the four algorithms are placed on the Pareto frontier of speed-vs-modularity in their experimental comparison. The headline result: PLM processes a **3.3-billion-edge web graph (`uk-2007-05`) in ~156 seconds at 32 threads** on a single 16-core/32-HT workstation. Preliminary version appeared at ICPP 2013; this is the journal version.

**Four algorithms, not one (Section III).**
1. **PLP** — *Parallel Label Propagation.* Their own variant of Raghavan-style LPA. Each iteration picks the most-frequent (weighted-dominant) neighbour label per node. Crucially, they introduce an **active-set optimisation** (Algorithm 1, lines 7, 12, 14): nodes whose label is already the heaviest in their neighbourhood become *inactive*, and are only reactivated if a neighbour's label is updated. Iterations terminate when the number of changed nodes per iteration falls below threshold $\theta = n \cdot 10^{-5}$ — they note the original Raghavan termination criterion ("stop when all nodes have the heaviest label among neighbours") doesn't converge on some inputs. Reported processing rate: **~50M edges/second**.
2. **PLM** — *Parallel Louvain Method.* The first parallel implementation of Louvain reported "for large inputs". Algorithm 3.
3. **PLMR** — *PLM with Refinement.* An extra `move` phase after each prolongation, re-evaluating node assignments at each level given the changes from the next coarser level. Explicitly inspired by **Noack–Rotta's** multilevel-refinement work (their reference [31]). Algorithm 4.
4. **EPP** — *Ensemble Preprocessing.* An ensemble-learning twist: run $b=4$ PLP instances in parallel, hash their per-node label tuples (via the `djb2` hash) to identify "core communities" — nodes that *all* base instances agreed on — coarsen by those, and run PLMR as the final algorithm on the result. Algorithm 5.

**PLM's actual parallelisation strategy (Section III-B).** This is where my prior summary was most wrong:
- **No batching, no cache-aware partitioning, no thread-local counter shadows.** The actual approach is a straightforward `parallel for v in V` with OpenMP `schedule(guided)` for load balancing. The `guided` schedule is the only concession to scale-free degree skew — it dispatches decreasing chunk sizes from a queue to threads, so threads that pull large neighbourhoods early end up with smaller tail-work.
- **Embraces data races.** The paper is unusually frank: "This approach may work on stale data so that a monotonous modularity increase is no longer guaranteed... such undesirable decisions can also be corrected in a following iteration, which is why the solution quality is not necessarily worse." Their honest justification: "Working only on independent sets of vertices in parallel would not provide a solution since the sets would have to be very small". And empirically: "Concerns about parallel correctness turned out to be theoretical for our set of benchmark graphs, all of which can be successfully processed with PLM." This is the explicit philosophical opposite of Grappolo's min-label heuristic.
- **The std::map → std::vector evolution.** An earlier PLM version stored per-node neighbouring-community edge-weights in a `std::map`; the lock plus map-allocation overhead was prohibitive. They later replaced the map with a `std::vector` reused per thread (factor-2 speedup, $O(p \cdot n)$ extra memory). They later went further and *eliminated even this* — recomputing the per-community weight each time a node is evaluated turned out to be faster than maintaining incremental volumes. The map-based version is preserved as **PLM\*** under tighter memory constraints (about 2× slower than PLM).
- **Parallel coarsening.** Earlier the coarsening phase ("graph rebuild") was a serial bottleneck. They split it: each thread scans a portion of edges, builds partial coarse graphs $G'_t$, then a second parallel pass merges these per-node by processing each node of $G'$ in parallel, merging adjacencies stored in each $G'_t$.

**PLMR (Section III-C).** After standard Louvain prolongs the coarse-level community assignment back to the fine-level graph, run an additional `move` pass. This is the key qualitative difference from vanilla Blondel-Louvain — and the key conceptual ancestor of Leiden's refinement phase. Cost: small running-time increase. Benefit: "(sometimes significant) improvement in modularity" (Figure 7c shows modularity gains up to ~0.02–0.10 on certain inputs).

**The resolution-limit knob.** PLM exposes a parameter $\gamma \in [0, 2m]$ that scales the null-model term: $\gamma = 0$ collapses to a single community, $\gamma = 1$ is standard modularity, $\gamma = 2m$ produces singletons. Tuning $\gamma$ is offered as the "practical remedy" against modularity's resolution limit (Lambiotte 2010, their reference [18]).

**Hardware (Table I).** 2 × 8-core Intel Xeon E5-2680 @ 2.70 GHz = **16 physical cores, 32 threads with HyperThreading**, 256 GB RAM, SUSE 13.1-64, GCC 4.8.1. (Not 32 physical cores as I had implied.)

**Empirical findings (Section V).**
- *PLP scaling* (Figure 1): on `uk-2007-05`, PLP gets factor-8 speedup from 1 → 32 threads; speedup beyond 16 is from HyperThreading and is therefore sublinear.
- *PLM scaling* (Figure 2): factor-9 speedup at 32 threads on `uk-2007-05`. Move and refinement phases scale well; coarsening "only partially profits from parallelization" (Figure 4 confirms across the 20-graph test set). Coarsening is the residual bottleneck.
- *PLM vs PLMR* (Figure 7c): refinement adds a small modularity bump at modest runtime cost.
- *PLM at scale* (Section V-H, Figure 9): on `uk-2007-05` (~3.3 B edges), PLM at 32 threads finishes in **156 s**, PLMR in 168 s, PLM\* (memory-bounded) in 203 s, PLP in 53 s, EPP(4,PLP,PLMR) in 219 s. CLU_TBB *fails to read the input file* at this scale.
- *Comparison with state-of-the-art* (Sections V-E, V-F, Figure 5):
  - Beats **CLU_TBB** (Auer–Bisseling, the DIMACS-challenge winner) on modularity by a clear margin; CLU_TBB is faster on the largest instances but cannot handle 3.3 B edges.
  - Beats **CEL** (Riedy et al.) on both modularity *and* speed.
  - Loses to **RG, CGGC, CGGCi** (Ovelgönne–Geyer-Schulz) on modularity by a small margin, but those algorithms are *orders of magnitude slower* (often hours where PLM takes seconds).
  - Original sequential Louvain is "still relatively fast (Figure 8a)" but eventually falls behind PLM for large graphs.
- *LFR benchmark* (Figure 6): PLM is robust at recovering ground-truth communities even at strong noise ($\mu = 0.8$); PLP / EPP are less robust at high mixing.

**Interesting bit #1 — the philosophy contrast with Grappolo.** Both papers (Staudt–Meyerhenke 2016 and Lu et al. 2015) parallelise Louvain on shared memory. They take *opposite* design philosophies on the parallel-correctness question. Grappolo introduces the *Minimum Label* heuristic to *prevent* swap pathologies and (within the limits of distance-1 colouring) make decisions race-free. PLM *embraces* the races, justifies them empirically ("turned out to be theoretical for our set of benchmark graphs"), and deliberately avoids the parallelism-killing effect of "small independent sets". On their respective benchmark suites, both papers report their approach wins. There is no published head-to-head; this is a real open question that the BTP could profitably address.

**Interesting bit #2 — the std::map → std::vector → no-storage trajectory.** PLM went through three implementation stages, each strictly faster than the previous, by *removing* incremental data structures rather than improving them. Final version: recompute the per-community edge-weight every time a node is evaluated, store nothing per-node. This is the cleanest demonstration in the parallel-Louvain literature that *redundant computation can beat coordinated update* on modern multicores — a lesson that GPU work has internalised even more aggressively (the BTP's GPU kernel similarly recomputes rather than caches).

**Interesting bit #3 — the active-set in PLP is a frontier in disguise.** PLP's "inactive nodes" (Algorithm 1 line 14) reactivated when neighbours change (line 12) is structurally identical to Sahu's later DF-Louvain *frontier* construction — just applied to LPA rather than to Louvain. The mechanism appears in the literature 8 years earlier than DF-Louvain, in a paper not usually cited as a frontier-method ancestor.

**Interesting bit #4 — three of four on the Pareto frontier.** Figure 5 plots all algorithms on the (modularity, time) plane. PLP, PLM, and PLMR all sit on the Pareto frontier. EPP is dominated (slightly worse modularity than PLM at higher cost) — and they say so honestly, then conclude "In practice, our acceleration of the PLM algorithm have made the ensemble approach less relevant." A nicely candid moment from a paper that could have over-claimed its own ensemble idea.

**Relevance for the BTP.** Three things:
1. **PLM is the right baseline citation when discussing "race-tolerant" parallel Louvain.** The BTP's GPU kernel is more on the PLM end of the spectrum (atomics + tolerated races) than on the Grappolo end (min-label + colouring). PLM's Section III-B justification is the canonical defence of that design choice.
2. **PLMR is the conceptual ancestor of Leiden refinement on shared-memory CPU.** If the BTP ever wants to add a refinement pass, PLMR is the simpler reference point than full Leiden — it adds *one* extra `move` per prolongation, no community-internal sub-Louvain.
3. **Headline numbers worth knowing.** "PLM does 3.3-billion-edge `uk-2007-05` in 156 seconds at 32 threads" is the high-water mark for *single-machine CPU* Louvain that any GPU comparison should beat or honestly explain.

Citation key: `staudt2016engineering`.

```bibtex
@article{staudt2016engineering,
  author  = {Staudt, Christian L. and Meyerhenke, Henning},
  title   = {Engineering Parallel Algorithms for Community Detection in Massive Networks},
  journal = {IEEE Transactions on Parallel and Distributed Systems},
  volume  = {27},
  number  = {1},
  pages   = {171--184},
  year    = {2016},
  doi     = {10.1109/TPDS.2015.2390633},
  note    = {Preliminary version at ICPP 2013. Implementations released in NetworKit (https://networkit.iti.kit.edu/).}
}
```

---

## Fazlali, Moradi, Tabatabaee Malazi (2017) — APLM (Adaptive Parallel Louvain Method)

A multicore-OpenMP parallel Louvain whose central contribution is an **adaptive thread-allocation scheme** for the modularity-gain calculation. Fazlali et al. frame the design problem as a trade-off between *fine-grained* decomposition (multiple threads cooperate to compute the modularity-gain Sigma for **one** neighbour-merge candidate) and *coarse-grained* decomposition (one thread evaluates **one** candidate, multiple candidates in parallel). APLM lets the algorithm switch between these regimes at runtime based on the number of idle cores.

**The two levels of parallelism (Section 4, paper-faithful).**

1. **Inter-neighbour parallelism.** When a community is selected and the algorithm needs to evaluate adding each of its neighbouring nodes, different OpenMP threads handle different neighbours.
2. **Intra-neighbour parallelism.** When the system has *more idle cores than neighbours to evaluate*, the modularity-gain $\Delta Q$ Sigma calculation for a *single* neighbour is itself split across multiple threads (using OpenMP nested parallelism with `OMP_DYNAMIC` disabled).

The runtime decision (Algorithm 2 in the paper, `NP_Modularity`):
- Let `n` = degree of selected node, `c` = number of available cores.
- If `c > n`: enable nested parallelism, set `t = (c - n) / n` threads per neighbour-evaluation.
- Else: one thread per neighbour (single-level only).

This is **not** a frontier / active-set / vertex-pruning method. There is no notion of "dirty" vertices, no convergence threshold on how many vertices are still moving, no skipped-vertex optimisation. APLM processes *all* vertices each iteration; the adaptivity is purely in *how many threads* are allocated to each neighbour-evaluation. (My earlier description framing it as a frontier method was wrong.)

**Hardware (Section 5).** AMD processor at 2.8 GHz with **32 physical cores** (2 hyperthreads each = 64 virtual cores), **128 GB RAM**, Red Hat Enterprise Linux 4.4, OpenMP via G++ 4.8.

**Benchmarks (Table 1) — five graphs from the LAW collection, not SNAP.**

| Benchmark | # nodes | # edges | edges/node |
|---|---|---|---|
| CNR-2000 | 325 557 | 2 738 969 | 8.41 |
| EU-2005 | 862 664 | 16 138 468 | 18.71 |
| IN-2004 | 1 382 908 | 13 591 473 | 9.83 |
| UK-2002 | 18 520 486 | 261 787 258 | 14.14 |
| UK-2007 | 105 896 555 | 3 301 876 564 | 31.18 |

**Baselines used for comparison.** PLM (Staudt & Meyerhenke 2016, fine-grained) and **CADS** (Coarser-grain Adaptive Decomposition Scheme, Bhowmick & Srinivasan 2013, one thread per neighbour). APLM's positioning is "adaptive between PLM and CADS".

**Headline results (Table 2, 64 cores).**

| Algorithm | CNR-2000 | EU-2005 | IN-2004 | UK-2002 | UK-2007 |
|---|---|---|---|---|---|
| PLM | 2.3 s | 5.1 s | 4.6 s | 24.3 s | 284.7 s |
| CADS | 2.1 s | 4.4 s | 4.2 s | 14.8 s | 214.3 s |
| **APLM** | **2.0 s** | **2.4 s** | **3.3 s** | **11.3 s** | **181.9 s** |

APLM is fastest on every benchmark. The paper's "50 % execution-time reduction" headline refers specifically to the EU-2005 result vs. CADS: $(4.4 - 2.4) / 4.4 \approx 45\%$. On the largest graph (UK-2007, 3.3 B edges), APLM is ~36 % faster than PLM and ~15 % faster than CADS.

**Scalability (Fig 4, EU-2005).** All three algorithms achieve sub-linear speedup over sequential Louvain. Speedup grows slowly from 1 to 16 cores. Going from 32 → 64 cores, APLM roughly doubles its speedup while PLM and CADS plateau. Reason: with 64 cores and a graph whose nodes have ~32 average neighbours, APLM's intra-neighbour parallelism activates; PLM and CADS have nothing equivalent.

**Quality (Fig 5, 64 cores).** APLM produces modularity **0.5–1.5 % higher** than PLM and CADS across the 5 benchmarks. The largest improvement is on UK-2007. All three algorithms have qualitatively similar modularity because they share the same hierarchical-clustering base.

**For the BTP.** APLM is the right citation when discussing **runtime-adaptive thread allocation** for the per-vertex modularity-gain inner loop. The BTP's GPU kernels make analogous degree-dependent decisions (warp-per-vertex vs. block-per-vertex) but bake them in at kernel-launch time rather than checking idle-core counts dynamically. APLM is **not** a precursor to delta-screening/DF-Louvain — that lineage runs through Aynaud 2010 → Zarayeneh 2019/2021 → Sahu 2024, with no Fazlali ancestor.

Citation key: `fazlali2017adaptive`.

```bibtex
@article{fazlali2017adaptive,
  author  = {Fazlali, Mahmood and Moradi, Ehsan and Tabatabaee Malazi, Hadi},
  title   = {Adaptive parallel {Louvain} community detection on a multicore platform},
  journal = {Microprocessors and Microsystems},
  volume  = {54},
  pages   = {26--34},
  year    = {2017},
  doi     = {10.1016/j.micpro.2017.08.002},
  publisher = {Elsevier}
}
```

*(Third author's full surname is **Tabatabaee Malazi**. Affiliations: Fazlali and Tabatabaee Malazi at Shahid Beheshti University; Moradi at Islamic Azad University, Kermanshah Branch.)*

---

## Ghosh, Halappanavar, Tumeo, Kalyanaraman, Lu, Chavarría-Miranda, Khan, Gebremedhin (2018) — Vite

The canonical **MPI distributed-memory** Louvain implementation, designed to push the algorithm beyond a single shared-memory node. Where Grappolo (Lu et al. 2015) saturates at one socket–group's memory bandwidth, Vite partitions the graph across MPI ranks (with optional intra-rank OpenMP threading) and tackles graphs that no single node can hold. Two distinct papers in 2018 cover the system: the IPDPS paper introduces the heuristics and reports static benchmarks; the HPEC companion re-evaluates the *same static system* on the 2018 Graph Challenge datasets and adds a third heuristic (incomplete coloring). **Neither paper is a dynamic/incremental Louvain** — both initialise from a singleton partition and cluster each input from scratch.

**Distributed data layout (IPDPS §III).** 1-D vertex partitioning: each rank owns a contiguous range $[v_{\text{lo}}, v_{\text{hi}})$ and stores the corresponding rows of the CSR. Edges crossing partition boundaries introduce **ghost vertices** — read-only proxies on the remote rank that mirror just enough state (current community label, weighted degree) to evaluate a local move. After every local-moving sub-iteration the implementation performs an `MPI_Alltoallv`-style exchange of updated labels for ghosts; the volume of this exchange is the dominant communication cost and the principal target of both heuristics.

**Heuristic 1 — Threshold cycling (IPDPS §IV.A).** Standard Louvain runs each level to a fixed modularity-gain tolerance $\theta$ (e.g. $10^{-6}$). Vite *cycles* through a tightening sequence — typically $10^{-3} \to 10^{-4} \to 10^{-5} \to 10^{-6}$ — and resets to the loose value at the start of each new level. The early loose passes converge fast and reduce the graph aggressively before the tight pass refines the result, which both reduces total iterations and shrinks the per-iteration ghost-exchange volume (because more vertices are clustered before tight passes).

**Heuristic 2 — Probabilistic Early Termination / ET (IPDPS §IV.B).** Each vertex carries a probability $p_v$ of being "active" in the current local-move sweep, decayed multiplicatively by factor $(1-\alpha)$, $\alpha \in (0,1]$ (Eq. 3: $P_{v,k} = P_{v,k-1}(1-\alpha)$ if the community was unchanged across the previous two iterations, else reset to $1$). Larger $\alpha$ ⇒ more aggressive early termination. Vertices that have been stable are sampled out of the sweep stochastically; those that change get $p_v$ reset. This is in spirit the same idea as Cordasco–Gargano's quiescence pruning and a precursor to Sahu's Dynamic Frontier — but applied to *static* graphs, purely as a convergence-acceleration heuristic. A variant **ETC** (Early-Termination with extra Communication) exchanges the activity bitmap each iteration so ghost owners can also skip; the paper reports ETC pays off on heavy-communication graphs only.

**ET preliminary results (IPDPS Table III, paper-faithful).** On the small-end benchmarks at $\alpha = 1$ (most aggressive pruning): **CNR-2000 → 2× speedup**, **Channel-500x100x100-b050 → 58.27× speedup** vs. baseline Vite. The Channel result is a soft outlier driven by an extremely well-clustered graph; CNR is more representative of typical web graphs.

**Hardware and benchmarks (IPDPS §V).** NERSC Cori, Phase II Haswell partition: dual-socket Intel Xeon E5-2698v3 (16 cores/socket, 32/node), 128 GB/node, Aries dragonfly interconnect. Test set: 12 graphs from UFL, SNAP, LAW, Network Repository — including `soc-friendster` (66 M vertices, 1.8 B edges), `uk-2007-05` (106 M vertices, 3.3 B edges), `nlpkkt240` (28 M, 373 M).

**Headline scaling claim (IPDPS §V.C).** **7× speedup over Grappolo at 4096 processes** on `soc-friendster`. **`uk-2007-05` (3.3 billion edges) processed in 32 seconds on 1024 cores** — Grappolo failed on this graph due to single-node memory exhaustion. Modularity within $\sim$2 % of Grappolo on every graph in the suite.

**Honest single-node comparison (IPDPS §V.B).** On 32 cores of a single Cori node, **Grappolo is $\sim$2.3× faster than Vite**. Vite only wins by scaling out: the two systems target different points in the design space, and Vite's overhead (MPI plumbing, ghost synchronisation) is amortised only beyond one node.

**Companion paper — HPEC 2018, "Scalable Distributed Memory Community Detection Using Vite".** An **evaluation paper**, *not* a new algorithm. It benchmarks the same static Vite on the **2018 Graph Challenge** stochastic-block-partition datasets, reporting quality (precision/recall/F-score vs. ground truth) alongside performance. The word "Streaming" in the title refers to the *benchmark suite* — the Graph Challenge ships each graph in `static`, `streamingEdge`, and `streamingSnowball` formats (the legend in Fig. 2) — **not** to streaming computation: Algorithm 1 initialises from a singleton partition (`C_curr ← {{u}|∀u∈V}`) every time, with no warm-start, seeding, or incremental update. *(Correction: an earlier version of this entry wrongly described it as an incremental/batched warm-start extension. It is not — it processes every snapshot from scratch. The genuinely warm-started naive-dynamic lineage is Aynaud 2010 → Halappanavar-2017 "seeded", which is why those live in `dynamic_louvain_cpu.md` and this paper stays here in static.)* The one algorithmic addition over IPDPS is **Incomplete Coloring** — partial distance-1 coloring (Jones–Plassmann, 32–40 color classes, remainder bundled into one class) to cut color-switching synchronisation overhead; it yields the highest modularity but is ~8× the slowest configuration. Hardware differs from IPDPS: **NERSC Edison** (Cray XC30, dual-socket 12-core Xeon E5-2695v2 Ivy Bridge, 24 cores/node, 64 GB/node, Aries), 12 MPI ranks/node × 2 OpenMP threads. Smaller author list (Ghosh, Halappanavar, Tumeo, Kalyanaraman, Gebremedhin).

**For the BTP.** Three uses:
1. **Defines the distributed-memory baseline** that GPU work must implicitly compete against. A single A100 (40 GB HBM) cannot hold uk-2007-05 in CSR form (3.3 B edges × ~12 bytes/edge ≈ 40 GB before any working space), so distributed-CPU and single-GPU systems address overlapping but non-identical graph-size regimes. Worth stating explicitly when comparing.
2. **Sources the Threshold-Cycling idea** that Sahu later ports into GVE-Louvain (Sahu uses tolerance-drop on aggregation, similar in spirit). If the BTP wants to add a CPU/GPU-shared scheduling improvement, citing Vite is the right ancestor.
3. **The ET probability decay** is a clean, low-overhead convergence-acceleration trick that maps naturally to GPU local-moving kernels (per-vertex activity flag, biased coin flip per sweep). Worth at least mentioning as future work even if not implemented.

Software: open source under BSD 3-clause, originally hosted at `hpc.pnl.gov/people/hala/grappolo.html` and on GitHub (`Exa-Graph/vite`).

Citation keys: `ghosh2018distributed` (IPDPS), `ghosh2018scalable` (HPEC).

```bibtex
@inproceedings{ghosh2018distributed,
  author    = {Ghosh, Sayan and Halappanavar, Mahantesh and Tumeo, Antonino and
               Kalyanaraman, Ananth and Lu, Hao and Chavarr{\'\i}a-Miranda, Daniel and
               Khan, Arif and Gebremedhin, Assefaw H.},
  title     = {Distributed Louvain Algorithm for Graph Community Detection},
  booktitle = {Proc.\ 32nd IEEE Int.\ Parallel and Distributed Processing Symp.\ (IPDPS)},
  pages     = {885--895},
  year      = {2018},
  doi       = {10.1109/IPDPS.2018.00098},
  publisher = {IEEE}
}

@inproceedings{ghosh2018scalable,
  author    = {Ghosh, Sayan and Halappanavar, Mahantesh and Tumeo, Antonino and
               Kalyanaraman, Ananth and Gebremedhin, Assefaw H.},
  title     = {Scalable Distributed Memory Community Detection Using {Vite}},
  booktitle = {Proc.\ 22nd IEEE High Performance Extreme Computing Conf.\ (HPEC)},
  pages     = {1--7},
  year      = {2018},
  doi       = {10.1109/HPEC.2018.8547534},
  publisher = {IEEE}
}
```

---

## Traag, Waltman, van Eck (2019) — Leiden

Strictly a Leiden paper, cited in the static-Louvain-CPU section because its central theoretical contribution is a precise diagnosis of Louvain's *badly-connected community* failure mode. The paper formally distinguishes two related-but-distinct pathologies — **disconnected** communities (induced subgraph has multiple connected components) and **badly connected** communities (the weaker condition that some non-trivial split would increase modularity) — and proves that Louvain produces both, while Leiden eliminates the first by construction and the second asymptotically.

**The disconnected-community example (Section II A, Fig. 2).** Vertex $v_0$ acts as a strong bridge between two parts of community $C$ (vertices 1–3 and 4–6). When $v_0$ is moved to a different community (because its own modularity-gain calculation finds a better target), the remaining $C$ becomes internally disconnected. Crucially, vertices 1–6 may still be *locally optimally assigned* to $C$ (their move would not strictly improve modularity), so Louvain never splits $C$. After aggregation, $C$ is collapsed into a super-vertex and the disconnection becomes invisible.

**Empirical prevalence (Section IV A, Fig. 4) — paper-faithful numbers.** First-iteration Louvain on six empirical networks (DBLP, Amazon, IMDB, Live Journal, Web of Science, Web UK 2005): **up to 25 % badly connected** (Amazon: 23 %, DBLP: 16 %, Web UK: 14 %); **up to ~16 % disconnected** (Web of Science: > 5 %; most others: ~1 %). **Iterating Louvain *worsens* the disconnection problem** even though modularity increases — a counter-intuitive empirical finding emphasised in the paper.

**The Leiden three-phase loop (Section III, paper-faithful).**

1. **Local moving.** Uses a **fast local move (FLM)** procedure (citing Bae et al. and Ozaki et al.): initialise a queue with all nodes in random order; pop one node, move it if doing so improves the quality function, and on a successful move add its neighbours-not-in-the-new-community back to the queue rear; repeat until empty. The first pass through the network is identical to Louvain's; later passes visit *only* vertices whose neighbourhood has changed. This is what makes Leiden *faster* than Louvain — not, as I previously claimed, fewer iterations from refinement.
2. **Refinement.** Builds a refined partition $P_{refined}$ starting from a *singleton* partition (each node in its own community). Locally merges nodes, but **only within each community of $P$** (the partition from phase 1). Crucially: a candidate merge is accepted **with probability proportional to its modularity-gain**, with degree-of-randomness controlled by parameter $\theta$ (default $\theta = 0.01$, useful range $[0.0005, 0.1]$). Quality-decreasing moves are **not** allowed — distinguishing this from simulated annealing. Only "well-connected to their community" candidates are eligible.
3. **Aggregation — the subtle trick.** The aggregate graph topology is built from $P_{refined}$ (each refined sub-community → one super-node), **but** the *initial partition* for the next level's local-moving phase is taken from $P$ (super-nodes that came from the same original community get the same initial label). This separation gives Leiden flexibility to explore alternative partitions of a community while preserving the multilevel hierarchy.

**Guarantees (Table I in the paper) — paper-faithful.**

|  | Louvain | Leiden |
|---|---|---|
| **Per-iteration:** $\gamma$-separation (no merges improve $Q$) | ✓ | ✓ |
| **Per-iteration:** $\gamma$-connectivity | — | ✓ |
| **Stable iteration:** node optimality | ✓ | ✓ |
| **Stable iteration:** subpartition $\gamma$-density | — | ✓ |
| **Asymptotic:** uniform $\gamma$-density | — | ✓ |
| **Asymptotic:** subset optimality | — | ✓ |

Definitions paraphrased from the paper: a community is *subpartition $\gamma$-dense* if it can be partitioned into two well-connected, non-separable, recursively-dense parts. *Uniform $\gamma$-density* means no subset can be separated from the community. *Subset optimality* (the strongest property) means no subset of a community could improve $Q$ by moving to a different community — implies all weaker properties.

A subtle Louvain–Leiden contrast (Section III): in Louvain, *after* a stable iteration all subsequent iterations are also stable (no further improvement possible). In Leiden, **after a stable iteration the algorithm may still improve** in later iterations because the refinement phase keeps exploring alternative partitions.

**Hardware and benchmarks (Section IV).** 64 Intel Xeon E5-4667v3 @ 2 GHz, 1 TB RAM. Six empirical networks (Table II): DBLP (317 k nodes), Amazon (335 k), IMDB (375 k), Live Journal (4 M), Web of Science (9.8 M), Web UK 2005 (39.3 M). Implementations: Java (`CWTSLeiden/networkanalysis`) and Python (`vtraag/leidenalg`).

**Headline runtime claim:** "Leiden being up to 20 times faster than Louvain in empirical networks" (verbatim from Section IV intro). The runtime advantage comes from FLM (fast local move), *not* from the refinement phase shortening iteration count.

**For the BTP.** Two specific uses:
1. *Diagnose quality gaps.* When the BTP's GPU kernel produces a partition with lower modularity than NetworkX or cuGraph, the badly-connected / disconnected-community pathology is one explanation that is **inherent to Louvain**, not to the parallelisation. The 14–25 % first-iteration badly-connected rate from Fig. 4 is the empirical anchor for "this is a known Louvain artifact, not a BTP bug".
2. *Bound future work.* Adding a Leiden-style refinement phase to the BTP's GPU kernel is well-defined future work — but the FLM queue mechanism is harder to parallelise on GPU than Louvain's all-vertices-per-iteration sweep, so the cost-benefit is non-trivial.

Citation key: `traag2019louvain`.

```bibtex
@article{traag2019louvain,
  author  = {Traag, V. A. and Waltman, L. and van Eck, N. J.},
  title   = {From {Louvain} to {Leiden}: guaranteeing well-connected communities},
  journal = {Scientific Reports},
  volume  = {9},
  number  = {1},
  pages   = {5233},
  year    = {2019},
  doi     = {10.1038/s41598-019-41695-z},
  eprint  = {1810.08473},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SI}
}
```

---

## Sahu (2023) — GVE-Louvain

A multicore-OpenMP Louvain implementation by Subhajit Sahu (IIIT Hyderabad). The paper's claim (verbatim from the conclusion): "as far as we are aware, [GVE-Louvain] stands as the most efficient implementation of the algorithm on multicore CPUs." The "GVE" name comes from "**Graph(Vertices, Edges)**" — Sahu's intended command-line graph-processing tool of which this Louvain implementation is the first algorithm.

**Headline result (Abstract, paper-faithful).** On a server with dual 16-core Intel Xeon Gold 6226R processors, GVE-Louvain outperforms Vite, Grappolo, NetworKit Louvain, and cuGraph Louvain (the latter on NVIDIA A100 GPU) by **50×, 22×, 20×, and 5.8×** respectively. On the largest test graph (`sk-2005`, 3.80 billion edges), it finds communities in **6.8 seconds — a processing rate of 560 million edges/s**. Strong scaling: **1.6× per doubling of threads** (1.6¹×… 1.6⁵× = ~10.5× from 1 to 32 threads).

**Hardware (Section 5.1.1).** CPU experiments: dual 16-core Intel Xeon Gold 6226R @ 2.90 GHz = **32 physical cores** (64 threads with HyperThreading), 93.4 GB system memory, CentOS Stream 8, GCC 8.5 + OpenMP 4.5. GPU experiments: NVIDIA A100 (108 SMs, 80 GB memory, 1935 GB/s bandwidth) + AMD EPYC-7742 (64 cores, 2.25 GHz), 512 GB DDR4, Ubuntu 20.04, GCC 9.4 + OpenMP 5.0 + CUDA 11.4.

**Test graphs (Section 5.1.3, Table 2).** **Thirteen graphs from the SuiteSparse Matrix Collection**, organised into four categories — *not* "the LAW / SNAP suite":
- **Web graphs (LAW):** indochina-2004, uk-2002, arabic-2005, uk-2005, webbase-2001, it-2004, sk-2005 (largest: 50.6 M vertices, 3.80 B edges)
- **Social (SNAP):** com-LiveJournal, com-Orkut
- **Road (DIMACS10):** asia_osm, europe_osm
- **Protein k-mer (GenBank):** kmer_A2a (171 M vertices), kmer_V1r (214 M vertices)

### Foundational design choice — Asynchronous parallel Louvain (§4.1 preamble)

Before the nine numbered knobs, §4.1 establishes a foundational design decision that the rest of the paper builds on, and that the abstract and introduction both call out by name: GVE-Louvain runs the **asynchronous version of Louvain**, i.e. **threads work independently on different parts of the graph without barrier-synchronising at the end of each local-move sweep**. Vertex community labels are read and written without locks; a thread that moves vertex $u$ updates the per-community $\Sigma_{tot}$ counters with `atomic` directives (Algorithm 2, line 11), but every other thread sees that update immediately, *not* at the next iteration boundary. There is no double-buffered "old labels / new labels" array.

The paper is explicit about the tradeoff (verbatim, §4.1): "*This allows for faster convergence but can also lead to more variability in the final result*", citing Blondel et al. and Halappanavar et al. (Grappolo) — i.e. the synchronous-vs-asynchronous question is a known design fork in the literature, and Sahu picks asynchronous for speed and accepts the run-to-run modularity noise (which Figure 3 then bounds at < 1 %, well within the noise of any of the 9 knobs).

This choice is what makes the rest of the optimisation stack coherent:

- **Per-thread hashtables can be allocated once and reused** across iterations (no need to merge results across threads at a barrier).
- **Vertex pruning (4.1.6) becomes meaningful**: a thread that re-processes a vertex marked unprocessed by *another* thread sees the most recent label assignment, so the prune flag is consistent with the live community state.
- **The Far-KV insight (4.1.9) only matters under asynchrony** — under a synchronous bulk-synchronous design, threads would wait at a barrier and false cache-sharing would amortise away.
- **The 1.6× per-doubling strong-scaling result (Section 5.4)** is bounded above by Amdahl on the *non-asynchronous* parts (renumbering, dendrogram lookup); the asynchronous local-moving phase itself scales near-ideally up to one socket.

The asynchronous-vs-synchronous tradeoff is also one of the open questions for any GPU port (the BTP): GPU local-moving kernels almost always go asynchronous-by-default because barrier-synchronising 100 K threads is ruinous, but the resulting non-determinism complicates correctness testing. Sahu's paper is the canonical CPU-side citation for *"asynchrony costs you reproducibility, buys you wall-clock"*.

### The nine optimisations — quantitative breakdown (Figure 3, Table-form)

On top of the asynchronous-threads foundation, the paper's most distinctive methodological feature is the *empirical measurement of nine separate optimisation knobs*, each tested with multiple alternatives on all 13 graphs, 5 runs each, with relative runtime and modularity reported. This is unusual rigor for a parallel-algorithms paper. The nine knobs and their measured impacts:

| # | Optimisation | Best setting | Measured improvement |
|---|---|---|---|
| 1 | OpenMP loop schedule | `dynamic` | 7% faster than `auto`, 0.4% modularity loss (likely noise) |
| 2 | Iteration cap per pass | 20 | 13% faster than 100 |
| 3 | Threshold-scaling tolerance drop rate | 10× per pass | 4% faster than no scaling, no quality loss |
| 4 | Initial tolerance | 0.01 | 14% faster than 10⁻⁶, no quality loss |
| 5 | Aggregation tolerance | 0.8 | 14% faster than disabled (=1), equivalent quality |
| 6 | Vertex pruning | enabled | 11% faster |
| 7 | Community vertices CSR vs 2D arrays | preallocated CSR + parallel prefix sum | **2.2× faster** |
| 8 | Super-vertex graph CSR vs 2D arrays | preallocated CSR + parallel prefix sum | **2.2× faster** |
| 9 | Hashtable design (Close-KV vs Far-KV vs Map) | **Far-KV** | **4.4× faster than `std::map`, 1.3× faster than Close-KV** |

**Cumulatively, these nine optimisations are why GVE-Louvain beats Grappolo by 22× on the same graph suite** despite both being multicore parallel Louvain implementations. The optimisations are surgical: each is a small change with a clearly-bounded measurable impact.

### The "Holey CSR" trick (Sections 4.1.7–4.1.8)

The cleanest novel data-structure idea in the paper. Standard parallel CSR construction needs to know each row's length before allocation; this is hard in the aggregation phase because the *unique* neighbouring communities per super-vertex aren't known until the work is done. The paper's solution:

1. **Over-estimate** each super-vertex's degree as the *total degree of all vertices in its community* (which is an upper bound, but easy to compute).
2. **Parallel exclusive scan** that over-estimate to get CSR offsets — zero-allocation, fully parallel.
3. Iterate over all original-graph edges in parallel; for each edge, look up the source/destination super-vertices; **atomically** insert the (super-vertex, edge-weight) pair into the corresponding super-vertex's CSR row.
4. The result has *gaps* (the row was sized to the over-estimate, but unique-community count is lower) — hence "Holey CSR". Subsequent reads tolerate the gaps.

This trades 2× memory waste for **2.2× speedup** vs. the 2D-array alternative, and — critically — eliminates *any* memory (re)allocation during the algorithm's hot path.

### The Far-KV insight in more detail (Section 4.1.9, Figure 2)

The most distinctive contribution. Each thread's hashtable has three pieces:
- A **keys vector** (community IDs encountered while scanning the current vertex's neighbours)
- A **values vector of size $|V|$** (full-size, indexed by community ID; stores accumulated edge weight)
- A **key count** (how many communities have been touched this vertex)

The collision-free part: because the values array is `|V|`-sized, every community ID hashes to its own slot — no collisions, no probing.

The Far-KV part: NetworKit allocates these per-thread tables in *contiguous* memory (Sahu calls this **Close-KV**). Even though each thread only writes to its own table, the cache lines storing different threads' tables overlap, causing **false cache-sharing**: a write by thread $i$ invalidates a cache line used by thread $j$, even though they touch disjoint logical data. Sahu's fix: allocate each thread's hashtable **far apart in memory** (separate heap allocations, or aligned to cache-line boundaries × thread count). The key count is *also* allocated separately on the heap because it is updated frequently.

**Reported speedups: 4.4× over `std::map`, 1.3× over Close-KV.** The `1.3×` over Close-KV — which itself uses the same data structure, just placed differently — *is purely a memory-layout effect*. This is the kind of subtle false-cache-sharing bug that doesn't show up in algorithmic analysis but kills real performance.

### Phase-and-pass breakdown (Section 5.3, Figures 6–7)

GVE-Louvain's runtime profile, averaged over the 13-graph suite:
- **Phase split:** 49% local-moving, 35% aggregation, 16% other (initialisation, renumbering communities, dendrogram lookup, resetting)
- **Pass split:** **67% of total runtime is in the first pass** — because the original graph is at full size; subsequent passes work on much smaller super-vertex graphs

Per-graph qualitative observations (Figure 6, Figure 7):
- *Web graphs / road / k-mer:* local-moving phase dominates
- *Social networks:* aggregation phase dominates (because they have lots of small communities → many aggregation buckets)
- *Higher runtime/|E| ratios* on lower-degree graphs (road, k-mer) and on graphs with poor community structure (com-LiveJournal, com-Orkut). This is a meaningful predictor: dense, well-clustered graphs are *easier* per edge.

### Strong scaling (Section 5.4, Figure 8)

- 1 → 32 threads: **10.4× speedup** (consistent with the 1.6× per doubling claim)
- 32 → 64 threads: only **11.4×** total — i.e. doubling threads adds only ~10% more performance. The paper explicitly attributes this to **NUMA effects** on the dual-socket server (cross-socket memory traffic).
- The bottleneck above 32 threads is also "various sequential steps/phases in the algorithm" — community renumbering and dendrogram lookup are listed as the residual sequential parts.

### Quality comparison (Section 5.2, Figure 5(c)) — refined numbers

GVE-Louvain modularity vs. each baseline (paper-faithful averages):
- **3.1% higher** than Vite (especially on web graphs where Vite under-clusters)
- **0.6% lower** than Grappolo and NetworKit — concentrated on social networks with poor clustering (com-LiveJournal, com-Orkut)
- **2.6% higher** than cuGraph Louvain — but largely because cuGraph fails on the well-clusterable web graphs (out-of-memory on 5 of 13 graphs); on the graphs where cuGraph runs, the gap is smaller

### A precise Louvain-vs-LPA tradeoff statement (Section 3.3)

Sahu provides a rare quantitative comparison of Louvain to LPA: "Louvain obtains high-quality communities, with **3.0–30% higher modularity than that obtained by LPA**, but requires **2.3–14× longer to converge**." This is a direct empirical tradeoff statement that the BTP can cite when defending the Louvain-over-LPA choice — and that most Louvain papers leave implicit.

### Headline implication and its follow-up

GVE-Louvain on a 32-core CPU outperforms cuGraph on an A100 GPU by **5.8×** on the same graphs. This is the central provocation that motivated Sahu's follow-up paper (`sahu2025cpuvsgpu`, the ν-Louvain comparison): if a well-tuned multicore CPU beats a top-tier GPU on Louvain, what's the point of GPU Louvain? The 2025 follow-up answers that even Sahu's own GPU implementation (ν-Louvain) only matches GVE-Louvain in performance — see the Sahu CPU-vs-GPU 2025 entry in `static_louvain_gpu.md`.

### Software and tooling

[github.com/puzzlef/louvain-communities-openmp](https://github.com/puzzlef/louvain-communities-openmp). The dynamic-version sibling (`louvain-communities-openmp-dynamic`) used in the BTP is Sahu's later DF-Louvain (`sahu2024df`). The Leiden equivalent is GVE-Leiden (`sahu2023gveleiden`).

### For the BTP

1. **Mandatory CPU baseline.** No 2024-or-later Louvain paper can omit GVE-Louvain.
2. **The asynchronous-threads design is the foundational citation** for the BTP's GPU kernel, which is also asynchronous-by-default (no per-iteration barrier, atomic writes to community-aggregate counters). Sahu's explicit framing — "*faster convergence but … more variability in the final result*" — is the CPU-side precedent that justifies the BTP's choice not to chase bit-reproducible results.
3. **The Far-KV insight is the most portable data-layout contribution.** Even though the BTP's GPU has a different memory model, the broader principle — *separate per-thread/per-warp state, communicate only at sync points, watch out for cache-line interference* — is directly applicable. On GPU the analogue is per-block shared-memory hash tables with explicit padding to avoid bank conflicts.
4. **The Holey-CSR trick is also portable** — it solves the same "don't know the row length until done" problem that the BTP's GPU aggregation kernel faces.
5. **The 67% first-pass observation** sets a clear ceiling: any optimisation that doesn't accelerate the first pass is at most a 33% win.
6. **The methodology is the model to copy.** The 9-knob × 5-alternatives × 5-runs × 13-graphs evaluation grid is the kind of empirical rigor the BTP's evaluation should aspire to.

Citation key: `sahu2023gve`.

```bibtex
@article{sahu2023gve,
  author  = {Sahu, Subhajit},
  title   = {{GVE-Louvain}: Fast {Louvain} Algorithm for Community Detection in Shared Memory Setting},
  journal = {arXiv preprint arXiv:2312.04876},
  year    = {2023},
  note    = {Latest arXiv version v6 (5 Aug 2024). Formal publication: in \emph{Complex Networks \& Their Applications XIII (COMPLEX NETWORKS 2024)}, Studies in Computational Intelligence vol. 1189, pp. 127--139, Springer, doi:10.1007/978-3-031-82435-7\_11. Software: \url{https://github.com/puzzlef/louvain-communities-openmp}.},
  url     = {https://arxiv.org/abs/2312.04876}
}
```
