# Dynamic Louvain — CPU

Writing material for the "Dynamic Louvain (CPU)" paragraph(s) of the Related Work section. Dynamic Louvain takes a graph that has changed (edge insertions, deletions, weight updates) and produces an updated community assignment without re-running Louvain from scratch on the new graph. Each entry below is a paper, a discussion of what it contributes, and the LaTeX citation key + BibTeX entry.

---

## Aynaud, Guillaume (2010) — Stabilized Louvain for Evolving Networks

The paper most often cited as "the earliest naive-dynamic Louvain". **The paper's own framing is different**, and worth being honest about: the explicit goal is **community-tracking stability** (so that "community 17" remains identifiable across snapshots) — *not* speed. The fact that the same modification later became the speed-oriented "warm-start dynamic Louvain" baseline is a downstream re-purposing.

**The actual contribution (paper-faithful, Sections II–III).**

1. **Empirical instability finding.** Removing a *single* random node from the arxiv co-authorship network (9 377 nodes / 24 107 edges) causes Louvain to move **1 500–3 000 nodes** in its partition (and Fast Greedy moves 2 000–3 000). Walktrap is more stable but still moves ~500. *None of the three static algorithms is suitable for tracking communities across snapshots due to this instability.*

2. **Stabilised Louvain (Section III).** Instead of starting each Louvain run with the singleton partition (every node in its own community), use the **previous snapshot's partition** as the initial partition. Then run the standard Louvain phases unchanged.

3. **Stability-vs-quality parameter $x$ (Section III-B).** The fully-stabilised algorithm sometimes over-constrains the partition. Knob: randomly select $x\%$ of nodes and put them alone in their own community at initialisation; the other $(100-x)\%$ get the previous partition. $x = 0\%$ = fully stabilised; $x = 100\%$ = classical Louvain. **$x = 2$–$5\%$** is the empirical sweet spot.

4. **Real-world evaluation.** A network of ~5 000 blogs monitored over 4 months — 120 daily snapshots. Each snapshot's vertex set = blogs with at least one post by day $t$; edges = inter-blog citation links from posts in $[0, t]$. Largest connected component used.

**What the paper does *not* claim** (corrections to the prior version of this entry):
- ❌ "5–20× faster" — the paper does not give speedup numbers. Its motivation and metrics are stability and modularity, not runtime.
- ❌ "modularity within 1–2 % of from-scratch on temporal email networks" — the paper uses arxiv (single-node-removal stability test) and a blog network (real dynamic), not email networks. The reported modularity gap between stabilised and classical Louvain is small but graph-dependent (Figure 4(b)).
- ❌ "warm-starting locks in worse local optima when graphs change a lot" — paper does discuss the trade-off (constrained initialisation can be too restrictive when the network changes substantially), but the framing is "increase $x$ when snapshots are similar" rather than "warm-starting is pathological".
- ❌ Page numbers 513–519 — actual is **508–514** (per the HAL bibliographic record).

**For the BTP.** This is the citation for the **stabilised-initialisation idea** that later evolved into "naive-dynamic Louvain" used as a speed baseline. When citing, be careful to credit this paper for the *technique* (initialise from previous partition) but use the right framing — the paper is about *tracking stability*, and only later authors (Sahu DF Louvain 2024, Halappanavar 2017) present it as a speed baseline. The "naive-dynamic" terminology comes from the latter community.

Citation key: `aynaud2010static`.

```bibtex
@inproceedings{aynaud2010static,
  author    = {Aynaud, Thomas and Guillaume, Jean-Loup},
  title     = {Static Community Detection Algorithms for Evolving Networks},
  booktitle = {Proceedings of the 8th International Symposium on Modeling and Optimization in Mobile, Ad Hoc, and Wireless Networks (WiOpt'10)},
  address   = {Avignon, France},
  pages     = {508--514},
  year      = {2010},
  note      = {HAL: inria-00492058}
}
```

*(Affiliations: both authors at LIP6, CNRS, Université Pierre et Marie Curie, Paris.)*

---

## Cordasco, Gargano (2010/2011) — Semi-synchronous Label Propagation

> ⚠️ **Audit note: this paper is *not* about dynamic graphs.** My earlier framing of it as a "label-propagation-based alternative to dynamic Louvain" with "edge-change-triggered local re-propagation within distance $k$" was a fabrication — none of that is in the paper. The paper presents a *static-graph* LPA variant; the only relevance to dynamic Louvain is by analogy, not by mechanism.

**What the paper actually contributes (paper-faithful, Sections III–V).** Cordasco and Gargano introduce a **semi-synchronous** variant of Raghavan et al.'s Label Propagation Algorithm (LPA), positioned as a third option between:

- **Synchronous LPA** (Algorithm 1): every vertex updates its label in parallel using neighbours' labels at step $i-1$. Embarrassingly parallel, but can produce *cyclic label oscillations* (especially on bipartite-like subgraphs) and may not terminate.
- **Asynchronous LPA** (Algorithm 2): vertices updated sequentially in random permutation; each uses the freshest available neighbour labels. Avoids oscillation but cannot be parallelised straightforwardly because of read-after-write dependencies.

**The semi-synchronous mechanism (Section V).** Pre-compute a graph $\chi$-colouring (any vertex colouring with $\chi(G)$ colours, using a parallelisable algorithm such as Boman-Bozdağ — *not* limited to 2-colouring as I had previously claimed). Within each color class, no two vertices are adjacent, so they can update *synchronously* without oscillation; iterate over colour classes sequentially. The paper proves this always converges (Theorem in Section V).

**Test graphs and metrics (Section VI).** Real-world graphs from the standard community-detection benchmark suite (Karate, Football, Polbooks, etc.) — *not* LFR-specific. Comparison is against synchronous and asynchronous LPA on modularity, runtime, and stability — *not* against Louvain (the paper does not benchmark Louvain).

**Why this paper does *not* belong straightforwardly in a "Dynamic Louvain — CPU" section.** Cordasco-Gargano's algorithm is a *static-graph* LPA. There is no edge-change-triggered re-propagation, no neighbourhood-bounded update rule, and no notion of snapshots. Including it in this section was based on my fabricated reading. The honest justification for keeping it as a *peripheral* reference: the colour-class-synchronous parallelism pattern is a known pattern in parallel community detection (also used in Grappolo for parallel-safety), and any future BTP work on parallel LPA-style methods should cite this paper for that pattern.

**Bibliographic confusion (verified):** there are multiple Cordasco-Gargano papers in this area. The arXiv preprint I have is **arXiv:1103.4550** (March 2011) titled "Community Detection via Semi-Synchronous Label Propagation Algorithms". The IEEE conference version was published in *Proc. IEEE BASNA 2010* (Business Applications of Social Network Analysis), and a later journal version appears in *International Journal of Social Network Mining* 1(1), pp. 3–26 (2012) titled "Label Propagation Algorithm: A Semi-Synchronous Approach". My earlier BibTeX cited the 2012 journal version with an MUSE workshop; the venue was BASNA 2010, then republished in IJSNM 2012.

Citation key: `cordasco2011semisync`.

```bibtex
@article{cordasco2011semisync,
  author  = {Cordasco, Gennaro and Gargano, Luisa},
  title   = {Community Detection via Semi-Synchronous Label Propagation Algorithms},
  journal = {arXiv preprint arXiv:1103.4550},
  year    = {2011},
  note    = {Conference version at IEEE BASNA 2010; journal version at Int. J. Social Network Mining 1(1):3--26, 2012, titled "Label Propagation Algorithm: A Semi-Synchronous Approach". Authors at Dipartimento di Informatica, University of Salerno, Italy.}
}
```

---

## Shang, Liu, Xie, Chen, Miao, Fang, Wu (2012) — Real-Time Algorithm for Tracking Community Structure

> ⚠️ **My earlier BibTeX cited the wrong paper entirely** — I had "Targeted revision: A learning-based approach... Physica A 2016" attached to this entry, which is a *different* Shang et al. paper. The arXiv preprint downloaded (1407.2683, July 2014) is the SNA-KDD '12 workshop paper "A Real-Time Detecting Algorithm for Tracking Community Structure of Dynamic Networks", not the Physica A paper. Author list, year, and venue are now corrected.

**What the paper actually does (paper-faithful, Section 4).** This is an **incremental modularity-optimisation** algorithm, not a community-evolution-event tracker. The mechanism:

1. Use **BGL** (Blondel-Guillaume-Lambiotte = Louvain) to compute an initial community partition on the snapshot-0 graph.
2. As edges are added one at a time, classify each edge into one of **four types**:
   - *Inner-community edge:* both endpoints exist and are in the same community.
   - *Cross-community edge:* both endpoints exist but are in different communities.
   - *Half-new edge:* one endpoint is a new vertex.
   - *New edge:* both endpoints are new vertices.
3. Apply a different community-update strategy per edge type, designed to greedily increase modularity (or minimise its loss).

Per-edge update cost: $O(1)$ in the typical case, $O(S)$ in the worst case ($S$ = size of community to update).

**My prior fabrications (now removed):**
- ❌ "Five event types: birth, death, merge, split, survival" — this taxonomy belongs to **Greene et al. 2010**, not Shang et al. (Greene is reference [24] in this paper.)
- ❌ "Jaccard overlap > 0.5 detection rule with configurable thresholds" — fabricated; not in the paper.
- ❌ "On Twitter retweet networks, ~80% of communities survive" — fabricated. Paper uses **Enron email** + 3 other real-world networks (paper citation, web voting, etc.), *not* Twitter.

**Test datasets (Section 5).** Enron Email + 3 other real-world networks (paper citation, web voting). Comparison baselines: BGL (Louvain) and CNM (Clauset-Newman-Moore).

**Headline results (Section 5):** Outperforms BGL and CNM in computing time; outperforms CNM in modularity.

**For the BTP.** Cite as one of the early per-edge incremental Louvain-based dynamic algorithms (2012 vintage). The four-edge-type taxonomy is a useful framing for *what update strategies to apply for which edge change* — directly comparable to the BTP's per-batch handling. **Note:** the Shang et al. 2014 *Physica A* paper ("Targeted revision: A learning-based approach for incremental community detection in dynamic networks", DOI 10.1016/j.physa.2015.09.039) is a *separate, later* paper by the same group; if the BTP needs that work cited, it requires a different BibTeX entry.

Citation key: `shang2012realtime`.

```bibtex
@inproceedings{shang2012realtime,
  author    = {Shang, Jiaxing and Liu, Lianchen and Xie, Feng and Chen, Zhen and Miao, Jiajia and Fang, Xuelin and Wu, Cheng},
  title     = {A Real-Time Detecting Algorithm for Tracking Community Structure of Dynamic Networks},
  booktitle = {Proceedings of the 6th SNA-KDD Workshop (SNA-KDD '12)},
  address   = {Beijing, China},
  year      = {2012},
  month     = {August},
  eprint    = {1407.2683},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SI}
}
```

*(Affiliations: Department of Automation and National CIMS Engineering Center, Tsinghua University; Institute of Command Automation, PLA University of Science and Technology; Institute of Computer Science, National University of Defense Technology, China.)*

---

## Cordeiro, Sarmento, Gama (2016) — Dynamic CD via Locality Modularity Optimization

**Author order corrected**: the first author is **Mário Cordeiro** (FEUP, Porto University), not Sarmento. Co-authors: Rui Portocarrero Sarmento and João Gama (LIAAD - INESC TEC, FEP, Porto). Published in *Social Network Analysis and Mining* 6(1):15 (2016), DOI 10.1007/s13278-016-0325-1.

**What the paper actually does (Sections 3.5 and 4, paper-faithful).** Modifies the standard Louvain algorithm so that, in subsequent snapshots of an evolving network, **only those *communities* affected by added/removed edges or nodes are re-optimised**, with the rest of the network kept unchanged. The unit of locality is the **community**, not a vertex-neighbourhood radius — my earlier "two-hop neighbourhood $S = \{u,v\} \cup N(u) \cup N(v) \cup N(N(u)) \cup N(N(v))$" formulation was fabricated.

**The edge-type taxonomy (Section 3.5).** Cordeiro et al. adopt the same four edge-type classification that Shang et al. (2012) introduced (cited as their reference [15]):
- *Cross-community edge* (both endpoints exist, different communities) → Op1 (no change) or Op2 (merge)
- *Inner-community edge* (both endpoints exist, same community) → Op1
- *Half-new edge* (one endpoint new) → Op3 (assign to existing community) or Op4 (new community)
- *New edge* (both endpoints new) → Op4

Plus four removal operations (Op5–Op8): keep unchanged, split community, remove terminal node, remove 2-node community.

**Three explicit goals (Section 1, paper's own framing):**
1. *Optimization* — full-network CD only at first snapshot; subsequent snapshots use Louvain on a much smaller aggregated network.
2. *Efficiency* — empirically keeps most communities of the previous snapshot unchanged; only 2–3 iterations needed to converge on the modified parts.
3. *Stability* — unaffected communities preserve same nodes and **same community ID** across snapshots (which solves the community-tracking problem that motivated Aynaud-Guillaume's work — see prior entry).

**Comparison baselines (Section 4 onward):** standard Louvain run from scratch each snapshot; LabelRank; LabelRankT; GANXiSw; AFOCS. Metrics: Newman's Modularity, Modularity with Split Penalty, Modularity Density, # detected communities, runtime. (My earlier text claimed comparison to "Zarayeneh delta-screening" — paper does not benchmark against delta-screening; the only screen-related work cited is the Shang et al. edge-classification.)

**My fabrications (now removed):**
- ❌ "Two-hop neighbourhood $S = \{u,v\} \cup N(u) \cup N(v) \cup N(N(u)) \cup N(N(v))$" — fabricated; locality unit is the *community*, not vertex-distance-2.
- ❌ "Local modularity $Q_L(S)$ as boundary-normalised contribution" — fabricated formalisation; paper uses standard modularity, just restricted to affected communities.
- ❌ "~95% of edge changes affect fewer than 50 vertices in their two-hop neighbourhood" — fabricated specific.
- ❌ "Comparison to delta-screening" — paper does not engage with the delta-screening line of work (which appeared 3 years later).

**For the BTP.** Cite as the locality-by-affected-community precursor to delta-screening (which uses locality-by-vertex-modularity-bound). The BTP's frontier kernel is closer to delta-screening's vertex-set definition than to Cordeiro's community-set definition. The shared insight: when an edge changes, *most* of the network's partition does not need to be touched.

Citation key: `cordeiro2016dynamic`.

```bibtex
@article{cordeiro2016dynamic,
  author  = {Cordeiro, M{\'a}rio and Sarmento, Rui Portocarrero and Gama, Jo{\~a}o},
  title   = {Dynamic community detection in evolving networks using locality modularity optimization},
  journal = {Social Network Analysis and Mining},
  volume  = {6},
  number  = {1},
  pages   = {15},
  year    = {2016},
  doi     = {10.1007/s13278-016-0325-1},
  publisher = {Springer-Verlag Wien}
}
```

---

## Seifikar, Farzi, Barati (2020) — C-Blondel

> ✅ **Now verified from the PDF** (IEEE TCSS 7(2):308–318, Apr 2020). This replaces the earlier no-PDF placeholder. Correction: the "destructive node" rule is a degree-centrality threshold $d_u \ge \alpha\,\bar{d}_{C_u}$ — *not* the "50% intra-community degree fraction" previously guessed.

### Introduction
Dynamic community detection tracks community evolution across network snapshots. The paper splits prior work into two groups: (1) run a static algorithm on every snapshot and map communities between consecutive ones — high modularity but slow; (2) reuse the previous snapshot's communities + historical information to cut runtime. **C-Blondel is in group 2**: it runs Louvain over a *compressed graph* derived from the previous snapshot, which is far smaller than the full graph, so it is faster than re-running Louvain.

### Novelty / Contributions
1. **Compressed-graph Louvain ("C" = Compressed).** Build a compressed graph $G_t^H$ whose **supernodes are the previous snapshot's communities** (or sub-communities) and **superedges are the inter-community edges**, then run Louvain on it. Since $|V|\gg|V^H|$ and $|E|\gg|E^H|$, Louvain on $G_t^H$ is cheap.
2. **Destructive-node heuristic.** Only communities holding a **destructive node** are re-optimised. A node is destructive (its removal can "blow up" / split its community) if its degree is high relative to its community: $d_u \ge \alpha\,\bar{d}_{C_u}$, where $\bar{d}_{C_u}$ is the community's average degree and **$\alpha$ (the "destruction parameter") is the algorithm's only parameter**. Lemma 1 formalises the blow-up condition.
3. **Unified change model.** All edits (appearing/disappearing nodes and edges) are reduced to **"remove a node from its community"** actions; only destructive removals trigger a split. Unlike **D-Blondel** (He et al. 2017), which computes the "tendency" of *all* nodes, C-Blondel computes it only for *destructive* nodes → faster.

### Algorithm
For snapshot $t$, instead of running Louvain on the full $G_t$:
- **Algorithm 1.** `ConstructCompressedGraph(G_{t-1}, G_t, C(G_{t-1}))` → $G_t^H$; then `C(G_t) ← Louvain(G_t^H)`.
- **Algorithm 2 (build compressed graph).** `PullOutChanges` extracts the edits; each is handled via `RemoveNode`:
  - *Remove node $u$* → `RemoveNode(u, C_u)`.
  - *Remove edge $(u,v)$* → if same community, `RemoveNode` both endpoints; if cross-community, the removal only *raises* modularity, so just lower the weight.
  - *Add node $u$* → add as a supernode and `RemoveNode` each neighbour from its community.
  - *Add edge $(u,v)$* → if different communities, `RemoveNode` both; if inner edge, do nothing.
- **Algorithm 3 (`RemoveNode`).** If $d_u \ge \alpha\,\bar{d}_{C_u}$ (destructive), run Louvain on $C_u$ and turn each resulting sub-community into a supernode; otherwise keep $C_u$ as a single supernode. So **only destructive nodes cause a community to be split and re-optimised** — everything else stays aggregated.
- **Complexity (Eq 12):** $O(|V^H|\log|V^H|) + O(2|\Delta G|\cdot n_r^c\log n_r^c)$ — Louvain on the small compressed graph plus the change-processing cost.

### More interesting points
- **The destruction parameter $\alpha$ is a single quality–speed knob.** Higher $\alpha$ → fewer destructive nodes → more nodes folded into supernodes → smaller compressed graph → **faster but lower modularity**; lower $\alpha$ → larger graph → **slower but higher modularity**. Empirically the **balance is near $\alpha\approx0.6$** (best modularity at $\alpha\approx0.1$, best speed at $\alpha=1$).
- $\alpha$ is **graph-dependent**: the paper ties it to the power-law degree exponent (estimated 4.17 for Cit-HepTh, 4.45 for Facebook, 3.05 for Enron); on Enron the behaviour is flat in $\alpha$ because 88.5% of nodes are below the average degree.
- The compressed-graph + destructive-node design means **unchanged communities are never touched** — the same instinct as Cordeiro (keep unaffected communities) but the *trigger* is node influence (degree) rather than a modularity-gain predicate.

### Baselines (graphs and baseline algorithms)
**Baseline algorithms:** **S-Blondel** (Greene et al. 2010 [16] — run Louvain on the *full original* graph each snapshot, map via Jaccard) and **D-Blondel** (He et al. 2017 [1] — division+agglomeration Louvain on a compressed graph, computing tendency of *all* nodes). Both Louvain-based. Metrics: **modularity** (quality), **execution time** (efficiency), number of communities; 30 runs averaged.

**Datasets (Table I; 31 / 31 / 15 snapshots over Jan 1993 – Apr 2003):**
| Dataset | Nodes | Edges | Avg deg | cc |
|---|---|---|---|---|
| Cit-HepTh (arXiv citations) | 21,550 | 201,099 | 18.66 | 0.292 |
| Enron Email | 83,910 | 325,526 | 7.70 | 0.402 |
| Facebook (New Orleans) | 61,096 | 614,797 | 14.62 | 0.218 |

**Results:** C-Blondel is **faster than S-Blondel and D-Blondel** at moderate-to-high $\alpha$ (on Cit-HepTh the speed-up reaches ~5× at $\alpha=1$, Fig 2), with **modularity comparable** to both (within ~0.02). At low $\alpha$ it is slower but slightly higher modularity. The headline is **execution-time superiority at comparable modularity**, tunable via $\alpha$.

### Takeaways
- A 2020 **CPU dynamic Louvain** that keeps unchanged communities **aggregated as supernodes** and only **splits/re-optimises communities holding a high-degree "destructive" node** ($d_u\ge\alpha\bar d_{C_u}$) — another member of the "only touch the affected part" incremental-Louvain family.
- **Distinct affected-set criterion:** unlike delta-screening (modularity-gain predicate) or DF-Louvain (frontier expansion), C-Blondel bounds the work by **node degree/influence**. Worth contrasting in the BTP's related-work as a different way of choosing what to recompute.
- The **compressed-graph reuse** (run Louvain on previous communities, disband only the affected ones) is conceptually close to Cordeiro's two-level scheme and to the BTP's incremental aggregation.
- **Vs the BTP:** CPU/sequential, needs the graph-dependent parameter $\alpha$ (tied to the power-law exponent), and modularity degrades at high $\alpha$. The BTP is GPU-parallel and uses screening / frontier marking with no such tuning knob.

Citation key: `seifikar2020cblondel`.

```bibtex
@article{seifikar2020cblondel,
  author  = {Seifikar, Mahsa and Farzi, Saeed and Barati, Masoud},
  title   = {{C-Blondel}: An Efficient {Louvain}-Based Dynamic Community Detection Algorithm},
  journal = {IEEE Transactions on Computational Social Systems},
  volume  = {7},
  number  = {2},
  pages   = {308--318},
  year    = {2020},
  doi     = {10.1109/TCSS.2020.2964197},
  note    = {Software: \url{https://github.com/MahsaSeifikar/CBlondel}}
}
```

---

## Zarayeneh, Kalyanaraman (2019) — Δ-screening (arXiv preprint)

The first publication of the **Δ-screening** technique. Author given name corrected: **Neda** Zarayeneh, not "Naw" (which was a typo in the prior BibTeX). Both authors at the School of EECS, Washington State University, Pullman.

**Critical scope restriction (Section III).** The paper considers **growing dynamic graphs only** — vertices and edges may be **added** but **not deleted**: $V_t \supseteq V_{t-1}$ and $E_t \supseteq E_{t-1}$. My earlier text claimed the technique handles "edge insertion/deletion/weight-change" with a special tighter bound for deletions; that was fabricated. Deletions are out of scope for this paper.

**The actual Δ-screening algorithm (Algorithm 2, paper-faithful).**

```
Input: G_t (current graph), Δt (newly added edges since t-1)
Output: R_t (subset of vertices to re-evaluate)

R_t ← ∅
Sort edges in Δt by source
For each source vertex i in S:
    Let j = argmax_{j' ∈ T(i)} ΔQ_{i → C_{t-1}(j')}    // i's best move target among new neighbours
    gain1 = ΔQ_{i → C_{t-1}(j)}                        // i moves to j's previous community
    gain2 = ΔQ_{j → C_{t-1}(i)}                        // j moves to i's previous community
    If gain1 ≥ gain2 AND gain1 > 0:
        R_t ← R_t ∪ {i, j} ∪ Γ(i) ∪ C_{t-1}(j)
return R_t
```

So when a new edge $(i, j)$ is added and $i$'s migration to $j$'s community is at least as attractive as the reverse direction *and* yields positive modularity gain, **R_t is augmented with i, j, all neighbours of i, and all vertices in j's previous community**. Two correctness lemmas (3.2, 3.4) prove that vertices outside this set cannot benefit from moving — they can safely keep their previous community labels.

**The four lemmas (Section III.E):**
- *Lemma 3.1:* Any neighbour of $i$ may be impacted → include in $R_t$.
- *Lemma 3.2:* Vertices in $C_{t-1}(i)$ that are *not* neighbours of $i$ have no incentive to move → exclude.
- *Lemma 3.3:* Vertices in $C_{t-1}(j)$ may be impacted → include.
- *Lemma 3.4:* Vertices outside both communities and not in $\Gamma(i)$ cannot be incentivised → exclude.

**Implementations.** Δ-screening incorporated into Louvain (called **dLouvain-S**) and SLM (called **dSLM-S**). Sequential C++ implementations. Compared against:
- Static Louvain / SLM (rerun from scratch each snapshot)
- Incremental baseline without screening (dLouvain-base, dSLM-base)

**Test inputs (Table I).**
- *Synthetic:* MIT Graph Challenge 2018 streaming networks; 50K and 5M node sizes; "ll" (low block overlap, low size variation) and "hh" (high overlap, high size variation) variants; 10 time steps.
- *Real-world:* Arxiv HEP-TH (27 770 vertices, 352 807 edges, 11 yearly snapshots 1993–2003); sx-stackoverflow (2 601 977 vertices, 63 497 050 temporal edges, multiple snapshots).

**Empirical effectiveness of screening (Section IV.B, Figure 3).** $|R_t|/|V_t|$ varies from **under 10 % on real-world inputs** in some time steps to **up to 100 % on synthetic inputs**. The paper attributes the gap to *locality*: real-world graph changes are spatially clustered, while the synthetic generator's edge-sampling adds edges almost randomly. (My earlier "99 % on power-law, 50 % on regular" was fabricated specific numbers; the paper's actual range is "under 10 %" to "100 %", varying by graph and time-step.)

**Headline runtime claim.** sx-stackoverflow (63 M temporal edges, 12 time steps): **1056 seconds total, 3× speedup over baseline**.

**For the BTP.** This is the originating reference for the Δ-screening technique — the UGRC-I report cites this version. The BTP's GPU delta-screening kernel should re-implement Algorithm 2 directly. Note the **growing-graph-only** restriction; if the BTP needs deletion support, it must use a different algorithm or extend the technique (the 2021 journal version, `zarayeneh2021delta`, may extend the scope — to verify).

Citation key: `zarayeneh2019delta`.

```bibtex
@article{zarayeneh2019delta,
  author  = {Zarayeneh, Neda and Kalyanaraman, Ananth},
  title   = {A Fast and Efficient Incremental Approach toward Dynamic Community Detection},
  journal = {arXiv preprint arXiv:1904.08553},
  year    = {2019},
  url     = {https://arxiv.org/abs/1904.08553},
  note    = {Earlier version of the IEEE TNSE 2021 paper. Authors at School of EECS, Washington State University.}
}
```

---

## Zarayeneh, Kalyanaraman (2021) — Delta-Screening (IEEE TNSE journal version)

The extended journal version of `zarayeneh2019delta`, published in *IEEE Transactions on Network Science and Engineering* (DOI 10.1109/TNSE.2021.3067665). Manuscript received December 2020, revised March 2021. License CC BY-NC-ND 4.0. Author given names corrected: **Neda** Zarayeneh and **Ananth** Kalyanaraman (the prior BibTeX had "Naw" Zarayeneh, which was wrong).

**The critical extension over the 2019 preprint (Section 3).** The journal version **adds support for edge deletions**, which the 2019 preprint did not handle. The full change-set notation is now $\Delta_t = \Delta_t^+ \cup \Delta_t^-$, where $\Delta_t^+ = E_t \setminus E_{t-1}$ are additions and $\Delta_t^- = E_{t-1} \setminus E_t$ are deletions. (Vertex additions/deletions are encoded as edge events.) My earlier text claimed both versions handle deletions; only the journal version does.

**Headline runtime results (Abstract, paper-faithful):**
- **Up to 5× speedup over the dynamic baseline** (i.e. naive-incremental-without-screening).
- **Up to 38× over the static baseline** (rerun-from-scratch).

(Note: my earlier "5–30× faster than from-scratch" range was wrong; actual numbers are 38× over static, 5× over dynamic baseline. And the comparison was *not* over a "SNAP temporal suite Reddit/Stack-Overflow/Wikipedia talk" — the paper uses different inputs; see below.)

**Other journal-version-specific content:**
- Three-way classification of dynamic CD literature (Section 2): static-based, stability-based, cross-time-step approaches — more thorough than the preprint's two-way split.
- Comparison includes **DynaMo** ([39] in this paper) — a 2020 incremental method that the preprint did not cover.
- Cites Seifikar et al. C-Blondel ([26] in this paper) explicitly.

**My fabrications (now removed):**
- ❌ "Batch-mode extension where union of per-edge candidate sets is computed" — the screening predicate is naturally batch-oriented (Algorithm 2 takes a *batch* $\Delta_t$, not single edges); calling this a "new batch-mode extension" was misleading.
- ❌ "Iterative extension as fixpoint computation" — fabricated; not in the paper.
- ❌ "Quality bound proofs: modularity within $O(\epsilon)$ of from-scratch where $\epsilon$ depends on screening slack" — fabricated; the paper has correctness lemmas (extended from the preprint) but no $O(\epsilon)$ bound of this form.
- ❌ "5–30× faster than from-scratch with modularity within 1% of from-scratch and within 0.3% of naive-dynamic" — wrong specific numbers; real headline is 38× / 5× as quoted above.

**For the BTP.** This is the canonical citation for the delta-screening technique — preferred over the 2019 preprint when both could apply. The BTP's GPU delta-screening kernel reproduces the predicate; **specifically check whether the BTP needs deletion support**, as that distinguishes the journal scope from the preprint. The PDF is in the BTP repo at `BTP/Delta-Screening_A_Fast_and_Efficient_Technique_to_.pdf`.

Citation key: `zarayeneh2021delta`.

```bibtex
@article{zarayeneh2021delta,
  author  = {Zarayeneh, Neda and Kalyanaraman, Ananth},
  title   = {Delta-Screening: A Fast and Efficient Technique to Update Communities in Dynamic Graphs},
  journal = {IEEE Transactions on Network Science and Engineering},
  volume  = {8},
  number  = {2},
  pages   = {1614--1629},
  year    = {2021},
  doi     = {10.1109/TNSE.2021.3067665},
  publisher = {IEEE}
}
```

---

## Sahu (2024) — DF Louvain (Dynamic Frontier)

A multicore-OpenMP dynamic Louvain by Subhajit Sahu (IIIT Hyderabad), arXiv 2404.19634v4 (latest 7 Sep 2024). Software at [github.com/puzzlef/louvain-communities-openmp-dynamic](https://github.com/puzzlef/louvain-communities-openmp-dynamic) — the repository the BTP clones for its CPU dynamic baseline.

**The paper's three contributions (Abstract):**
1. **Parallel Dynamic Frontier (DF) Louvain** — a new approach for incrementally identifying affected vertices.
2. **Parallel implementations of Naive-Dynamic (ND) and Delta-Screening (DS) Louvain** — these did not previously exist in parallel form; the paper introduces them as comparison baselines.
3. **Auxiliary-information reuse** — incrementally maintaining $K_{t-1}$ (vertex weighted degrees) and $\Sigma_{t-1}$ (community total edge weights) across snapshots, instead of recomputing each snapshot from scratch.

**The actual DF expansion mechanism (Section 4.1, paper-faithful).**

*Initial marking* — When the batch update arrives, mark source vertex $s$ as affected only for:
- Edge deletions $(s, t) \in \Delta_t^-$ where $s$ and $t$ are in the **same community**, OR
- Edge insertions $(s, t) \in \Delta_t^+$ where $s$ and $t$ are in **different communities**.

The other two cases (deletions across communities, insertions within same community) are *ignored* — they are unlikely to change anyone's community.

*Incremental marking via vertex pruning* — During local-moving, when vertex $v$ changes its community, mark all of $v$'s neighbours as affected and mark $v$ itself as not affected. (My earlier "iteration-0 frontier = changed-edge endpoints; iteration-N frontier = neighbours of vertices that moved in iteration N-1" framing was approximately right but missed the *initial filter* on which edge changes trigger marking.)

*Application scope* — DF marking applies only to the **first Louvain pass**. Subsequent passes process all super-vertices because each pass takes <15% of total time.

**The auxiliary-information insight (Section 4.2).** Computing $K$ and $\Sigma$ from scratch is a real bottleneck for dynamic Louvain. Sahu's measurement: reusing $K_{t-1}$ and $\Sigma_{t-1}$ (instead of recomputing) gives **average speedups of 11.8× for ND, 2.9× for DS, and 48.5× for DF** (with peaks up to 107× on small batches), on graphs from Table 3 with random batch updates of $10^{-7}|E|$ to $0.1|E|$ (80% insertions, 20% deletions). The DF speedup is the largest because DF's per-vertex bookkeeping makes recomputation especially costly.

**Hardware (paper-faithful).** 64-core AMD EPYC-7742 processor.

**Headline empirical results (Abstract — verified):**
- *On real-world dynamic graphs:* DF is **179× faster than Static**, **7.2× faster than ND**, **5.3× faster than DS**.
- *On large graphs with random batch updates:* **183× / 13.8× / 8.7×** respectively.
- *Threading:* 1.6× per doubling of threads.

(My earlier "1.5–3× faster than DS, 5–10× faster than ND" and "SNAP / LAW suite" framing were both wrong. The actual speedups are larger and on a different benchmark.)

**Comparison Table 1 (paper-faithful) — properties of dynamic CD approaches:**

| Approach | Year | Fully dynamic | Batch update | Process subset | Use auxiliary info | Parallel |
|---|---|---|---|---|---|---|
| Aynaud et al. [3] | 2010 | ❌ | ❌ | ❌ | ❌ | ❌ |
| Chong et al. [11] | 2013 | ❌ | ❌ | ❌ | ❌ | ❌ |
| Meng et al. [36] | 2016 | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cordeiro et al. [13] | 2016 | ❌ | ❌ | ✓ | ❌ | ❌ |
| Zarayeneh et al. [72] | 2021 | ✓ | ✓ | ✓ | ❌ | ❌ |
| **Sahu (DF) [Ours]** | 2024 | ✓ | ✓ | ✓ | ✓ | ✓ |

**For the BTP.** Mandatory citation. Direct CPU analogue of the BTP's "GPU frontier dynamic Louvain" kernel. **The auxiliary-information-reuse insight is portable to the BTP**: the GPU kernel currently recomputes $K$ and $\Sigma$ at the start of each snapshot; replacing this with incremental maintenance from $\Delta_t^-$ and $\Delta_t^+$ may give a substantial speedup (Sahu's CPU experiments suggest 48× in the right regime).

Citation key: `sahu2024df`.

```bibtex
@article{sahu2024df,
  author  = {Sahu, Subhajit},
  title   = {{DF Louvain}: Fast Incrementally Expanding Approach for Community Detection on Dynamic Graphs},
  journal = {arXiv preprint arXiv:2404.19634},
  year    = {2024},
  note    = {Latest version v4 (7 Sep 2024). Software: \url{https://github.com/puzzlef/louvain-communities-openmp-dynamic}.},
  url     = {https://arxiv.org/abs/2404.19634}
}
```

---

## Halappanavar, Lu, Kalyanaraman, Tumeo (2017) — Scalable Static and Dynamic Community Detection Using Grappolo

The direct dynamic extension of the 2015 Grappolo paper — same first author, same first software, plus a dynamic-graph algorithm bolted on top. Halappanavar et al. (a) add two new parallelisation heuristics to Grappolo's static pipeline ("data caching" and "threshold scaling"), and (b) sketch and minimally evaluate two dynamic-graph algorithms ("unseeded" and "seeded" clustering). Funded by the DARPA HIVE Graph Challenge; benchmarked on the HIVE Challenge datasets and on the SNAP suite. This is the earliest peer-reviewed *parallel* dynamic Louvain — it pre-dates Zarayeneh's delta-screening (2019/2021) and Sahu's DF Louvain (2024) by several years.

**Static heuristics in this paper (Section II).** Four total: the two carried over from the 2015 paper (Vertex Following + Minimum Label, Graph Coloring) and two genuinely new ones:

- **Data Caching.** In the 2015 implementation, each iteration of the local-moving phase used a `std::map` to accumulate per-neighbouring-community edge weights for the current vertex. Map allocation and irregular memory access dominated runtime as the community count shrank. Halappanavar et al. replace the map with a *vector* that is reused across iterations — same vertex's accumulator is over-written rather than reallocated. Reported speedup: up to **10× on inputs where community counts decrease rapidly**. The honest caveat: on inputs where the community count stays large for many iterations (slow-converging graphs), the vector approach pays a comparison cost (must scan to check existing entries) and can *lose* performance vs. the map. They enable Data Caching by default in all reported experiments anyway.

- **Threshold Scaling.** The new heuristic; closest thing to a novel algorithmic contribution in the paper. The Louvain termination threshold $\theta$ (the minimum modularity gain that justifies another iteration) is held *fixed* in the 2015 paper. Threshold Scaling instead uses a **higher** $\theta$ ($10^{-2}$) in the early phases and a **lower** $\theta$ ($10^{-6}$) towards the end. Combined with coloring, this means: in the early colourful-parallel phases, terminate aggressively (don't waste threads chasing tiny modularity gains while colouring is hurting parallelism), and only switch to the tight threshold once colouring is disabled. Empirical: faster convergence *and* better final modularity than coloring alone — the only case in the paper where two heuristics combine to beat both individually.

**Dynamic graph model (Section III).** Edge edits between snapshots are insertions or deletions; vertices may be old or new. Two algorithm variants:

- **Unseeded clustering** — At each timestep $i$, treat $G_i$ as a brand-new input and run Grappolo from scratch. Implicitly handles all changes correctly. Disadvantage: full recomputation cost regardless of how localised the changes are. (This is essentially "no warm start at all" — strictly *worse* than naive-dynamic, used as a quality reference.)
- **Seeded clustering** — Initialise the vertices of $G_i$ with their community labels at the end of timestep $i-1$, then run Grappolo. **This is naive-dynamic à la Aynaud–Guillaume**, but with the Grappolo parallel infrastructure underneath. Advantage: rapid convergence if edits are localised; better community-label persistence across snapshots (useful for tracking). Disadvantage: stale starting point may be locally suboptimal for $G_i$.

**What this paper is not.** It is *not* delta-screening — there is no per-edge predicate that decides which vertices to re-evaluate. It is *not* a frontier method — there is no expanding active set. It is the *parallel naive-dynamic* baseline. Crucially, the paper itself is honest about this: Section IV explicitly notes "We plan to extend this analysis to a set of larger inputs and include experimental results for dynamic community detection" — i.e., the dynamic-experimental story is largely deferred to future work. The dynamic algorithm is described and implemented; it is just not benchmarked at scale here.

**Empirical results (Section IV).**
- *Hardware:* 2× 10-core Xeon E5-2680 v2 @ 2.80 GHz, 768 GB DDR3, GCC 4.9.2 -Ofast, RHEL 6 (kernel 2.6.32). HyperThreading disabled — so 20 hardware threads max.
- *Quality on ground-truth synthetic inputs (Table I, four `simulated_blockmodel_graph` sizes 20K–5M vertices):* Both Basic (VF+ML+Caching) and Advanced (Basic + Coloring + Threshold Scaling) configurations achieve **100% precision and 100% recall** on the smaller inputs; on the 5M-vertex graph, F-score drops to 0.84 (Basic) / 0.84 (Advanced).
- *Performance on 47 real-world inputs (Table II, ranging from `ca-GrQc` to `friendster`):* Speedup of Advanced over Basic varies wildly across inputs — typical range 1×–4×, occasionally up to ~5× (Figure 2). Notable: speedups don't scale meaningfully past ~10 threads on most inputs because many of the HIVE Challenge graphs are too small (under 10⁶ edges) to hide thread-launch overhead. The dynamic algorithm is **not benchmarked separately** in Section IV.
- *Largest input:* friendster (119 M vertices, 1.8 B edges) — completes at 20 threads in ~813 seconds (Advanced) vs. ~2,520 seconds (Basic). Modularity 0.556 (Basic) → 0.475 (Advanced); a quality regression on friendster, which the paper does not investigate.

**Interesting bit #1 — the Advanced-vs-Basic modularity comparison (Figure 3).** Across all 47 inputs, Advanced typically matches or slightly exceeds Basic in modularity, but on a handful of inputs it produces visibly lower modularity. The hypothesis (not stated explicitly in the paper but inferable): aggressive early-phase termination via threshold scaling can lock in a suboptimal partition that subsequent low-threshold phases cannot escape. This is the Threshold Scaling analogue of the resolution-limit / merge-then-stuck phenomenon in vanilla Louvain — and a plausible cause of the friendster modularity regression.

**Interesting bit #2 — institutional / authorial detail.** Hao Lu had moved from WSU (where he was at the 2015 paper) to **Oak Ridge National Laboratory** by the time of this paper. This is the only paper in the Grappolo lineage where the author affiliations span PNNL, ORNL, and WSU simultaneously — a mini-snapshot of the DOE-lab collaboration that DARPA HIVE explicitly funded.

**Interesting bit #3.** The paper concludes by promising two extensions: (i) *distributed-memory implementation using MPI+OpenMP with incomplete coloring and threshold scaling*, and (ii) *dynamic community detection with community tracking and efficient seeding*. (i) eventually became cuVite / the Ghosh–Gawande line of distributed-memory work; (ii) was largely *not* delivered by this group, leaving the field open for Zarayeneh–Kalyanaraman (delta-screening) and then Sahu (DF Louvain) to claim that ground.

**Relevance for the BTP.** Three concrete things:
1. **Citation hygiene.** Any claim of the form "no parallel dynamic Louvain existed before Sahu 2024" is wrong — Halappanavar 2017 existed, even if its dynamic experiments were thin. The BTP's novelty argument has to engage with this paper specifically and frame the contribution against it (delta-screening + frontier on GPU vs. naive-dynamic on CPU).
2. **Threshold Scaling is portable.** Their threshold-scaling idea is a single-line change in the BTP's GPU kernel: use $10^{-2}$ for the first few phases and $10^{-6}$ thereafter. Plausibly worth a 1–3× wall-clock speedup at minimal quality cost. Worth trying.
3. **The friendster modularity regression is a useful warning.** When the BTP runs its own kernels on graphs where Threshold Scaling is enabled, it should specifically check for the same pattern (Advanced producing *lower* modularity than Basic on certain inputs) — and either fix it or report it honestly.

Citation key: `halappanavar2017scalable`.

```bibtex
@inproceedings{halappanavar2017scalable,
  author    = {Halappanavar, Mahantesh and Lu, Hao and Kalyanaraman, Ananth and Tumeo, Antonino},
  title     = {Scalable Static and Dynamic Community Detection Using {Grappolo}},
  booktitle = {2017 IEEE High Performance Extreme Computing Conference (HPEC)},
  year      = {2017},
  pages     = {1--6},
  address   = {Waltham, MA, USA},
  publisher = {IEEE},
  doi       = {10.1109/HPEC.2017.8091047},
  isbn      = {978-1-5386-3472-1}
}
```

*Note:* The previous placeholder entry under this heading referred to a "Vite / streaming Louvain (PNNL series)". That work is the Ghosh et al. *cuVite* line, which is already covered in [`static_louvain_gpu.md`](static_louvain_gpu.md) under the Gawande et al. (2022) entry. The placeholder is retired here in favour of the verified Halappanavar 2017 dynamic-Grappolo paper, which is the actually-existing PNNL-Halappanavar dynamic Louvain reference.
