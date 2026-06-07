# REMEMBER — project handoff / state dump

Context-compaction survival notes for the dynamic-Louvain-on-GPU work. Newest
understanding wins. Paths are relative to `BTP/louvain/` unless noted.

## 1. Project

- BTP (B.Tech Project), IIT Madras, Raadhes Chandaluru under Prof. Rupesh Nasre.
  Continuation of **UGRC-I "Dynamic Louvain on GPUs with CUDA"** (report:
  `BTP/Report/UGRC_I_Dynamic_Louvain_on_GPUs.pdf`).
- Goal: GPU dynamic Louvain community detection; evaluate vs NetworkX on real
  (static + temporal) graphs; implement Naive-Dynamic (ND), Delta-Screening (DS),
  Dynamic-Frontier (DF).
- User wants: be **critical about correctness & efficiency**; the rewrite must
  follow the **papers, not the old code**; run scripts must **not** generate/convert
  graphs. User often runs the GPU code themselves (sometimes rejects my runs) but
  allows compiles / small validations.

## 2. Main deliverable: `algorithm/df_louvain.cu` (I wrote this from scratch)

One process computes **Static + ND + DF + DS** via a single shared `louvain()`
driver. Key design (see `notes/dynamic_louvain.md` for full detail):
- **Vertex-parallel** local move, **one kernel launch per iteration** (no
  cooperative kernels). Per-vertex open-addressing **hashtable** (global mem,
  `nextPow2(deg+1)`) for O(deg) best-community selection.
- **SoA CSR** `DGraph{off,dst,w}`; `w`/`K`/`Σ`/`Q` are `double`. M = directed arcs
  = 2·edges.
- Move gain: `dQ = 2(Kic-Kid)/M - 2·ki·(ki+Sc-Sd)/M²` (M=total directed weight).
- Aggregation: Thrust `sort_by_key`+`reduce_by_key` on key `src*nc+dst`; self-loops
  kept (carry σ_c).
- Dendrogram tracked in `d_orig` (folded each pass).
- Marking on host: `markND`/`markDF`/`markDS`. DF grows the frontier on the GPU
  (moved vertex → neighbours added to range). DS = **parallel adaptation** of
  Zarayeneh: screens with only newly-inserted edge weights + drops the `gain1≥gain2`
  deferral (over-marks = safe superset).
- `main()` keeps **3 independent states** (sND, sDF, sDS) carried across batches;
  prints Q/comms/affected/time per method per batch; writes DF communities to argv[1].
- Constants (top of file): `TOLERANCE=1e-2`, `TOLERANCE_DROP=10`, `AGG_TOLERANCE=0.8`,
  `MAX_ITERATIONS=20`, `MAX_PASSES=10`, `MOVE_EPS 1e-12` (a `#define` — user changed
  from `static const` because device code can't read a host const), `TPB=256`.
- **Compile:** `nvcc -O3 -arch=sm_86 df_louvain.cu -o df_louvain` (sm_86 = Ampere /
  RTX 3050; the spin-lock needs Volta+ Independent Thread Scheduling). compile_dyn.sh
  has this line appended.

## 3. THE bug and THE fix (most important)

**Symptom:** Static returned modularity *below the all-singletons baseline* on
dense graphs (CollegeMsg −0.158, ER n100 −0.042) — impossible for correct Louvain.
Also caused **under-coarsening** (3–110× too many communities on static SNAP).
**Root cause:** the original lock-free async local move committed moves scored
against **stale Σ** (community totals being atomicAdd'd by other threads) → accepted
modularity-*decreasing* moves + 2-cycle swaps. (The OLD `cuda_dynamic_louvain.cu`
had a related bug from commit `fc2708a "Degree optimization"`: k_i read from
`community_degree` indexed by community id, so ~85% of vertices got k_i=0.)
**Fix (verified-commit-under-lock):** optimistic best-community pick → lock the
`(d,best_c)` community pair (`atomicCAS`, min,max order) → **re-read Σ and re-scan
K under the lock** (membership frozen) → commit only if exact `dQ2 > MOVE_EPS`.
Guarantees Q monotonically non-decreasing ⇒ no sub-singleton, no swap oscillation.
No deadlock (ordered locks + sm_86 ITS). Details: `notes/bugfixes.md` (may have been
deleted by user at one point — recreate if missing) + memory files
`df-louvain-sub-singleton-q` / `df-louvain-undercoarsening` (both marked FIXED).

**Result:** Static now MATCHES NetworkX on static SNAP (com-amazon 0.926 vs 0.926,
com-dblp 0.821 vs 0.822, ca-GrQc 0.860 vs 0.861) at **3–336× speed**;
under-coarsening gone (was a symptom of the same bug — tuning AGG_TOLERANCE/MAX_PASSES
did NOT help, confirming it was the move kernel).

## 4. Residual issues (secondary, post-fix)

1. **Large temporal-initial Static under-coarsens** (sx-askubuntu/superuser stop at
   ~3 passes, Q 0.215/0.328); dynamic variants recover to ~0.42/0.40. Likely
   AGG_TOLERANCE→1.0 pass-stop tuning.
2. **ER n10000 over-merges** to 2 communities (structureless edge case).
3. **sx-superuser batch-5 ND** collapses (0.233); DF/DS steadier there.
4. **Host CSR rebuilt per batch** (no incremental graph/K/Σ maintenance — the DF
   paper's auxiliary-info trick is NOT implemented).
5. **GPU memory:** peak ≈ **184 B/edge + 80 B/node** (worst case; bottleneck =
   Thrust aggregation temporaries coexisting with hashtable). 4 GB RTX 3050 ceiling
   ≈ **18 M edges**: com-Youtube/web-Google fit (~640/865 MB); **LiveJournal (35 M
   edges, ~6.7 GB) OOMs**. See `notes/dynamic_louvain_analysis.md`.

## 5. Comparison results (df vs NetworkX) — in `notes/df_vs_nx_{snap,temporal,random}_comparison.md`

- **Static SNAP:** df ≈ NetworkX modularity & community counts, 3–336× faster.
- **Temporal:** CollegeMsg fixed (0.246, tracks NX); **sx-mathoverflow df ND BEATS
  NX** (0.321 vs 0.310); askubuntu/superuser within ~0.04–0.05, **160–193× faster**;
  DF is the steadiest variant; inter-batch drift mostly gone.
- **Random ER:** all positive now, ND competitive/beats NX on n100–n500; n10000
  over-merges. ER = stress test (no real community structure).
- These docs have full per-graph, per-batch tables (Static/initial + 5 batches ×
  ND/DF/DS/NetworkX).

## 6. Directory map (`BTP/louvain/`)

- `algorithm/`: **df_louvain.cu** (the good one); OLD/buggy: cuda_static_louvain.cu,
  cuda_dynamic_louvain.cu, cuda_dynamic_louvain_nodebased.cu; **nx_louvain.py**;
  compile_dyn.sh.
- `real_graphs/`
  - `snap/` — static SNAP. `graphs/` (raw + `*_converted.txt`), `outputs/{cuda_static,
    normal,node_based,networkx,df}/`, **run_benchmarks.sh** + **run_df.sh** (both live
    HERE now, moved from real_graphs/; LOUVAIN_DIR = `dirname(dirname(SCRIPT_DIR))`).
  - `snap_temporal/` — `graphs/` (temporal `SRC DST UNIXTS`), `converted/`,
    `outputs/{df,networkx,...}/`, `convert_snap_temporal_to_dynamic.py` (I removed its
    `_nx.txt` output), run_benchmarks.sh, run_df.sh.
  - `classic/` — karate + dolphins. `prepare_classic.py` (karate from networkx,
    dolphins from `graphs/dolphins.mtx`), graphs/, outputs/{df,networkx}/, run_df.sh,
    run_nx.sh.
  - `LiveJournal/` — `graphs/com-LiveJournal.txt` (converted, 4.0M/34.7M, static),
    outputs/, run_df.sh, run_nx.sh (warn: OOM/huge).
  - `konect/` (ucidata-zachary = karate, arenas-jazz, arenas-pgp, petster-hamster),
    `convert_snap_to_dynamic.py`, `snap_results_table.py`.
- `generate/` — synthetic ER graphs. graphgen.py (emits BOTH unweighted `-o` and
  weighted `--raw-output`), run_benchmarks.sh (NetworkX now reads full dynamic input,
  no truncated `_nx`), run_df.sh, outputs/.
- `notes/` — dynamic_louvain.md (df_louvain impl), dynamic_louvain_analysis.md
  (memory+overhead), bugfixes.md, benchmark_graphs.md, df_vs_nx_*_comparison.md,
  explanation.md, etc.

## 7. Graphs added this session

- **com-Youtube** (1.13M/2.99M) + **web-Google** (876K/4.32M deduped) → `snap/graphs/`
  as `*_converted.txt`; registered in snap/run_benchmarks.sh, snap/run_df.sh,
  snap_results_table.py (NOT nx_louvain.py — too big for NetworkX; NX_SKIP_ABOVE guards).
- **karate** (34/78), **dolphins** (62/159) → `classic/`.

## 8. I/O format (df_louvain, nx_louvain, CUDA bins)

`n m` / `m` × `u v` / `n_batches` / per batch: `n_del n_ins` + del `u v` + ins `u v`.
Weights read as `u v` (=1). Static graphs end with trailing `0` (n_batches=0).
`nx_louvain.py` now parses batches and re-runs Louvain per batch (per-invocation +
total time); SNAP_DIR=`real_graphs/snap/graphs`.

## 9. The report (`BTP/report/`)

- `chap-introduction.md` (LaTeX content) — Motivation, Research Objectives,
  **Contributions** (I polished bullets 1–4; bullets 5–6 are `TODO` benchmarking/
  results — **do NOT fill**, user is still getting results). Typos remain in
  Motivation ("optimize a the", "udpates", "Updation").
- `chap-prelim.tex` — Modularity / Louvain / **Dynamic Louvain CDA** (I wrote the
  dynamic section). Notation: `k_i`, `\sum_c`, `k_{i,c}`, `m`, `\Delta Q_{i:a\to b}`
  via `\eqref{eq:modularity_gain}`.
- `work-theory.tex` — **UGRC Work** (sequential CPU louvain, simple dynamic CPU,
  static GPU louvain, benchmarked vs cuGraph+own CPU; left: GPU correctness bugs/
  collapse, only random graphs), **Project Timeline** (5-phase TABLE, Period column =
  Month 1–5 placeholders for real dates), **Dynamic Louvain CDA** (ND/DS/DF). DS
  described as parallel adaptation (note it omits gain1≥gain2 + uses inserted weights).
- Citations used: `aynaud2010static`, `zarayeneh2021delta`,
  `sahu2024dflouvainfastincrementally` — **verify these keys exist in the report's
  .bib** (couldn't locate the bib; `newman2004finding`/`blondel2008` are used).
- Standardise "Louvain" capitalisation (body uses lowercase inconsistently).

## 10. Key papers / algorithms

- DF Louvain (Sahu, `2404.19634v4.pdf`): Alg 1 DF, Alg 2 ND, Alg 3 (parallel DS),
  Alg 4 dynamic Louvain, Alg 5 local-move, Alg 6 aggregate, Alg 7 updateWeights.
  Modularity Eq 1, delta-modularity Eq 2.
- Delta-Screening (Zarayeneh, `BTP/Delta-Screening...pdf`): Alg 2 additions
  (gain1≥gain2 AND gain1>0; mark i,j*,Γ(i),C(j*)), Alg 3 deletions (intra-community →
  mark C(i)∪Γ(i)∪Γ(j)).
- GVE-Louvain (Sahu, `report_research/.../10-Sahu-GVE-Louvain`) = the static base.

## 11. Environment & tooling

- Windows 11, **RTX 3050 Laptop 4 GB, sm_86 (compute 8.6)**, CUDA 12.6, nvcc on PATH
  (also in git-bash). `py -3` = **Python 3.14.2** with **networkx 3.6.1**; `python`
  also works. git-bash for Bash tool; **curl + gzip yes, wget + unzip NO** (use
  Python `zipfile`). Bash tool HAS network access (downloaded from snap.stanford.edu,
  nrvis.com). Python 3.14 had a transient shutdown-quirk (exit 120) once but reran fine.
- SNAP temporal download URLs: `https://snap.stanford.edu/data/<name>.txt.gz`;
  com-youtube under `/data/bigdata/communities/`; dolphins from
  `https://nrvis.com/download/data/misc/dolphins.zip` (umich mirror was 403).

## 12. Next likely tasks

- Re-run + regenerate the 3 comparison docs whenever df_louvain changes.
- Possible: implement free-hashtable-before-aggregate / float weights / 32-bit keys
  to fit LiveJournal in 4 GB; fix large-temporal Static pass-stop (AGG_TOLERANCE→~0.95).
- Report writing continues; fill TODOs only when the user has benchmark results.
