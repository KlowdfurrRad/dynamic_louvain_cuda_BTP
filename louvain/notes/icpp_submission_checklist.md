# ICPP Submission Checklist

## What We Have
- CUDA static Louvain
- Three dynamic variants (naive, frontier, delta-screening)
- Node-based vs edge-based kernel comparison
- Benchmarks on 8 real SNAP graphs

## TODO

### Bugs to Fix
- Edge-based kernel crashes on denser graphs (illegal memory access during aggregation)
- `cudaMemcpyFromSymbol` on `volatile __device__` returns stale 0 — breaks early stopping
- Reported modularity != true modularity (computed on aggregated graph, not original)

### Baselines Needed
- CPU sequential Louvain (NetworkX or similar)
- Reference OpenMP dynamic Louvain (already in repo: `louvain-communities-openmp-dynamic/`)
- NVIDIA cuGraph Louvain (RAPIDS)
- Without baselines, paper will be rejected

### Dynamic Benchmarks
- Currently only running static (0 batches) — need actual batch insert/delete workloads
- Vary batch size (1%, 5%, 10%, 20% of edges)
- Show dynamic speedup vs re-running static from scratch
- Use temporal graphs or synthetic batch generation

### Scalability & Profiling
- Scaling plots: time vs graph size, time vs batch size
- GPU profiling with `nsys`/`ncu`: occupancy, memory throughput, kernel time breakdown
- Test on different GPU architectures if possible

### Algorithmic Contribution (pick/strengthen one)
- GPU-native delta-screening (currently host-side — move to GPU)
- Frontier propagation inside cooperative kernel
- Edge-based vs node-based parallelism tradeoffs (quality vs robustness)

### Paper
- 10-12 pages, double-column, ACM/IEEE format
- Sections: Intro, Background, Design, Implementation, Evaluation, Related Work, Conclusion
- Figures: pseudocode, speedup charts, modularity tables, scaling plots, kernel breakdowns
- Double-blind review — no identifying info

### ICPP Info
- Deadline typically March-April for September conference
- Check [icpp-conf.org](https://icpp-conf.org) for 2026 dates

---

## More Realistic Venues (for a BTP-level project)

### Workshops (co-located with top conferences, lower bar, still peer-reviewed)
- **GrAPL** (Graph Algorithms Building Blocks) — workshop at IPDPS. Graph algorithms on GPUs is exactly their scope.
- **HPGP** (High Performance Graph Processing) — workshop at various HPC conferences
- **AsHES** (Accelerators and Hybrid Exascale Systems) — workshop at IPDPS, GPU-focused
- **IA^3** (Irregular Applications: Architectures and Algorithms) — workshop at SC

### Indian Conferences
- **HiPC** (International Conference on High Performance Computing) — India-based, good for GPU/parallel work, has a student research symposium track
- **COMSNETS** — if framed as community detection for network analysis
- **ICDCN** — distributed computing, could fit dynamic graph updates angle

### Journals (professor-suggested)
- **IJPP** (International Journal of Parallel Programming) — good fit, accepts GPU implementation papers, reasonable bar
- **TPDS** (IEEE Transactions on Parallel and Distributed Systems) — top-tier journal, harder but very prestigious
- **JPDC** (Journal of Parallel and Distributed Computing) — solid mid-tier, fits well
- **Concurrency and Computation: Practice and Experience** — accepts GPU implementation papers
- **IEEE Access** — open access, faster review, lower bar

### Student/Poster Tracks
- Most top conferences (SC, IPDPS, PPoPP, ICPP) have **student poster sessions** — much lower bar, great for a BTP
- **SC Student Cluster Competition** — if your institution participates

### Recommendation
**HiPC student research symposium** or a **workshop paper at IPDPS (GrAPL/AsHES)** are the most realistic targets. Fix the bugs, add one CPU baseline, run dynamic benchmarks with batches, and you have a solid submission.