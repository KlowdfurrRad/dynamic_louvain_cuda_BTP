# CUDA Louvain Community Detection

## Project Layout

```
louvain/
├── algorithm/
│   ├── cuda_static_louvain.cu              # GPU static Louvain (standalone)
│   ├── cuda_dynamic_louvain.cu             # Static + 3 dynamic variants (edge-based kernel)
│   ├── cuda_dynamic_louvain_nodebased.cu   # Same as above with node-based kernel
│   ├── nx_louvain.py                       # NetworkX CPU baseline benchmark
│   └── compile_dyn.sh                      # Compile script
├── real_graphs/
│   ├── snap/                               # SNAP graphs + converted versions
│   ├── konect/                             # KONECT graphs
│   ├── outputs/{cuda_static,normal,node_based}/   # Benchmark outputs + community files
│   ├── convert_snap_to_dynamic.py          # SNAP → our input format
│   ├── run_benchmarks.sh                   # Run all 3 implementations on all graphs
│   └── results_table.py                    # Parse outputs + compute true modularity
├── generate/
│   └── graphgen.py                         # Random graph + batch generator
├── notes/                                  # All documentation
└── test_dynamic_input.txt                  # Sample dynamic input
```

## Requirements

- NVIDIA GPU with compute capability >= 6.0
- CUDA Toolkit (nvcc)
- Python 3 with `networkx` (for the CPU baseline)

## Compile

```bash
cd algorithm
nvcc -rdc=true -arch=sm_60 cuda_static_louvain.cu -o static_louvain
nvcc -rdc=true -arch=sm_60 cuda_dynamic_louvain.cu -o dynamic_louvain
nvcc -rdc=true -arch=sm_60 cuda_dynamic_louvain_nodebased.cu -o dynamic_louvain_nodebased
```

> Adjust `-arch=sm_60` to match your GPU (e.g. `sm_75` Turing, `sm_86` Ampere, `sm_89` Ada Lovelace).

## Run

All three executables take an optional output filename as the first argument (community assignments) and read the graph from stdin.

```bash
./static_louvain communities.txt < graph.txt
./dynamic_louvain communities.txt < graph.txt
./dynamic_louvain_nodebased communities.txt < graph.txt
```

The output community file format:
```
<n_communities>          <- total community count on the first line
0 <community_of_0>
1 <community_of_1>
...
```

### Input format

Static / single-graph runs:
```
n_nodes m_edges
u1 v1
u2 v2
...
0                        <- 0 batches (required for the dynamic binary)
```

Dynamic runs (batch updates):
```
n_nodes m_edges
u1 v1
...
n_batches
n_deletions n_insertions
del_u1 del_v1 del_w1
...
ins_u1 ins_v1 ins_w1
...
```

**Note:** the current code reads only `u v` for the initial graph (weight is hardcoded to `1.0`), but reads `u v w` for batch deletions and insertions. This inconsistency is documented in the bug list.

## Benchmarking

```bash
# Convert SNAP files to our format
python real_graphs/convert_snap_to_dynamic.py snap/ca-GrQc.txt snap/ca-GrQc_converted.txt

# Run all 3 implementations on all SNAP graphs
cd real_graphs
bash run_benchmarks.sh

# Print comparison table (with independently computed true modularity)
python results_table.py

# CPU baseline (NetworkX)
python ../algorithm/nx_louvain.py
```

## Debug Mode

```bash
nvcc -rdc=true -arch=sm_60 -DDEBUG cuda_dynamic_louvain.cu -o dynamic_louvain_debug
```

## Documentation

- [explanation.md](explanation.md) — algorithm and code walkthrough
- [dynamic_louvain.md](dynamic_louvain.md) — design notes for the dynamic variants
- [real_small_graph_datasets.md](real_small_graph_datasets.md) — benchmark dataset list
- [nx_vs_cuda_comparison.md](nx_vs_cuda_comparison.md) — current quality/speed gap vs NetworkX
- [icpp_submission_checklist.md](icpp_submission_checklist.md) — publication targets and TODOs
