# CUDA Louvain Community Detection

## Files

| File | Description |
|------|-------------|
| `cuda_static_louvain.cu` | Static Louvain algorithm on GPU |
| `cuda_dynamic_louvain.cu` | Static + Naive Dynamic + Frontier Dynamic + Delta-Screening Dynamic Louvain on GPU |
| `test_dynamic_input.txt` | Sample input for the dynamic version |

## Requirements

- NVIDIA GPU with compute capability >= 6.0
- CUDA Toolkit (nvcc)

## Compile

### Static Louvain

```bash
nvcc -rdc=true -arch=sm_60 cuda_static_louvain.cu -o static_louvain
```

### Dynamic Louvain

```bash
nvcc -rdc=true -arch=sm_60 cuda_dynamic_louvain.cu -o dynamic_louvain
```

> Adjust `-arch=sm_60` to match your GPU (e.g. `sm_75` for Turing, `sm_86` for Ampere, `sm_89` for Ada Lovelace).

## Run

### Static Louvain

```bash
./static_louvain < input.txt
```

**Input format:**

```
n_nodes m_edges
u1 v1 w1
u2 v2 w2
...
```

- `n_nodes`: number of vertices
- `m_edges`: number of undirected edges
- Each edge line: source, destination, weight (undirected — both directions are added internally)

### Dynamic Louvain

```bash
./dynamic_louvain < test_dynamic_input.txt
```

**Input format:**

```
n_nodes m_edges
u1 v1 w1
...
n_batches
n_deletions n_insertions
del_u1 del_v1 del_w1
...
ins_u1 ins_v1 ins_w1
...
```

Each batch specifies edge deletions followed by edge insertions. The program:

1. Runs **static Louvain** on the initial graph
2. For each batch:
   - Applies the edge updates
   - Runs **Naive Dynamic Louvain** (all vertices, warm-started from previous communities)
   - Runs **Frontier Dynamic Louvain** (only affected vertices processed in first pass)
   - Runs **Delta-Screening Dynamic Louvain** (screens insertions for positive modularity gain)
   - Prints modularity and timing comparison of all four approaches

## Example

Using the provided test input (3 communities of sizes 4, 4, 2 with weak inter-community bridges):

```bash
nvcc -rdc=true -arch=sm_60 cuda_dynamic_louvain.cu -o dynamic_louvain
./dynamic_louvain < test_dynamic_input.txt
```

## Debug Mode

Compile with `-DDEBUG` to enable verbose per-node logging:

```bash
nvcc -rdc=true -arch=sm_60 -DDEBUG cuda_dynamic_louvain.cu -o dynamic_louvain_debug
```
