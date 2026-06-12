# Memory Coalescing: AoS vs SoA

A small microbenchmark study behind the *"coalesced memory access"* point in the
report — why the present Louvain implementation stores the graph as a
**structure-of-arrays (SoA)** instead of the UGRC **array-of-structures (AoS)**
`Edge {src, dest, weight}` layout.

## The one idea

GPU DRAM is fetched in **32-byte sectors**. The cost of a read is set by the
**number of sectors touched**, *not* the number of bytes the kernel actually
uses. AoS interleaves an edge's fields, so reading a few fields of many edges
drags whole records through the bus; SoA puts each field in its own array, so a
warp's 32 lanes read contiguous, fully-used sectors.

## Setup

- **GPU:** NVIDIA RTX 3050 Laptop (Ampere, `sm_86`, 4 GB), CUDA 12.6.
- One thread per edge, 256 threads/block, timed with CUDA events over 20 reps
  after a warm-up. Effective GB/s = bytes the layout forces through the bus
  (sectors touched + the output write) ÷ time.
- Build: `./compile.sh` → `mem_coalesce`, `mem_coalesce_fat`,
  `mem_coalesce_fat_sparse`.

## The three tests

| file | record | fields the kernel reads | what it isolates |
|---|---|---|---|
| `mem_coalesce.cu` | `Edge` (16 B) | `dest`, `weight` (12 B) | the real Louvain case: small record, small waste |
| `mem_coalesce_fat.cu` | `FatEdge` (64 B) | **all 10 fields** | AoS *best case* — every byte used |
| `mem_coalesce_fat_sparse.cu` | `FatEdge` (64 B) | `dest` (off 4) + `w6` (off 48) | AoS *worst case* — 2 fields in different sectors |

Both layouts always allocate the **same data**; only the access pattern differs.

## Why the speed-up is just arithmetic

Per edge, count bytes that must cross the bus (read + the 8 B output write):

| test | AoS B/edge | SoA B/edge | predicted SoA speed-up | **measured** |
|---|---|---|---|---|
| `mem_coalesce`      | 16 + 8 = 24 | 12 + 8 = 20 | **1.2×** | ~1.2× &nbsp;*(see note)* |
| `mem_coalesce_fat`  | 64 + 8 = 72 | 64 + 8 = 72 | **1.0×** | **1.01×** |
| `mem_coalesce_fat_sparse` | 64 + 8 = 72 | 12 + 8 = 20 | **3.6×** | **3.5×** |

- **`fat` (1.01×):** both kernels hit ~183 GB/s — near the card's peak. When every
  byte is used, AoS wastes nothing and both layouts saturate DRAM; **layout stops
  mattering.**
- **`fat_sparse` (3.5×):** `dest` and `w6` fall in *different* 32 B sectors, so AoS
  must pull the whole 64 B record for 12 useful bytes, while SoA touches only the
  two needed arrays. The predicted 3.6× and measured 3.5× match — the missing 0.1
  is fixed overhead (partial last warp, TLB, write effects).

> **Note on `mem_coalesce`:** a 100 M-edge run reported only ~24 GB/s for *both*
> layouts. That run oversubscribed the 4 GB card (AoS 1.6 + SoA 1.6 + out 0.8 =
> 4.0 GB) and was paging over PCIe — **not** a coalescing measurement. Rerun
> in-VRAM (`./mem_coalesce 50000000`, ~2.0 GB) for the valid ~1.2×.

## Takeaway

The penalty scales with **how much of each record is wasted per access**:

- all bytes used → layout is irrelevant (≈1.0×);
- small record, slight waste (the real Louvain edge scan) → small win (≈1.2×);
- wide record read sparsely → large win (≈3.5×).

The Louvain kernels sit at the modest end — but SoA costs nothing to adopt and
never hurts, which is why the present implementation uses SoA CSR
(`off` / `dst` / `w`) throughout.
