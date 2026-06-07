# NetworkX vs CUDA Louvain Comparison

## NetworkX (CPU baseline)

| Graph | Nodes | Edges | Modularity | Communities | Time (s) |
|-------|-------|-------|------------|-------------|----------|
| ca-GrQc | 5,241 | 14,484 | 0.858981 | 389 | 0.28 |
| facebook | 4,039 | 88,234 | 0.834971 | 16 | 0.76 |
| ca-HepTh | 9,875 | 25,973 | 0.769321 | 473 | 0.83 |
| ca-HepPh | 12,006 | 118,489 | 0.659640 | 310 | 1.43 |
| ca-AstroPh | 18,771 | 198,050 | 0.628977 | 323 | 3.75 |
| email-Enron | 36,692 | 183,831 | 0.604123 | 1,271 | 6.63 |
| com-amazon | 334,863 | 925,872 | 0.926024 | 231 | 70.86 |
| com-dblp | 317,080 | 1,049,866 | 0.821487 | 212 | 85.06 |

## CUDA Implementations (True Modularity)

| Graph | Static Q | Static T | Static Comms | Dyn-edge Q | Dyn-edge T | Dyn-edge Comms | Dyn-node Q | Dyn-node T | Dyn-node Comms |
|-------|----------|----------|--------------|------------|------------|----------------|------------|------------|----------------|
| ca-GrQc | 0.7316 | 148ms | 1,125 | 0.7187 | 24ms | 1,197 | 0.7053 | 57ms | 1,257 |
| facebook | 0.8261 | 3,395ms | 69 | 0.8143 | 2,586ms | 90 | 0.8057 | 1,890ms | 99 |
| ca-HepTh | 0.6040 | 885ms | 2,061 | 0.5996 | 436ms | 2,067 | 0.5820 | 261ms | 2,143 |
| ca-HepPh | CRASH | — | — | 0.5928 | 52,807ms | 1,750 | 0.5800 | 1,152ms | 1,898 |
| ca-AstroPh | 0.5511 | 53,707ms | 2,018 | 0.5431 | 18,800ms | 2,046 | 0.5431 | 3,256ms | 2,065 |
| email-Enron | CRASH | — | — | 0.5642 | 66,433ms | 3,764 | 0.5452 | 21,179ms | 3,978 |
| com-amazon | 0.6759 | 8,504ms | 52,321 | 0.6765 | 4,340ms | 52,286 | 0.6760 | 2,931ms | 52,708 |
| com-dblp | 0.5874 | 569,947ms | 59,662 | 0.5864 | 7,340ms | 60,561 | 0.5925 | 17,897ms | 59,757 |

## Quality Gap (NetworkX vs Best CUDA)

| Graph | NX Q | Best CUDA Q | Q Gap | NX Comms | CUDA Comms | Comms Ratio |
|-------|------|-------------|-------|----------|------------|-------------|
| ca-GrQc | 0.859 | 0.732 | **−0.127** | 389 | 1,125 | 2.9× |
| facebook | 0.835 | 0.826 | −0.009 | 16 | 69 | 4.3× |
| ca-HepTh | 0.769 | 0.604 | **−0.165** | 473 | 2,061 | 4.4× |
| ca-HepPh | 0.660 | 0.593 | −0.067 | 310 | 1,750 | 5.6× |
| ca-AstroPh | 0.629 | 0.551 | −0.078 | 323 | 2,018 | 6.2× |
| email-Enron | 0.604 | 0.564 | −0.040 | 1,271 | 3,764 | 3.0× |
| com-amazon | **0.926** | 0.677 | **−0.249** | 231 | 52,286 | **226×** |
| com-dblp | **0.821** | 0.593 | **−0.228** | 212 | 59,662 | **281×** |

## Speed Comparison (NetworkX vs Best CUDA)

| Graph | NetworkX (s) | Best CUDA (s) | Speedup |
|-------|--------------|---------------|---------|
| ca-GrQc | 0.28 | 0.024 | **11.5×** |
| facebook | 0.76 | 1.89 | 0.40× (slower) |
| ca-HepTh | 0.83 | 0.26 | 3.2× |
| ca-HepPh | 1.43 | 1.15 | 1.2× |
| ca-AstroPh | 3.75 | 3.26 | 1.2× |
| email-Enron | 6.63 | 21.18 | 0.31× (slower) |
| com-amazon | 70.86 | 2.93 | **24.2×** |
| com-dblp | 85.06 | 7.34 | **11.6×** |

## Key Observations

1. **Quality is significantly worse** across the board. Modularity gap ranges from 0.009 (facebook) to 0.249 (com-amazon).
2. **Community counts are 3×–280× higher** than NetworkX. On com-amazon and com-dblp, the CUDA versions never coarsen below ~50,000 communities while NetworkX merges down to ~200.
3. **Root cause:** the `cudaMemcpyFromSymbol` on `volatile __device__ double delta_Q_sum` returns stale 0, triggering early termination after the first aggregation pass. Static and dynamic codes only run 1-2 passes instead of the 5-10 needed for full coarsening.
4. **Speed wins are real but currently meaningless** — winning 24× on com-amazon while computing a 280× worse partition is not publishable.
5. **Where CUDA is competitive in quality** (facebook: 0.826 vs 0.835), the speed is actually worse than NetworkX.

## Required Fix

Replace the `__device__ volatile double delta_Q_sum` with a `cudaMalloc`'d device pointer that can be properly read via `cudaMemcpy`. After fixing, re-run all benchmarks. Expected outcome:
- Community counts drop 10-100×
- Modularity rises 0.05-0.20 toward NetworkX values
- Runtimes increase (more passes will run)
- True speedup vs CPU should still be 5-20× on large graphs
