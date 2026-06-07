import time

print("=" * 60)
print("cuGraph GPU Graph Analytics Test")
print("=" * 60)

# Check GPU availability
import cupy as cp
print(f"\nGPU Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
print(f"CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")

import cudf
import cugraph

# Build a small test graph using cuDF DataFrames
sources = [0, 0, 1, 1, 2, 3, 3, 4]
targets = [1, 2, 3, 4, 3, 4, 5, 5]
weights = [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0]

gdf = cudf.DataFrame({
    "src": sources,
    "dst": targets,
    "weight": weights
})

G = cugraph.Graph()
G.from_cudf_edgelist(gdf, source="src", destination="dst", edge_attr="weight", renumber=True)

print(f"\nGraph created: {G.number_of_vertices()} vertices, {G.number_of_edges()} edges")

# 1. PageRank
print("\n--- PageRank ---")
t0 = time.time()
pr = cugraph.pagerank(G)
t1 = time.time()
print(pr.sort_values("pagerank", ascending=False).to_pandas().to_string(index=False))
print(f"Time: {t1 - t0:.4f}s")

# 2. BFS — use a vertex we know exists from the PageRank results
print("\n--- BFS ---")
valid_start = int(pr["vertex"].iloc[0])
print(f"Starting from vertex: {valid_start}")
bfs_result = cugraph.bfs(G, start=valid_start)
print(bfs_result.sort_values("vertex").to_pandas().to_string(index=False))

# 3. SSSP (Single Source Shortest Path)
print("\n--- SSSP ---")
print(f"Starting from vertex: {valid_start}")
sssp_result = cugraph.sssp(G, source=valid_start)
print(sssp_result.sort_values("vertex").to_pandas().to_string(index=False))

# 4. Louvain community detection
print("\n--- Louvain Community Detection ---")
parts, modularity = cugraph.louvain(G)
print(f"Modularity: {modularity:.4f}")
print(parts.sort_values("vertex").to_pandas().to_string(index=False))

print("\n" + "=" * 60)
print("All tests passed successfully!")
print("=" * 60)