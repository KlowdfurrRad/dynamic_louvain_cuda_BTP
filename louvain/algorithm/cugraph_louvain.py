#!/usr/bin/env python3
"""cuGraph (GPU) Louvain benchmark — static and DYNAMIC (batched) inputs.

The GPU counterpart of nx_louvain.py: it reads the same dynamic input format as
the CUDA binaries / df_louvain:

    n m
    u v            x m          (initial undirected edges; weight assumed 1)
    n_batches
    n_del n_ins
    u v            x n_del      (deletions)
    u v            x n_ins      (insertions)
    ... repeated per batch

cuGraph (like NetworkX) has no incremental mode, so for each batch we apply the
edge deletions/insertions and re-run cugraph.louvain FROM SCRATCH. For every
Louvain invocation (the initial graph and one per batch) we report its modularity
/ community count / runtime, plus the TOTAL Louvain time over all invocations.

Only the cugraph.louvain() call is timed (the cuDF edge-list build and the
GPU graph construction are excluded), matching nx_louvain.py which times only the
louvain_communities() call and not graph construction.

Requires RAPIDS (cudf, cugraph, cupy) — i.e. a Linux/WSL/Colab box with an NVIDIA
GPU; RAPIDS does not run on native Windows.

Usage:
    python cugraph_louvain.py <graph_file>     # one file (may contain batches)
    python cugraph_louvain.py                  # run on all SNAP graphs
    python cugraph_louvain.py <file> --resolution 1.0
"""

import argparse
import os
import sys
import time

try:
    import cudf
    import cugraph
    import cupy as cp
except ImportError as e:
    sys.stderr.write(
        "ERROR: could not import RAPIDS (cudf / cugraph / cupy): %s\n"
        "cuGraph requires a RAPIDS install on Linux/WSL/Colab with an NVIDIA GPU.\n"
        % e)
    sys.exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SNAP converted graphs live under real_graphs/snap/graphs/.
SNAP_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "real_graphs", "snap", "graphs"))

# Order: smallest to largest
GRAPHS = [
    ("ca-GrQc", "ca-GrQc_converted.txt"),
    ("facebook", "facebook_combined_converted.txt"),
    ("ca-HepTh", "ca-HepTh_converted.txt"),
    ("ca-HepPh", "ca-HepPh_converted.txt"),
    ("ca-AstroPh", "ca-AstroPh_converted.txt"),
    ("email-Enron", "email-Enron_converted.txt"),
    ("com-amazon", "com-amazon_converted.txt"),
    ("com-dblp", "com-dblp_converted.txt"),
]


def parse_dynamic(path):
    """Parse the dynamic input into (n, initial_edges, batches).

    initial_edges : list of (u, v)
    batches       : list of (deletions, insertions), each a list of (u, v)
    Edges are read as exactly two integer tokens (weight is implicitly 1).
    """
    with open(path, "r") as f:
        toks = f.read().split()
    i = 0

    def nxt():
        nonlocal i
        t = toks[i]
        i += 1
        return t

    n = int(nxt())
    m = int(nxt())
    initial = [(int(nxt()), int(nxt())) for _ in range(m)]

    batches = []
    if i < len(toks):
        nb = int(nxt())
        for _ in range(nb):
            n_del = int(nxt())
            n_ins = int(nxt())
            dels = [(int(nxt()), int(nxt())) for _ in range(n_del)]
            inss = [(int(nxt()), int(nxt())) for _ in range(n_ins)]
            batches.append((dels, inss))
    return n, initial, batches


def build_graph(edges):
    """Build an undirected cuGraph from a list of (u, v) undirected edges
    (weight 1). Each undirected edge is supplied once; cuGraph symmetrises."""
    src = [u for u, v in edges]
    dst = [v for u, v in edges]
    gdf = cudf.DataFrame({
        "src": src,
        "dst": dst,
        "weight": cp.ones(len(edges), dtype="float32"),
    })
    G = cugraph.Graph()  # undirected
    G.from_cudf_edgelist(gdf, source="src", destination="dst",
                         edge_attr="weight", renumber=True)
    return G


def run_louvain(edges, resolution):
    """Build the graph (untimed) and time one cugraph.louvain() invocation.
    Returns (time_seconds, modularity, n_communities)."""
    G = build_graph(edges)
    cp.cuda.runtime.deviceSynchronize()      # finish graph build before timing

    t0 = time.perf_counter()
    parts, mod = cugraph.louvain(G, resolution=resolution)
    cp.cuda.runtime.deviceSynchronize()      # finish louvain before stopping timer
    elapsed = time.perf_counter() - t0

    # parts: cudf DataFrame with columns 'vertex' and 'partition'
    ncomm = int(parts["partition"].nunique())
    return elapsed, float(mod), ncomm


def process(name, path, resolution=1.0):
    """Run Louvain on the initial graph and re-run after each batch update."""
    print(f"Running {name}...", flush=True)
    if not os.path.isfile(path):
        print(f"  SKIP: file not found ({path})", flush=True)
        return None

    n, initial, batches = parse_dynamic(path)

    # maintain the current undirected edge set; rebuild the graph each invocation
    edge_set = set((u, v) if u < v else (v, u) for u, v in initial if u != v)

    invocations = []  # (label, nodes, edges, Q, communities, time)

    t, q, c = run_louvain(list(edge_set), resolution)
    invocations.append(("initial", n, len(edge_set), q, c, t))
    print(f"  initial : nodes={n} edges={len(edge_set)} "
          f"Q={q:.6f} communities={c} time={t:.4f}s", flush=True)

    for bi, (dels, inss) in enumerate(batches, start=1):
        for u, v in dels:
            edge_set.discard((u, v) if u < v else (v, u))
        for u, v in inss:
            if u != v:
                edge_set.add((u, v) if u < v else (v, u))

        t, q, c = run_louvain(list(edge_set), resolution)
        invocations.append((f"batch {bi}", n, len(edge_set), q, c, t))
        print(f"  batch {bi:<3}: nodes={n} edges={len(edge_set)} "
              f"Q={q:.6f} communities={c} time={t:.4f}s", flush=True)

    total = sum(inv[5] for inv in invocations)
    print(f"  TOTAL Louvain time over {len(invocations)} invocation(s): {total:.4f}s",
          flush=True)

    return {
        "name": name,
        "invocations": invocations,
        "total_time": total,
        "nodes": invocations[0][1],
        "edges": invocations[0][2],
        "modularity": invocations[0][3],
        "communities": invocations[0][4],
        "time": invocations[0][5],
    }


def print_table(results):
    print()
    print(f"{'Graph':<14} {'Nodes':>10} {'Edges':>12} {'Modularity':>12} "
          f"{'Communities':>13} {'Init t(s)':>10} {'Total t(s)':>11}")
    print("-" * 87)
    for r in results:
        if r is None:
            continue
        print(f"{r['name']:<14} {r['nodes']:>10} {r['edges']:>12} "
              f"{r['modularity']:>12.6f} {r['communities']:>13} "
              f"{r['time']:>10.4f} {r['total_time']:>11.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph", nargs="?", default=None,
                    help="path to a single graph file (may contain batches); "
                         "default: run all SNAP graphs")
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="Louvain resolution parameter (default 1.0)")
    # accepted for CLI parity with nx_louvain.py; cuGraph louvain takes no seed
    ap.add_argument("--seed", type=int, default=1, help="(ignored by cuGraph)")
    args = ap.parse_args()

    print(f"cuGraph on {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}",
          flush=True)

    results = []
    if args.graph:
        results.append(process(os.path.basename(args.graph), args.graph,
                               resolution=args.resolution))
    else:
        for name, filename in GRAPHS:
            path = os.path.join(SNAP_DIR, filename)
            results.append(process(name, path, resolution=args.resolution))

    print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
