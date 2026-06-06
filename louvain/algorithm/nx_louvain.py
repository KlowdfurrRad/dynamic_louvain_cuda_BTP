#!/usr/bin/env python3
"""NetworkX Louvain benchmark — static and DYNAMIC (batched) inputs.

Reads the same dynamic input format as the CUDA binaries / df_louvain:

    n m
    u v            x m          (initial undirected edges; weight assumed 1)
    n_batches
    n_del n_ins
    u v            x n_del      (deletions)
    u v            x n_ins      (insertions)
    ... repeated per batch

NetworkX has no incremental mode, so for each batch we apply the edge
deletions/insertions to the graph and re-run louvain_communities FROM SCRATCH.
We report, for every Louvain invocation (the initial graph and one per batch),
its modularity / community count / runtime, plus the TOTAL Louvain time summed
over all invocations.

Usage:
    python nx_louvain.py <graph_file>     # one file (may contain batches)
    python nx_louvain.py                  # run on all SNAP graphs (static)
    python nx_louvain.py <file> --seed 42
"""

import argparse
import os
import sys
import time

import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SNAP converted graphs now live under real_graphs/snap/graphs/.
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
    """Parse the dynamic input file into (n, initial_edges, batches).

    initial_edges : list of (u, v)
    batches       : list of (deletions, insertions), each a list of (u, v)
    Edges are read as exactly two integer tokens (weight is implicitly 1),
    matching the converter output and df_louvain's reader.
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


def run_louvain(G, seed):
    """Run one Louvain invocation; return (time_seconds, modularity, n_communities)."""
    t0 = time.perf_counter()
    communities = louvain_communities(G, weight="weight", seed=seed)
    elapsed = time.perf_counter() - t0
    q = modularity(G, communities, weight="weight")
    return elapsed, q, len(communities)


def process(name, path, seed=1):
    """Run Louvain on the initial graph and re-run after each batch update.

    Returns a dict with the per-invocation results and the total Louvain time,
    or None if the file is missing.
    """
    print(f"Running {name}...", flush=True)
    if not os.path.isfile(path):
        print(f"  SKIP: file not found ({path})", flush=True)
        return None

    n, initial, batches = parse_dynamic(path)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v in initial:
        G.add_edge(u, v, weight=1.0)

    invocations = []  # (label, nodes, edges, Q, communities, time)

    # --- initial (static) Louvain ---
    t, q, c = run_louvain(G, seed)
    invocations.append(("initial", G.number_of_nodes(), G.number_of_edges(), q, c, t))
    print(f"  initial : nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
          f"Q={q:.6f} communities={c} time={t:.4f}s", flush=True)

    # --- one re-run per batch update ---
    for bi, (dels, inss) in enumerate(batches, start=1):
        for u, v in dels:
            if G.has_edge(u, v):
                G.remove_edge(u, v)
        for u, v in inss:
            G.add_edge(u, v, weight=1.0)

        t, q, c = run_louvain(G, seed)
        invocations.append((f"batch {bi}", G.number_of_nodes(), G.number_of_edges(), q, c, t))
        print(f"  batch {bi:<3}: nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
              f"Q={q:.6f} communities={c} time={t:.4f}s", flush=True)

    total = sum(inv[5] for inv in invocations)
    print(f"  TOTAL Louvain time over {len(invocations)} invocation(s): {total:.4f}s",
          flush=True)

    return {
        "name": name,
        "invocations": invocations,
        "total_time": total,
        # convenience: the initial-graph result (used by the summary table)
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
    ap.add_argument("--seed", type=int, default=1, help="RNG seed")
    args = ap.parse_args()

    results = []
    if args.graph:
        results.append(process(os.path.basename(args.graph), args.graph, seed=args.seed))
    else:
        for name, filename in GRAPHS:
            path = os.path.join(SNAP_DIR, filename)
            results.append(process(name, path, seed=args.seed))

    print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
