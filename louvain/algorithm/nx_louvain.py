#!/usr/bin/env python3
"""NetworkX Louvain benchmark on all SNAP graphs.

Reads each converted SNAP graph (first line: "n m"; following lines: "u v";
trailing "0" for n_batches), runs networkx.community.louvain_communities,
and prints modularity, number of communities, and runtime to stdout.

Usage:
    python nx_louvain.py                  # run on all SNAP graphs
    python nx_louvain.py <graph_file>     # run on a single graph file
"""

import argparse
import os
import sys
import time

import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "real_graphs", "snap"))

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


def load_graph(path: str) -> nx.Graph:
    G = nx.Graph()
    with open(path, "r") as f:
        header = f.readline().split()
        n = int(header[0])
        G.add_nodes_from(range(n))
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue  # skips trailing "0" (n_batches) line
            u, v = int(parts[0]), int(parts[1])
            w = float(parts[2]) if len(parts) >= 3 else 1.0
            G.add_edge(u, v, weight=w)
    return G


def run_one(name: str, path: str, seed: int = 1):
    """Returns dict with results, or None if file missing."""
    print(f"Running {name}...", flush=True)
    if not os.path.isfile(path):
        print(f"  SKIP: file not found ({path})", flush=True)
        return None

    G = load_graph(path)
    t0 = time.perf_counter()
    communities = louvain_communities(G, weight="weight", seed=seed)
    elapsed = time.perf_counter() - t0
    q = modularity(G, communities, weight="weight")

    return {
        "name": name,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "modularity": q,
        "communities": len(communities),
        "time": elapsed,
    }


def print_table(results):
    print()
    print(f"{'Graph':<14} {'Nodes':>10} {'Edges':>12} {'Modularity':>12} {'Communities':>13} {'Time (s)':>10}")
    print("-" * 75)
    for r in results:
        if r is None:
            continue
        print(f"{r['name']:<14} {r['nodes']:>10} {r['edges']:>12} "
              f"{r['modularity']:>12.6f} {r['communities']:>13} {r['time']:>10.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph", nargs="?", default=None,
                    help="path to a single graph file (default: run all SNAP graphs)")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed")
    args = ap.parse_args()

    results = []
    if args.graph:
        results.append(run_one(os.path.basename(args.graph), args.graph, seed=args.seed))
    else:
        for name, filename in GRAPHS:
            path = os.path.join(SNAP_DIR, filename)
            results.append(run_one(name, path, seed=args.seed))

    print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
