#!/usr/bin/env python3
"""NetworKit (parallel CPU) Louvain benchmark -- static and DYNAMIC (batched) inputs.

The NetworKit counterpart of nx_louvain.py: it reads the same dynamic input format
as the CUDA binaries / df_louvain:

    n m
    u v            x m          (initial undirected edges; weight assumed 1)
    n_batches
    n_del n_ins
    u v            x n_del      (deletions)
    u v            x n_ins      (insertions)
    ... repeated per batch

NetworKit's PLM (Parallel Louvain Method) has no incremental mode, so for each
batch we apply the edge deletions/insertions and re-run PLM FROM SCRATCH. For every
Louvain invocation (the initial graph and one per batch) we report its modularity /
community count / runtime, plus the TOTAL Louvain time over all invocations.

Only the PLM detection (construction + run) is timed -- not graph
construction/modification -- matching nx_louvain.py which times only the
louvain_communities() call. The output format is identical to nx_louvain.py so the
same compact_results.py parser handles it.

NetworKit's PLM is multi-threaded (OpenMP); by default it uses all cores, so the
timing reflects parallel CPU performance. PLM is not fully deterministic across
thread counts even with a fixed seed; pass --threads 1 for reproducibility.

Requires NetworKit:  pip install networkit

Usage:
    python networkit_louvain.py <graph_file>     # one file (may contain batches)
    python networkit_louvain.py                  # run on all SNAP graphs
    python networkit_louvain.py <file> --seed 42 --threads 1
"""

import argparse
import os
import sys
import time

try:
    import networkit as nk
except ImportError as e:
    sys.stderr.write("ERROR: networkit not installed: %s\n"
                     "Install it with:  pip install networkit\n" % e)
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


def run_louvain(G, seed):
    """Run one PLM invocation; return (time_seconds, modularity, n_communities)."""
    try:
        nk.setSeed(seed, True)   # reproducible (per-thread seeding)
    except Exception:
        pass
    t0 = time.perf_counter()
    plm = nk.community.PLM(G, refine=False)   # plain (unrefined) Louvain
    plm.run()
    part = plm.getPartition()
    elapsed = time.perf_counter() - t0
    q = nk.community.Modularity().getQuality(part, G)
    return elapsed, q, part.numberOfSubsets()


def process(name, path, seed=1):
    """Run Louvain on the initial graph and re-run after each batch update."""
    print(f"Running {name}...", flush=True)
    if not os.path.isfile(path):
        print(f"  SKIP: file not found ({path})", flush=True)
        return None

    n, initial, batches = parse_dynamic(path)

    # NetworKit nodes are 0..n-1, matching the converted input. Unweighted graph
    # (all edge weights are 1, so modularity is identical to the weighted case).
    G = nk.Graph(n)
    for u, v in initial:
        if u != v and not G.hasEdge(u, v):
            G.addEdge(u, v)

    invocations = []  # (label, nodes, edges, Q, communities, time)

    # --- initial (static) Louvain ---
    t, q, c = run_louvain(G, seed)
    invocations.append(("initial", G.numberOfNodes(), G.numberOfEdges(), q, c, t))
    print(f"  initial : nodes={G.numberOfNodes()} edges={G.numberOfEdges()} "
          f"Q={q:.6f} communities={c} time={t:.4f}s", flush=True)

    # --- one re-run per batch update ---
    for bi, (dels, inss) in enumerate(batches, start=1):
        for u, v in dels:
            if G.hasEdge(u, v):
                G.removeEdge(u, v)
        for u, v in inss:
            if u != v and not G.hasEdge(u, v):
                G.addEdge(u, v)

        t, q, c = run_louvain(G, seed)
        invocations.append((f"batch {bi}", G.numberOfNodes(), G.numberOfEdges(), q, c, t))
        print(f"  batch {bi:<3}: nodes={G.numberOfNodes()} edges={G.numberOfEdges()} "
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
    ap.add_argument("--seed", type=int, default=1, help="RNG seed")
    ap.add_argument("--threads", type=int, default=None,
                    help="number of OpenMP threads (default: NetworKit default = all cores; "
                         "use 1 for reproducibility)")
    args = ap.parse_args()

    try:
        nk.setLogLevel("ERROR")          # silence NetworKit's progress chatter
    except Exception:
        pass
    if args.threads:
        try:
            nk.setNumberOfThreads(args.threads)
        except Exception:
            pass
    try:
        print(f"NetworKit {nk.__version__} on {nk.getMaxNumberOfThreads()} thread(s)",
              flush=True)
    except Exception:
        pass

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
