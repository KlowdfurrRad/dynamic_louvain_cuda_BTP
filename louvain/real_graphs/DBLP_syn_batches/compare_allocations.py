#!/usr/bin/env python3
"""Compare df_louvain's per-batch community allocations and their modularity.

Needs the per-batch assignment dumps from df_louvain run with env DF_DUMP=1
(DF_DUMP=1 bash run_df.sh), which writes under outputs/df/:
    <graph>_b0.txt                 (the shared static start)
    <graph>_b<k>_<ND|DF|DS>.txt    (each method after batch k),  each line "v comm".

For each graph it reports, per batch:

  (a) Modularity of each assignment evaluated on the graph AT that batch:
        - static(frozen): the batch-0 (static) partition kept UNCHANGED, and
        - the three dynamic variants ND, DF, DS.
      The frozen column is the counterfactual "what if we never re-clustered"; comparing it
      with ND/DF/DS shows how much quality the dynamic updates recover.

  (b) How much the partition moved between consecutive batches: the adjusted Rand index
      (ARI = 1 -> identical) and an approximate count of reassigned vertices.

Modularity is  Q = (internal edges) / m  -  sum_c (D_c / 2m)^2,  with D_c the total degree
of community c and m the number of edges in the (current) graph (unweighted).

Usage:  py -3 compare_allocations.py
"""
import glob
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DF_DIR = os.path.join(SCRIPT_DIR, "outputs", "df")
GRAPHS_DIR = os.path.join(SCRIPT_DIR, "graphs")


def load(path):
    """Read a "v comm" dump into a dict {vertex: community}."""
    comm = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                comm[int(p[0])] = int(p[1])
    return comm


def read_dynamic(path):
    """(n, initial_edges, batches) from the dynamic input graphs/<graph>.txt.
    batches is a list of (deletions, insertions), each a list of canonical (u<v) edges."""
    with open(path) as f:
        n, m = map(int, f.readline().split())
        edges = []
        for _ in range(m):
            a, b = map(int, f.readline().split())
            edges.append((a, b) if a < b else (b, a))
        head = f.readline().split()
        nb = int(head[0]) if head else 0
        batches = []
        for _ in range(nb):
            nd, ni = map(int, f.readline().split())
            dels, inss = [], []
            for _ in range(nd):
                a, b = map(int, f.readline().split())
                dels.append((a, b) if a < b else (b, a))
            for _ in range(ni):
                a, b = map(int, f.readline().split())
                inss.append((a, b) if a < b else (b, a))
            batches.append((dels, inss))
    return n, edges, batches


def degrees(n, edge_set):
    deg = [0] * n
    for a, b in edge_set:
        deg[a] += 1
        deg[b] += 1
    return deg


def modularity(n, edge_set, deg, comm):
    """Q of partition `comm` (dict v->community) on the unweighted graph `edge_set`."""
    m = len(edge_set)
    if m == 0:
        return 0.0
    two_m = 2.0 * m
    D = defaultdict(float)
    for v in range(n):
        D[comm.get(v, v)] += deg[v]
    internal = 0
    for a, b in edge_set:
        if comm.get(a, a) == comm.get(b, b):
            internal += 1
    return internal / m - sum((d / two_m) ** 2 for d in D.values())


def compare(a, b):
    """(ARI, reassigned, n) over the vertices shared by label dicts a and b."""
    keys = a.keys() & b.keys()
    n = len(keys)
    if n == 0:
        return float("nan"), 0, 0
    cont = defaultdict(int)
    asz = defaultdict(int)
    bsz = defaultdict(int)
    for v in keys:
        cont[(a[v], b[v])] += 1
        asz[a[v]] += 1
        bsz[b[v]] += 1

    def c2(x):
        return x * (x - 1) // 2

    index = sum(c2(x) for x in cont.values())
    sa = sum(c2(x) for x in asz.values())
    sb = sum(c2(x) for x in bsz.values())
    tot = c2(n)
    exp = (sa * sb / tot) if tot else 0.0
    mx = 0.5 * (sa + sb)
    ari = 1.0 if mx == exp else (index - exp) / (mx - exp)
    dom = defaultdict(int)
    for (ai, _bj), c in cont.items():
        if c > dom[ai]:
            dom[ai] = c
    reassigned = n - sum(dom.values())
    return ari, reassigned, n


def fmt(x, w=8):
    return f"{x:>{w}.6f}" if x is not None else f"{'--':>{w}}"


def main():
    starts = sorted(glob.glob(os.path.join(DF_DIR, "*_b0.txt")))
    if not starts:
        print(f"No dumps in {DF_DIR}.")
        print("Produce them with:  DF_DUMP=1 bash run_df.sh")
        return

    for s in starts:
        graph = os.path.basename(s)[:-len("_b0.txt")]
        inp = os.path.join(GRAPHS_DIR, f"{graph}.txt")
        print(f"\n=== {graph} ===")
        if not os.path.isfile(inp):
            print(f"  input {graph}.txt not found in graphs/ -- cannot compute modularity.")
            continue

        n, init_edges, batches = read_dynamic(inp)
        nb = len(batches)
        b0 = load(s)

        print("  Modularity of each assignment on the graph at each batch")
        print("  (static = batch-0 partition kept FROZEN; ND/DF/DS re-optimise each batch):")
        print(f"    {'batch':>5} | {'static(frozen)':>14} | {'ND':>8} | {'DF':>8} | {'DS':>8}")

        edge_set = set(init_edges)
        deg = degrees(n, edge_set)
        q0 = modularity(n, edge_set, deg, b0)            # batch 0: all four equal b0
        print(f"    {0:>5} | {fmt(q0, 14)} | {fmt(q0)} | {fmt(q0)} | {fmt(q0)}")

        prev = {"ND": b0, "DF": b0, "DS": b0}
        ari_rows = []
        for k in range(1, nb + 1):
            dels, inss = batches[k - 1]
            for e in dels:
                edge_set.discard(e)
            for e in inss:
                edge_set.add(e)
            deg = degrees(n, edge_set)

            cur = {}
            for meth in ("ND", "DF", "DS"):
                p = os.path.join(DF_DIR, f"{graph}_b{k}_{meth}.txt")
                cur[meth] = load(p) if os.path.isfile(p) else None

            q_static = modularity(n, edge_set, deg, b0)  # FROZEN static on this batch's graph
            q_m = {meth: (modularity(n, edge_set, deg, cur[meth]) if cur[meth] else None)
                   for meth in ("ND", "DF", "DS")}
            print(f"    {k:>5} | {fmt(q_static, 14)} | "
                  f"{fmt(q_m['ND'])} | {fmt(q_m['DF'])} | {fmt(q_m['DS'])}")

            for meth in ("ND", "DF", "DS"):
                if cur[meth] is not None and prev[meth] is not None:
                    ari, reassigned, _ = compare(prev[meth], cur[meth])
                    ari_rows.append((k, meth, ari, reassigned))
            for meth in ("ND", "DF", "DS"):
                if cur[meth] is not None:
                    prev[meth] = cur[meth]

        if ari_rows:
            print("  Partition movement between consecutive batches (ARI = 1 -> identical):")
            print(f"    {'batch':>5} | {'method':>6} | {'ARI':>7} | reassigned")
            for k, meth, ari, reassigned in ari_rows:
                print(f"    {k:>5} | {meth:>6} | {ari:>7.4f} | {reassigned}")


if __name__ == "__main__":
    main()
