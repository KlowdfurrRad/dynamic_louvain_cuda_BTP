#!/usr/bin/env python3
"""Add SYNTHETIC random batch updates to a static graph, producing a dynamic
Louvain input (initial graph + batches) for df_louvain / the CUDA binaries.

Follows the random-batch-update methodology of DF Louvain (Sahu, arXiv:2404.19634,
Section 5.1.4): each batch is a configurable mix of edge insertions and deletions
(set by --ins-ratio; the paper uses 80% : 20%), insertions are uniformly random
vertex pairs that are not already edges, deletions are uniformly random existing
edges, the graph is kept undirected, and no vertices are added or removed. Batches
are applied cumulatively (later batches see the effect of earlier ones), so the
stream resembles a real evolving graph.

No graph snapshot ever contains a duplicate edge: the initial graph is de-duplicated,
each batch's insertions avoid every edge already present in the current graph (and one
another), and an edge is never both deleted and re-inserted within the same batch.

Input  : a converted static graph in the CUDA format --
             n m
             u v            (x m)
             0              (or any trailing batch section -- ignored)
Output : the same initial graph followed by synthetic batches --
             n m
             u v            (x m)
             n_batches
             n_del n_ins    (per batch)
             u v            (x n_del   deletions)
             u v            (x n_ins   insertions)
             ...

Each undirected edge is listed once as "u v" (df_louvain adds both directions).

Usage:
    python gen_synthetic_batches.py <static_input> <dynamic_output> \
        [--n-batches 5] [--batch-frac 0.001] [--ins-ratio 0.8] [--seed 42]

  --batch-frac : edges changed per batch, as a fraction of the ORIGINAL |E|
                 (so n_changed = round(batch_frac * m); n_ins = ins_ratio * n_changed,
                  n_del = the rest). The paper sweeps 1e-7 .. 0.1.
"""

import argparse
import os
import random
import sys


def read_static(path):
    """Read the initial graph: returns (n_nodes, list_of_unique_undirected_edges).
    Self-loops and duplicate edges are dropped so the initial snapshot is clean."""
    seen = set()
    edges = []
    with open(path) as f:
        first = f.readline().split()
        n, m = int(first[0]), int(first[1])
        for _ in range(m):
            parts = f.readline().split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue
            e = (u, v) if u < v else (v, u)
            if e in seen:
                continue
            seen.add(e)
            edges.append(e)
    return n, edges


def gen_batches(n, edges, n_batches, batch_frac, ins_ratio, rng):
    """Generate cumulative batches. Returns a list of (deletions, insertions), each a
    list of (u,v) undirected pairs.

    Guarantees no duplicate edges in any snapshot: deletions are distinct existing
    edges; insertions are distinct vertex pairs absent from the CURRENT graph (checked
    before deletions are applied, so a deleted edge is never re-inserted in the same
    batch); the running graph is then updated (delete, then insert) for the next batch."""
    edge_set = set(edges)                       # running graph; source of truth
    m0 = len(edge_set)
    n_changed = max(1, round(batch_frac * m0))
    n_ins = max(0, round(ins_ratio * n_changed))
    n_del = max(0, n_changed - n_ins)

    batches = []
    for b in range(n_batches):
        # --- deletions: distinct existing edges ---
        cur = list(edge_set)
        k_del = min(n_del, len(cur))
        dels = rng.sample(cur, k_del) if k_del else []

        # --- insertions: distinct vertex pairs absent from the current graph. Checking
        #     against edge_set BEFORE removing the deletions ensures an edge is never
        #     deleted and re-inserted in the same batch, and never duplicates a survivor.
        inss = []
        seen_ins = set()
        attempts = 0
        cap = n_ins * 50 + 100
        while len(inss) < n_ins and attempts < cap:
            attempts += 1
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (u, v) if u < v else (v, u)
            if e in edge_set or e in seen_ins:
                continue
            seen_ins.add(e)
            inss.append(e)
        if len(inss) < n_ins:
            print(f"  warn: batch {b+1} only found {len(inss)}/{n_ins} insertions "
                  f"(graph nearly complete?)", file=sys.stderr)

        # --- apply the batch to the running graph for the next iteration ---
        edge_set.difference_update(dels)
        edge_set.update(seen_ins)

        batches.append((dels, inss))
    return batches


def write_dynamic(path, n, edges, batches):
    with open(path, "w") as f:
        f.write(f"{n} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")
        f.write(f"{len(batches)}\n")
        for dels, inss in batches:
            f.write(f"{len(dels)} {len(inss)}\n")
            for u, v in dels:
                f.write(f"{u} {v}\n")
            for u, v in inss:
                f.write(f"{u} {v}\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="converted static graph (n m / u v / 0)")
    ap.add_argument("output", help="dynamic output (initial + synthetic batches)")
    ap.add_argument("--n-batches", type=int, default=5)
    ap.add_argument("--batch-frac", type=float, default=0.001,
                    help="edges changed per batch as a fraction of |E| (default 0.001)")
    ap.add_argument("--ins-ratio", type=float, default=0.8,
                    help="fraction of each batch that is insertions (default 0.8)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    n, edges = read_static(args.input)
    print(f"Read {n} nodes, {len(edges)} undirected edges from {args.input}")

    batches = gen_batches(n, edges, args.n_batches, args.batch_frac,
                          args.ins_ratio, rng)
    per = [(len(d), len(i)) for d, i in batches]
    print(f"Generated {len(batches)} batches (del, ins per batch): {per}")

    write_dynamic(args.output, n, edges, batches)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
