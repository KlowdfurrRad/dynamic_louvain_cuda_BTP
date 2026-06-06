#!/usr/bin/env python3
"""Convert a SNAP temporal-edge-list file into the CUDA dynamic Louvain input format.

SNAP temporal files (sx-mathoverflow, sx-askubuntu, sx-superuser, CollegeMsg, ...)
have format, one edge per line:

    SRC DST UNIXTS

where UNIXTS is a Unix timestamp. The edges are ordered (or can be ordered) by
timestamp.

We split the edge stream chronologically:
  - The first `initial_pct` fraction forms the initial (static) graph.
  - The remaining edges are split evenly into `n_batches` batches of
    insertions (temporal graphs in SNAP contain only insertions).

Duplicate undirected edges (u,v) and (v,u) across different timestamps are
deduped to the earliest timestamp so the final graph is a simple undirected
graph, matching the assumptions of the CUDA Louvain kernels.

Output format (matching what the CUDA binaries in algorithm/ expect after the
"read u v, ignore weight" change):

    n_nodes n_initial_edges
    u1 v1                    (× n_initial_edges)
    ...
    n_batches
    n_del n_ins              (first batch header)
    (no deletion lines, n_del == 0 for temporal graphs)
    u v                      (× n_ins for the first batch)
    ...
    n_del n_ins              (second batch header)
    ...

NetworkX now reads this same dynamic input directly (nx_louvain.py re-runs
Louvain after each batch), so no separate initial-only `_nx.txt` file is written.

Usage:
    python convert_snap_temporal_to_dynamic.py <input> <output>
        [--initial-pct 0.8] [--n-batches 5]
"""

import argparse
import os
import sys


def convert(input_path, output_path, initial_pct, n_batches):
    print(f"Reading {input_path} ...", flush=True)

    edges = []   # list of (ts, u_norm, v_norm) where u_norm < v_norm
    seen = set()  # for dedup of undirected edges
    skipped_self_loops = 0

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                u, v, ts = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if u == v:
                skipped_self_loops += 1
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            edges.append((ts, a, b))

    print(f"  Unique undirected edges: {len(edges)}", flush=True)
    print(f"  Skipped self-loops: {skipped_self_loops}", flush=True)

    # Sort chronologically
    edges.sort(key=lambda e: e[0])

    # Remap nodes to contiguous 0..N-1
    nodes = sorted({a for _, a, _ in edges} | {b for _, _, b in edges})
    node_map = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    n_edges = len(edges)

    print(f"  Unique nodes: {n_nodes}", flush=True)

    # Split chronologically into initial + batches
    n_initial = int(n_edges * initial_pct)
    initial = edges[:n_initial]
    rest = edges[n_initial:]

    # Divide `rest` into `n_batches` nearly equal chunks (final chunk takes the remainder)
    batches = []
    if n_batches <= 0:
        batches = []
    elif not rest:
        batches = [[] for _ in range(n_batches)]
    else:
        base = len(rest) // n_batches
        extra = len(rest) % n_batches
        idx = 0
        for i in range(n_batches):
            this_size = base + (1 if i < extra else 0)
            batches.append(rest[idx:idx + this_size])
            idx += this_size

    print(f"  Initial edges: {len(initial)} ({initial_pct*100:.0f}%)", flush=True)
    print(f"  Batches: {n_batches}; per-batch insertion counts: "
          f"{[len(b) for b in batches]}", flush=True)

    # Write the CUDA-format dynamic input
    with open(output_path, "w") as f:
        f.write(f"{n_nodes} {len(initial)}\n")
        for _, a, b in initial:
            f.write(f"{node_map[a]} {node_map[b]}\n")
        f.write(f"{n_batches}\n")
        for batch in batches:
            f.write(f"0 {len(batch)}\n")
            # No deletion lines (n_del == 0).
            for _, a, b in batch:
                f.write(f"{node_map[a]} {node_map[b]}\n")

    print(f"Wrote {output_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="SNAP temporal edge-list file (u v ts per line)")
    ap.add_argument("output", help="converted CUDA-format output file")
    ap.add_argument("--initial-pct", type=float, default=0.8,
                    help="fraction of earliest edges used as the initial graph (default 0.8)")
    ap.add_argument("--n-batches", type=int, default=5,
                    help="number of batches for the remaining edges (default 5)")
    args = ap.parse_args()

    if not 0.0 < args.initial_pct < 1.0:
        print("ERROR: --initial-pct must be in (0, 1)", file=sys.stderr)
        sys.exit(1)
    if args.n_batches < 0:
        print("ERROR: --n-batches must be >= 0", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    convert(args.input, args.output, args.initial_pct, args.n_batches)


if __name__ == "__main__":
    main()
