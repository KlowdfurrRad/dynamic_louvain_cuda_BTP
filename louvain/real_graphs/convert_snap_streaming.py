#!/usr/bin/env python3
"""Memory-efficient (two-pass, streaming) SNAP -> dynamic-Louvain converter.

For very large graphs (e.g. com-Orkut, ~117M edges) where loading the whole edge
list into memory is infeasible. Unlike convert_snap_to_dynamic.py (which holds all
edges in a set), this makes two passes over the file and only ever keeps the
vertex-id remap in memory, so peak RAM is ~O(#vertices), not O(#edges):

  pass 1 -- scan the file, collect the set of vertex ids and count the edges;
  pass 2 -- re-scan, remap each edge to the contiguous range 0..N-1, and write it.

ASSUMPTION: the input lists each undirected edge once (true for SNAP com-* ungraph
files). Self-loops are dropped; duplicate edges are NOT removed (removing them would
need an O(#edges) set -- and SNAP com-* graphs have none).

SNAP format : '#' comment lines, then 'FromNodeId<TAB>ToNodeId' per line.
Output      : 'n m', then m 'u v' lines (remapped to 0..N-1), then a trailing '0'
              (n_batches = 0, i.e. a static graph).

Usage:
  python convert_snap_streaming.py <input_snap_file> <output_file>
"""
import sys


def edges(path):
    """Yield (u, v) integer edges from a SNAP file, skipping comments/self-loops."""
    with open(path) as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            p = line.split()
            if len(p) < 2:
                continue
            try:
                u, v = int(p[0]), int(p[1])
            except ValueError:
                continue
            if u != v:
                yield u, v


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python convert_snap_streaming.py <input_snap_file> <output_file>")
    inp, out = sys.argv[1], sys.argv[2]

    # --- pass 1: vertex ids + edge count ---
    nodes = set()
    m = 0
    for u, v in edges(inp):
        nodes.add(u)
        nodes.add(v)
        m += 1
    n = len(nodes)
    print(f"  pass 1: {n} vertices, {m} edges")

    # --- build the contiguous remap, then drop the id set ---
    remap = {x: i for i, x in enumerate(sorted(nodes))}
    nodes.clear()

    # --- pass 2: stream edges through the remap and write ---
    with open(out, "w") as g:
        g.write(f"{n} {m}\n")
        for u, v in edges(inp):
            g.write(f"{remap[u]} {remap[v]}\n")
        g.write("0\n")
    print(f"  wrote {out}  ({n} vertices, {m} edges, 0 batches)")


if __name__ == "__main__":
    main()
