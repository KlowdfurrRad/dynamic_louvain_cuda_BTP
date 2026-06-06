"""
Convert a SNAP-format undirected graph file to the dynamic Louvain input format
(without batches).

SNAP format:
  # comment lines
  FromNodeId\tToNodeId

Output format (for cuda_dynamic_louvain):
  n_nodes m_edges
  u1 v1
  u2 v2
  ...
  0           (n_batches = 0, i.e., no dynamic updates)

Usage:
  python convert_snap_to_dynamic.py <input_snap_file> <output_file>

Example:
  python convert_snap_to_dynamic.py \
      "com-LiveJournal.ungraph.txt/com-lj.ungraph.txt" \
      livejournal_dynamic_input.txt
"""

import sys
import os


def convert_snap_to_dynamic(input_path, output_path):
    edges = []
    nodes = set()

    print(f"Reading {input_path} ...")
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue  # skip self-loops
            # store with smaller id first to avoid duplicates
            if u > v:
                u, v = v, u
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)

    print(f"  Raw edges read: {len(edges)}")

    # Deduplicate
    edges = list(set(edges))

    # Remap node IDs to contiguous 0..N-1
    sorted_nodes = sorted(nodes)
    node_map = {old: new for new, old in enumerate(sorted_nodes)}

    n_nodes = len(sorted_nodes)
    edges = sorted((node_map[u], node_map[v]) for u, v in edges)
    m_edges = len(edges)

    print(f"  Unique nodes: {n_nodes}")
    print(f"  Unique edges: {m_edges}")
    print(f"Writing {output_path} ...")

    with open(output_path, "w") as f:
        f.write(f"{n_nodes} {m_edges}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")
        f.write("0\n")  # no batches

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_snap_file> <output_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.isfile(input_path):
        print(f"Error: {input_path} not found")
        sys.exit(1)

    convert_snap_to_dynamic(input_path, output_path)
