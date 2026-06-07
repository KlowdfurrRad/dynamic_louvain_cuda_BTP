#!/usr/bin/env python3
"""Compact the per-method Louvain metrics in classic/outputs/ into one CSV.

Scans classic/outputs/<method>/<graph>.txt run logs (method = df / networkx /
cugraph / ...), extracts the modularity, community count, and runtime of each
graph x method, and writes classic/results.csv -- one tidy row per (graph,
method). Node/edge counts are read from graphs/<graph>_converted.txt.

The two log formats handled:
  df_louvain : "Static: Q=0.415598  communities=4  passes=3  time=140 ms"
  nx/cugraph : "  initial : nodes=34 edges=78 Q=0.385355 communities=3 time=0.0027s"

For these static (0-batch) classic graphs each method has a single result.
Times are normalised to seconds (df's ms are divided by 1000).

Usage:  py -3 compact_results.py
"""

import csv
import glob
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
GRAPHS_DIR = os.path.join(SCRIPT_DIR, "graphs")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results.csv")

# df_louvain Static line
RE_DF = re.compile(
    r"Static:\s*Q=([-\d.eE]+)\s+communities=(\d+)\s+passes=(\d+)\s+time=(\d+)\s*ms")
# nx_louvain.py / cugraph_louvain.py "initial :" line
RE_NX = re.compile(
    r"initial\s*:\s*nodes=(\d+)\s+edges=(\d+)\s+Q=([-\d.eE]+)\s+"
    r"communities=(\d+)\s+time=([-\d.eE]+)s")

# preferred column/print order for methods
METHOD_ORDER = {"df": 0, "networkx": 1, "networkit": 2, "cugraph": 3}


def graph_nm(name):
    """(nodes, edges) from graphs/<name>_converted.txt, else (None, None)."""
    p = os.path.join(GRAPHS_DIR, f"{name}_converted.txt")
    if not os.path.isfile(p):
        return None, None
    with open(p) as f:
        parts = f.readline().split()
    return int(parts[0]), int(parts[1])


def parse_log(path):
    """Return dict(modularity, communities, time_s, nodes, edges) or None."""
    text = open(path, errors="replace").read()
    m = RE_DF.search(text)
    if m:
        q, comm, passes, ms = m.groups()
        return dict(modularity=float(q), communities=int(comm), passes=int(passes),
                    time_s=int(ms) / 1000.0, nodes=None, edges=None)
    m = RE_NX.search(text)
    if m:
        n, e, q, comm, t = m.groups()
        return dict(modularity=float(q), communities=int(comm), passes=None,
                    time_s=float(t), nodes=int(n), edges=int(e))
    return None


def main():
    if not os.path.isdir(OUTPUTS_DIR):
        print(f"No outputs/ directory at {OUTPUTS_DIR}")
        return

    methods = sorted(
        (d for d in os.listdir(OUTPUTS_DIR)
         if os.path.isdir(os.path.join(OUTPUTS_DIR, d))),
        key=lambda d: (METHOD_ORDER.get(d, 99), d))

    # graph names = <graph>.txt logs (excluding *_communities.txt) across methods
    graphs = set()
    for meth in methods:
        for f in glob.glob(os.path.join(OUTPUTS_DIR, meth, "*.txt")):
            b = os.path.basename(f)[:-4]
            if b.endswith("_communities"):
                continue
            graphs.add(b)
    graphs = sorted(graphs)

    rows = []
    for g in graphs:
        gn, ge = graph_nm(g)
        for meth in methods:
            path = os.path.join(OUTPUTS_DIR, meth, f"{g}.txt")
            if not os.path.isfile(path):
                continue
            r = parse_log(path)
            if r is None:
                print(f"  warn: could not parse metrics from {meth}/{g}.txt")
                continue
            rows.append({
                "graph": g,
                "method": meth,
                "nodes": r["nodes"] if r["nodes"] is not None else gn,
                "edges": r["edges"] if r["edges"] is not None else ge,
                "modularity": round(r["modularity"], 6),
                "communities": r["communities"],
                "passes": r["passes"] if r["passes"] is not None else "",
                "time_s": round(r["time_s"], 6),
            })

    cols = ["graph", "method", "nodes", "edges", "modularity", "communities",
            "passes", "time_s"]
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # readable echo to stdout
    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}\n")
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) if rows else len(c)
              for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
