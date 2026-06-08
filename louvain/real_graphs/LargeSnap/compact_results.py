#!/usr/bin/env python3
"""Compact the per-method Louvain metrics in LargeSnap/outputs/ into one CSV.

Scans LargeSnap/outputs/<method>/<graph>.txt run logs (method = df / networkx /
networkit / cugraph) and writes LargeSnap/results.csv -- one tidy row per (graph,
method). These large SNAP graphs are static (0 batches), so each method has a
single result. Output naming is the same as the other directories (snap, classic,
...): outputs/<method>/<graph>.txt.

Log formats handled:
  df_louvain : "Static: Q=0.95  communities=..  passes=..  time=.. ms"
  nx/networkit/cugraph : "  initial : nodes=.. edges=.. Q=.. communities=.. time=..s"

Columns: graph, nodes, edges, method, variant, batch, modularity, communities,
         passes, affected, time_s
  variant : df -> Static ;  nx,networkit,cugraph -> rerun   (batch always 0)
  nodes/edges from graphs/<graph>_converted.txt ; time normalised to seconds.

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

KNOWN_METHODS = ["df", "networkx", "networkit", "cugraph"]
METHOD_ORDER = {m: i for i, m in enumerate(KNOWN_METHODS)}
VARIANT_ORDER = {"Static": 0, "rerun": 0, "ND": 1, "DF": 2, "DS": 3}


def input_path(graph):
    return os.path.join(GRAPHS_DIR, f"{graph}_converted.txt")


RE_DF_STATIC = re.compile(
    r"Static:\s*Q=([-\d.eE]+)\s+communities=(\d+)\s+passes=(\d+)\s+time=(\d+)\s*ms")
RE_DF_BATCH = re.compile(
    r"\[batch (\d+)\]\s*(ND|DF|DS):\s*Q=([-\d.eE]+)\s+comms=(\d+)"
    r"(?:\s+affected0=(\d+))?\s+time=(\d+)\s*ms")
RE_NX_INIT = re.compile(
    r"initial\s*:\s*nodes=(\d+)\s+edges=(\d+)\s+Q=([-\d.eE]+)\s+"
    r"communities=(\d+)\s+time=([-\d.eE]+)s")
RE_NX_BATCH = re.compile(
    r"batch\s+(\d+)\s*:\s*nodes=(\d+)\s+edges=(\d+)\s+Q=([-\d.eE]+)\s+"
    r"communities=(\d+)\s+time=([-\d.eE]+)s")


def read_nm(path):
    """(nodes, edges) from the first line of the converted input, else (None, None)."""
    if not os.path.isfile(path):
        return None, None
    with open(path) as f:
        parts = f.readline().split()
    return int(parts[0]), int(parts[1])


def parse_df(text):
    out = []
    m = RE_DF_STATIC.search(text)
    if m:
        q, comm, passes, ms = m.groups()
        out.append(dict(variant="Static", batch=0, modularity=float(q),
                        communities=int(comm), passes=int(passes), affected=None,
                        time_s=int(ms) / 1000.0))
    for m in RE_DF_BATCH.finditer(text):
        b, var, q, comm, aff, ms = m.groups()
        out.append(dict(variant=var, batch=int(b), modularity=float(q),
                        communities=int(comm), passes=None,
                        affected=int(aff) if aff is not None else None,
                        time_s=int(ms) / 1000.0))
    return out


def parse_nx(text):
    out = []
    m = RE_NX_INIT.search(text)
    if m:
        _n, _e, q, comm, t = m.groups()
        out.append(dict(variant="rerun", batch=0, modularity=float(q),
                        communities=int(comm), passes=None, affected=None,
                        time_s=float(t)))
    for m in RE_NX_BATCH.finditer(text):
        b, _n, _e, q, comm, t = m.groups()
        out.append(dict(variant="rerun", batch=int(b), modularity=float(q),
                        communities=int(comm), passes=None, affected=None,
                        time_s=float(t)))
    return out


def parse_log(method, path):
    text = open(path, errors="replace").read()
    if method == "df":
        res = parse_df(text)
        return res if res else parse_nx(text)
    res = parse_nx(text)
    return res if res else parse_df(text)


def main():
    if not os.path.isdir(OUTPUTS_DIR):
        print(f"No outputs/ directory at {OUTPUTS_DIR}")
        return

    methods = [m for m in KNOWN_METHODS
               if os.path.isdir(os.path.join(OUTPUTS_DIR, m))]

    graphs = set()
    for meth in methods:
        for f in glob.glob(os.path.join(OUTPUTS_DIR, meth, "*.txt")):
            b = os.path.basename(f)[:-4]
            if not b.endswith("_communities"):
                graphs.add(b)
    graphs = sorted(graphs)

    rows = []
    for g in graphs:
        nodes, edges = read_nm(input_path(g))
        for meth in methods:
            path = os.path.join(OUTPUTS_DIR, meth, f"{g}.txt")
            if not os.path.isfile(path):
                continue
            results = parse_log(meth, path)
            if not results:
                print(f"  warn: could not parse metrics from {meth}/{g}.txt")
                continue
            for r in results:
                rows.append({
                    "graph": g, "nodes": nodes, "edges": edges, "method": meth,
                    "variant": r["variant"], "batch": r["batch"],
                    "modularity": round(r["modularity"], 6),
                    "communities": r["communities"],
                    "passes": r["passes"] if r["passes"] is not None else "",
                    "affected": r["affected"] if r["affected"] is not None else "",
                    "time_s": round(r["time_s"], 6),
                })

    rows.sort(key=lambda r: (r["graph"], METHOD_ORDER.get(r["method"], 9),
                             r["batch"], VARIANT_ORDER.get(r["variant"], 9)))

    cols = ["graph", "nodes", "edges", "method", "variant", "batch",
            "modularity", "communities", "passes", "affected", "time_s"]
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
