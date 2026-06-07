#!/usr/bin/env python3
"""Compact the per-method Louvain metrics in snap/outputs/ into one CSV.

Scans snap/outputs/<method>/<graph>.txt run logs (method = df / networkx /
cugraph) and writes snap/results.csv -- one tidy row per (graph, method, variant,
batch). The SNAP graphs here are static (0 batches), so each method has a single
result, but the schema also handles batched inputs (see snap_temporal/generate).

Log formats handled:
  df_louvain : "Static: Q=0.926020  communities=222  passes=6  time=174 ms"
               "[batch 1] ND: Q=.. comms=.. time=.. ms"
               "[batch 1] DF: Q=.. comms=.. affected0=.. time=.. ms"   (DF/DS only)
  nx/cugraph : "  initial : nodes=.. edges=.. Q=.. communities=.. time=..s"
               "  batch 1  : nodes=.. edges=.. Q=.. communities=.. time=..s"

Columns: graph, nodes, edges, method, variant, batch, modularity, communities,
         passes, affected, time_s
  variant : df -> Static / ND / DF / DS ;  nx,cugraph -> rerun
  batch   : 0 = initial/static, 1..N = batch number
  passes  : df Static only ;  affected : df DF/DS only ;  time normalised to seconds
  nodes/edges : the initial graph size (from graphs/<file>_converted.txt)

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

KNOWN_METHODS = ["df", "networkx", "networkit", "cugraph"]   # ignore old cuda_static/normal/node_based
METHOD_ORDER = {m: i for i, m in enumerate(KNOWN_METHODS)}
VARIANT_ORDER = {"Static": 0, "rerun": 0, "ND": 1, "DF": 2, "DS": 3}

# graph name -> converted input file (for nodes/edges); default <name>_converted.txt
GRAPH_FILES = {
    "ca-GrQc": "ca-GrQc_converted.txt",
    "facebook": "facebook_combined_converted.txt",
    "ca-HepTh": "ca-HepTh_converted.txt",
    "ca-HepPh": "ca-HepPh_converted.txt",
    "ca-AstroPh": "ca-AstroPh_converted.txt",
    "email-Enron": "email-Enron_converted.txt",
    "com-amazon": "com-amazon_converted.txt",
    "com-dblp": "com-dblp_converted.txt",
    "com-Youtube": "com-Youtube_converted.txt",
    "web-Google": "web-Google_converted.txt",
}


def input_path(graph):
    return os.path.join(GRAPHS_DIR, GRAPH_FILES.get(graph, f"{graph}_converted.txt"))


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
    """(nodes, edges) from the first line of the input graph, else (None, None)."""
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
