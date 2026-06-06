"""
Parse outputs from all three Louvain implementations, compute true modularity
from the original graphs + community files, and print a comparison table.

Usage: python results_table.py
"""

import os
import re
from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(BASE_DIR, "snap", "outputs")
GRAPHS_DIR = os.path.join(BASE_DIR, "snap", "graphs")

GRAPHS = [
    "ca-GrQc", "facebook", "ca-HepTh", "ca-HepPh",
    "ca-AstroPh", "email-Enron", "com-amazon", "com-dblp"
]

# Map graph name to converted file name
GRAPH_FILES = {
    "ca-GrQc": "ca-GrQc_converted.txt",
    "facebook": "facebook_combined_converted.txt",
    "ca-HepTh": "ca-HepTh_converted.txt",
    "ca-HepPh": "ca-HepPh_converted.txt",
    "ca-AstroPh": "ca-AstroPh_converted.txt",
    "email-Enron": "email-Enron_converted.txt",
    "com-amazon": "com-amazon_converted.txt",
    "com-dblp": "com-dblp_converted.txt",
}

VARIANTS = [
    ("cuda_static", "Static"),
    ("normal", "Dyn (edge)"),
    ("node_based", "Dyn (node)"),
]


def load_graph(graph_name):
    """Load graph from converted file. Returns (n_nodes, adjacency dict of {node: [(neighbor, weight)]})."""
    filepath = os.path.join(GRAPHS_DIR, GRAPH_FILES[graph_name])
    adj = defaultdict(list)
    with open(filepath) as f:
        header = f.readline().split()
        n_nodes, m_edges = int(header[0]), int(header[1])
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                break  # hit the "0" (n_batches) line
            u, v = int(parts[0]), int(parts[1])
            w = 1.0
            adj[u].append((v, w))
            adj[v].append((u, w))
    return n_nodes, adj


def load_communities(comm_filepath):
    """Load community assignments from file. Returns dict {node: community}."""
    if not os.path.isfile(comm_filepath):
        return None
    community = {}
    with open(comm_filepath) as f:
        first_line = f.readline().strip()
        # First line might be community count (no space) or "node comm" pair
        if " " in first_line:
            parts = first_line.split()
            community[int(parts[0])] = int(parts[1])
        # else: it's the count header, skip it
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                community[int(parts[0])] = int(parts[1])
    return community


def compute_modularity(n_nodes, adj, community):
    """Compute modularity Q = sum_c [ L_c/m - (d_c/2m)^2 ]
    where L_c = number of intra-community edges (each counted once),
    d_c = sum of degrees in community c, m = total edges."""
    m = 0.0  # total edge weight (each undirected edge counted once)
    comm_internal = defaultdict(float)  # intra-community edge weight per community
    comm_degree = defaultdict(float)    # total degree per community

    for u in range(n_nodes):
        cu = community.get(u, u)
        degree_u = 0.0
        for v, w in adj[u]:
            degree_u += w
            if community.get(v, v) == cu:
                comm_internal[cu] += w  # counted from both sides, divide later
        comm_degree[cu] += degree_u
        m += degree_u

    # m is sum of all directed edge weights = 2 * undirected total
    # comm_internal counted from both endpoints, so it equals 2 * L_c
    Q = 0.0
    for c in comm_degree:
        L_c = comm_internal[c]  # = 2 * actual internal weight (double-counted)
        d_c = comm_degree[c]
        Q += (L_c / m) - (d_c / m) ** 2

    return Q


def parse_output(filepath):
    """Parse CUDA output log for reported modularity, time, and community count."""
    if not os.path.isfile(filepath):
        return None, None, None

    text = open(filepath).read()

    if "CUDA error" in text:
        return "CRASH", None, None

    mod_matches = re.findall(r"([-\d.e+]+) is the (?:current|final) modularity", text)
    modularity = float(mod_matches[-1]) if mod_matches else None

    time_match = re.search(r"(?:Static e|E)xecution time:\s*([\d.]+)\s*m(?:illi)?s", text)
    time_ms = float(time_match.group(1)) if time_match else None

    comm_match = re.search(r"final (\d+) communities", text)
    if comm_match:
        n_communities = int(comm_match.group(1))
    else:
        comm_file = filepath.replace(".txt", "_communities.txt")
        if os.path.isfile(comm_file):
            with open(comm_file) as f:
                first_line = f.readline().strip()
                if " " not in first_line:
                    n_communities = int(first_line)
                else:
                    comms = {first_line.split()[1]}
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            comms.add(parts[1])
                    n_communities = len(comms)
        else:
            fnm = re.search(r"Final number of nodes:\s*(\d+)", text)
            n_communities = int(fnm.group(1)) if fnm else None

    return modularity, time_ms, n_communities


def main():
    # Cache loaded graphs
    graph_cache = {}

    # Collect all results first
    results = {}  # (graph, variant_folder) -> (reported_Q, true_Q, time, comms)

    for graph in GRAPHS:
        for folder, _ in VARIANTS:
            filepath = os.path.join(OUTPUTS_DIR, folder, f"{graph}.txt")
            reported_Q, time_ms, n_communities = parse_output(filepath)

            true_Q = None
            if reported_Q not in (None, "CRASH"):
                comm_file = os.path.join(OUTPUTS_DIR, folder, f"{graph}_communities.txt")
                community = load_communities(comm_file)
                if community:
                    if graph not in graph_cache:
                        print(f"Loading graph {graph}...")
                        graph_cache[graph] = load_graph(graph)
                    n_nodes, adj = graph_cache[graph]
                    true_Q = compute_modularity(n_nodes, adj, community)

            results[(graph, folder)] = (reported_Q, true_Q, time_ms, n_communities)

    # Print table
    print()
    print(f"{'Graph':<16}", end="")
    for _, label in VARIANTS:
        print(f"{'Rep.Q':>8} {'True Q':>8} {'Time':>9} {'Comms':>7}", end="  ")
    print()
    print("-" * (16 + len(VARIANTS) * 36))

    for graph in GRAPHS:
        print(f"{graph:<16}", end="")
        for folder, _ in VARIANTS:
            reported_Q, true_Q, time_ms, n_communities = results[(graph, folder)]

            if reported_Q == "CRASH":
                print(f"{'CRASH':>8} {'':>8} {'':>9} {'':>7}", end="  ")
            elif reported_Q is not None:
                t_str = f"{time_ms:.0f}ms" if time_ms else "?"
                c_str = str(n_communities) if n_communities else "?"
                tq_str = f"{true_Q:.4f}" if true_Q is not None else "N/A"
                print(f"{reported_Q:>8.4f} {tq_str:>8} {t_str:>9} {c_str:>7}", end="  ")
            else:
                print(f"{'N/A':>8} {'':>8} {'':>9} {'':>7}", end="  ")
        print()


if __name__ == "__main__":
    main()