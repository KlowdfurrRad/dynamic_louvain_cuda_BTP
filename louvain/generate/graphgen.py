# Graph generator for static and dynamic Louvain community detection.
#
# Produces up to TWO views of the same generated graph:
#   * raw / weighted          : edge lines are "u v w"  (random integer weights)
#   * program-compatible      : edge lines are "u v"    (no weights)
# The Louvain binaries (cuda_static_louvain, cuda_dynamic_louvain[_nodebased],
# df_louvain) read only "u v" and treat every weight as 1, so they consume the
# program-compatible file; the weighted raw file is kept for reference/other use.
#
# Output file names are taken as parameters:
#   -o / --output PATH     program-compatible (unweighted) file
#   --raw-output PATH      raw (weighted) file
# If neither is given, the program-compatible graph is printed to stdout.
#
# Examples:
#   # Dynamic graph, write both views:
#   python graphgen.py --nodes 300 --prob 0.02 --batches 3 --deletions 10 --insertions 15 \
#       -o graphs/g.txt --raw-output graphs/g_raw.txt
#
#   # Program-compatible graph to stdout:
#   python graphgen.py --nodes 1000 --prob 0.01 --batches 5 --deletions 20 --insertions 30
#
#   # Static graph only (no batch updates), both views:
#   python graphgen.py --nodes 500 --prob 0.03 --static -o g.txt --raw-output g_raw.txt
#
#   # Reproducible run with a fixed seed:
#   python graphgen.py --nodes 200 --prob 0.05 --batches 2 --deletions 5 --insertions 10 --seed 42 -o t.txt

import networkx as nx
import random
import argparse
import os


def create_weighted_random_graph(num_nodes, probability_of_edge, min_weight, max_weight):
    """Create a weighted Erdos-Renyi random graph."""
    G = nx.erdos_renyi_graph(num_nodes, probability_of_edge)
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(min_weight, max_weight)
    return G


def generate_batch(G, n_deletions, n_insertions, min_weight, max_weight):
    """
    Generate a batch of edge deletions and insertions on graph G.
    Deletions are randomly chosen from existing edges.
    Insertions are randomly chosen non-edges.
    G is modified in-place. Returns (del_edges, ins_edges), each a list of
    (u, v, w) tuples.
    """
    edges = list(G.edges(data=True))

    actual_deletions = min(n_deletions, len(edges))
    del_sample = random.sample(edges, actual_deletions)
    del_edges = []
    for u, v, data in del_sample:
        del_edges.append((u, v, data['weight']))
        G.remove_edge(u, v)

    nodes = list(G.nodes())
    ins_edges = []
    added = 0
    attempts = 0
    while added < n_insertions and attempts < n_insertions * 20:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v and not G.has_edge(u, v):
            w = random.randint(min_weight, max_weight)
            ins_edges.append((u, v, w))
            G.add_edge(u, v, weight=w)
            added += 1
        attempts += 1

    return del_edges, ins_edges


def build_dynamic(num_nodes, prob_edge, n_batches,
                  deletions_per_batch, insertions_per_batch,
                  min_weight, max_weight):
    """Build the structured dynamic graph: (n_nodes, n_initial, initial, batches).
    initial : list of (u, v, w); batches : list of (del_edges, ins_edges)."""
    G = create_weighted_random_graph(num_nodes, prob_edge, min_weight, max_weight)
    n_nodes = G.number_of_nodes()
    n_initial = G.number_of_edges()
    initial = [(u, v, data['weight']) for u, v, data in G.edges(data=True)]

    batches = []
    for _ in range(n_batches):
        del_edges, ins_edges = generate_batch(
            G, deletions_per_batch, insertions_per_batch, min_weight, max_weight)
        batches.append((del_edges, ins_edges))
    return n_nodes, n_initial, initial, batches


def build_static(num_nodes, prob_edge, min_weight, max_weight):
    """Build the structured static graph: (n_nodes, n_initial, initial, None)."""
    G = create_weighted_random_graph(num_nodes, prob_edge, min_weight, max_weight)
    initial = [(u, v, data['weight']) for u, v, data in G.edges(data=True)]
    return G.number_of_nodes(), G.number_of_edges(), initial, None


def format_graph(n_nodes, n_initial, initial, batches, weighted):
    """Format the structured graph as text. weighted=True emits 'u v w' edge
    lines, weighted=False emits 'u v'. batches=None => static (no batch section)."""
    def edge_line(u, v, w):
        return f"{u} {v} {w}" if weighted else f"{u} {v}"

    lines = [f"{n_nodes} {n_initial}"]
    for u, v, w in initial:
        lines.append(edge_line(u, v, w))

    if batches is not None:
        lines.append(str(len(batches)))
        for del_edges, ins_edges in batches:
            lines.append(f"{len(del_edges)} {len(ins_edges)}")
            for u, v, w in del_edges:
                lines.append(edge_line(u, v, w))
            for u, v, w in ins_edges:
                lines.append(edge_line(u, v, w))
    return "\n".join(lines)


def _write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate graph inputs for static and dynamic Louvain "
                    "(raw weighted and/or program-compatible unweighted views)")
    parser.add_argument("--nodes", type=int, default=300,
                        help="Number of nodes (default: 300)")
    parser.add_argument("--prob", type=float, default=0.02,
                        help="Edge probability for Erdos-Renyi model (default: 0.02)")
    parser.add_argument("--min-weight", type=int, default=1,
                        help="Minimum edge weight for the raw view (default: 1)")
    parser.add_argument("--max-weight", type=int, default=10,
                        help="Maximum edge weight for the raw view (default: 10)")
    parser.add_argument("--batches", type=int, default=3,
                        help="Number of dynamic batch updates (default: 3)")
    parser.add_argument("--deletions", type=int, default=10,
                        help="Deletions per batch (default: 10)")
    parser.add_argument("--insertions", type=int, default=15,
                        help="Insertions per batch (default: 15)")
    parser.add_argument("--static", action="store_true",
                        help="Generate static input only (no batches)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="program-compatible (unweighted) output file")
    parser.add_argument("--raw-output", type=str, default=None,
                        help="raw (weighted) output file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.static:
        n_nodes, n_initial, initial, batches = build_static(
            args.nodes, args.prob, args.min_weight, args.max_weight)
    else:
        n_nodes, n_initial, initial, batches = build_dynamic(
            args.nodes, args.prob, args.batches,
            args.deletions, args.insertions,
            args.min_weight, args.max_weight)

    prog_text = format_graph(n_nodes, n_initial, initial, batches, weighted=False)
    raw_text  = format_graph(n_nodes, n_initial, initial, batches, weighted=True)

    wrote_any = False
    if args.raw_output:
        _write(args.raw_output, raw_text)
        print(f"Written raw (weighted) to {args.raw_output}")
        wrote_any = True
    if args.output:
        _write(args.output, prog_text)
        print(f"Written program-compatible (unweighted) to {args.output}")
        wrote_any = True
    if not wrote_any:
        # Default: emit the program-compatible (unweighted) graph to stdout.
        print(prog_text)
