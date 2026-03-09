# Graph generator for static and dynamic Louvain community detection.
#
# Examples:
#   # Dynamic graph (300 nodes, 3 batches of 10 deletions + 15 insertions each):
#   python graphgen.py --nodes 300 --prob 0.02 --batches 3 --deletions 10 --insertions 15 -o ../test_dynamic_input.txt
#
#   # Large dynamic graph to stdout:
#   python graphgen.py --nodes 1000 --prob 0.01 --batches 5 --deletions 20 --insertions 30
#
#   # Static graph only (no batch updates):
#   python graphgen.py --nodes 500 --prob 0.03 --static -o ../static_input.txt
#
#   # Reproducible run with a fixed seed:
#   python graphgen.py --nodes 200 --prob 0.05 --batches 2 --deletions 5 --insertions 10 --seed 42 -o ../test.txt
#
#   # Custom edge weights:
#   python graphgen.py --nodes 100 --prob 0.1 --min-weight 1 --max-weight 100 --static

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


def format_initial_graph(G):
    """Format the initial graph as 'n_nodes m_edges' followed by edge lines."""
    lines = []
    lines.append(f"{G.number_of_nodes()} {G.number_of_edges()}")
    for u, v, data in G.edges(data=True):
        lines.append(f"{u} {v} {data['weight']}")
    return lines


def generate_batch(G, n_deletions, n_insertions, min_weight, max_weight):
    """
    Generate a batch of edge deletions and insertions on graph G.
    Deletions are randomly chosen from existing edges.
    Insertions are randomly chosen non-edges.
    G is modified in-place to reflect the changes.
    Returns (deletion_lines, insertion_lines).
    """
    edges = list(G.edges(data=True))
    non_edges_sample = []

    # Pick random deletions from existing edges
    actual_deletions = min(n_deletions, len(edges))
    del_edges = random.sample(edges, actual_deletions)
    del_lines = []
    for u, v, data in del_edges:
        del_lines.append(f"{u} {v} {data['weight']}")
        G.remove_edge(u, v)

    # Pick random insertions from non-edges
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    attempts = 0
    ins_lines = []
    added = 0
    while added < n_insertions and attempts < n_insertions * 20:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v and not G.has_edge(u, v):
            w = random.randint(min_weight, max_weight)
            ins_lines.append(f"{u} {v} {w}")
            G.add_edge(u, v, weight=w)
            added += 1
        attempts += 1

    return del_lines, ins_lines


def generate_dynamic_graph(num_nodes, prob_edge, n_batches,
                           deletions_per_batch, insertions_per_batch,
                           min_weight=1, max_weight=10):
    """
    Generate a complete dynamic graph input file:
      - Initial graph
      - n_batches batch updates (deletions + insertions each)
    Returns the full output as a string.
    """
    G = create_weighted_random_graph(num_nodes, prob_edge, min_weight, max_weight)
    lines = format_initial_graph(G)
    lines.append(str(n_batches))

    for _ in range(n_batches):
        del_lines, ins_lines = generate_batch(
            G, deletions_per_batch, insertions_per_batch, min_weight, max_weight)
        lines.append(f"{len(del_lines)} {len(ins_lines)}")
        lines.extend(del_lines)
        lines.extend(ins_lines)

    return "\n".join(lines)


def generate_static_graph(num_nodes, prob_edge, min_weight=1, max_weight=10):
    """Generate a static graph input file (no batches)."""
    G = create_weighted_random_graph(num_nodes, prob_edge, min_weight, max_weight)
    return "\n".join(format_initial_graph(G))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate graph inputs for static and dynamic Louvain")
    parser.add_argument("--nodes", type=int, default=300,
                        help="Number of nodes (default: 300)")
    parser.add_argument("--prob", type=float, default=0.02,
                        help="Edge probability for Erdos-Renyi model (default: 0.02)")
    parser.add_argument("--min-weight", type=int, default=1,
                        help="Minimum edge weight (default: 1)")
    parser.add_argument("--max-weight", type=int, default=10,
                        help="Maximum edge weight (default: 10)")
    parser.add_argument("--batches", type=int, default=3,
                        help="Number of dynamic batch updates (default: 3)")
    parser.add_argument("--deletions", type=int, default=10,
                        help="Deletions per batch (default: 10)")
    parser.add_argument("--insertions", type=int, default=15,
                        help="Insertions per batch (default: 15)")
    parser.add_argument("--static", action="store_true",
                        help="Generate static input only (no batches)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file (default: stdout)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.static:
        result = generate_static_graph(
            args.nodes, args.prob, args.min_weight, args.max_weight)
    else:
        result = generate_dynamic_graph(
            args.nodes, args.prob, args.batches,
            args.deletions, args.insertions,
            args.min_weight, args.max_weight)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(result)
        print(f"Written to {args.output}")
    else:
        print(result)