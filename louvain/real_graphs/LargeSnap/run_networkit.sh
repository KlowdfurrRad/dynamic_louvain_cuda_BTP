#!/bin/bash
# Run the NetworKit (parallel CPU) Louvain baseline on the large SNAP graphs in LargeSnap/.
#
# Inputs:  graphs/<name>_converted.txt  --- converted CUDA format ("n m" header,
#          "u v" edges, trailing "0" for n_batches). networkit_louvain.py parses this
#          and performs a single static Louvain run. Missing inputs are skipped.
# Output:  outputs/networkit/<name>_networkit.txt   (modularity / communities / time)
#
# Requires NetworKit:  pip install networkit
# WARNING: these graphs are large -- com-LiveJournal (~4.0M nodes, ~34.7M edges) and
# com-Orkut (~3.1M nodes, ~117M edges). NetworKit's PLM itself is fast (parallel, C++),
# but networkit_louvain.py loads the full edge list in Python first, so com-Orkut needs
# tens of GB of RAM -- run it on a high-memory node.
#
# Usage:
#   cd louvain/real_graphs/LargeSnap && bash run_networkit.sh
#   Or:  bash real_graphs/LargeSnap/run_networkit.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/networkit"

NK_SCRIPT="$ALG_DIR/networkit_louvain.py"
SEED=42

GRAPHS=(com-LiveJournal com-Orkut)

mkdir -p "$OUT_DIR"

if [ ! -f "$NK_SCRIPT" ]; then
    echo "ERROR: networkit_louvain.py not found at $NK_SCRIPT"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}_converted.txt"

    echo ""
    echo "=========================================="
    echo "  NetworKit on $name"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (convert it first)"
        continue
    fi

    echo "  Running NetworKit ..."
    python "$NK_SCRIPT" "$input" --seed "$SEED" \
        > "$OUT_DIR/${name}.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  NetworKit: OK  (log: $OUT_DIR/${name}.txt)"
    else
        echo "  NetworKit: FAILED (see $OUT_DIR/${name}.txt)"
    fi
done
