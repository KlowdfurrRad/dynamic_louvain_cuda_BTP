#!/bin/bash
# Run the NetworkX CPU baseline on the large SNAP graphs in LargeSnap/.
#
# Inputs:  graphs/<name>_converted.txt  --- converted CUDA format ("n m" header,
#          "u v" edges, trailing "0" for n_batches). nx_louvain.py parses this and,
#          with 0 batches, performs a single static Louvain run. Missing inputs skipped.
# Output:  outputs/networkx/<name>_nx.txt   (modularity / communities / time)
#
# WARNING: these graphs are very large -- com-LiveJournal (~4.0M nodes, ~34.7M edges)
# and com-Orkut (~3.1M nodes, ~117M edges). NetworkX needs tens to hundreds of GB of
# RAM and can take hours; it is likely impractical on a typical machine. Run on a
# high-memory node, or skip the NetworkX baseline for graphs this size.
#
# Usage:
#   cd louvain/real_graphs/LargeSnap && bash run_nx.sh
#   Or:  bash real_graphs/LargeSnap/run_nx.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/networkx"

NX_SCRIPT="$ALG_DIR/nx_louvain.py"
SEED=42

GRAPHS=(com-LiveJournal com-Orkut)

mkdir -p "$OUT_DIR"

if [ ! -f "$NX_SCRIPT" ]; then
    echo "ERROR: nx_louvain.py not found at $NX_SCRIPT"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}_converted.txt"

    echo ""
    echo "=========================================="
    echo "  NetworkX on $name"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (convert it first)"
        continue
    fi

    echo "  Running NetworkX (this may take a long time) ..."
    python "$NX_SCRIPT" "$input" --seed "$SEED" \
        > "$OUT_DIR/${name}.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  NetworkX: OK  (log: $OUT_DIR/${name}.txt)"
    else
        echo "  NetworkX: FAILED (see $OUT_DIR/${name}.txt)"
    fi
done
