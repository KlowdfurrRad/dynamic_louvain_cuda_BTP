#!/bin/bash
# Run the NetworkX CPU baseline on the classic small benchmark graphs
# (Zachary karate, dolphins). These are tiny, so NetworkX runs instantly.
#
# Inputs:  graphs/<name>_converted.txt  (static CUDA input, 0 batches) produced by
#          prepare_classic.py. nx_louvain.py parses it and runs a single Louvain.
# Output:  outputs/networkx/<name>.txt   (modularity / communities / time)
#
# Usage:
#   cd louvain/real_graphs/classic && bash run_nx.sh
#   Or:  bash real_graphs/classic/run_nx.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/networkx"

NX_SCRIPT="$ALG_DIR/nx_louvain.py"
SEED=42

GRAPHS=(karate dolphins)

mkdir -p "$OUT_DIR"

if [ ! -f "$NX_SCRIPT" ]; then
    echo "ERROR: nx_louvain.py not found at $NX_SCRIPT"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}_converted.txt"

    echo ""
    echo "=========================================="
    echo "  $name   (NetworkX)"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (run 'py -3 prepare_classic.py' first)"
        continue
    fi

    python "$NX_SCRIPT" "$input" --seed "$SEED" \
        > "$OUT_DIR/${name}.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  NetworkX: OK  (log: $OUT_DIR/${name}.txt)"
    else
        echo "  NetworkX: FAILED (see $OUT_DIR/${name}.txt)"
    fi
done

echo ""
echo "=========================================="
echo "  NetworkX classic benchmarks complete."
echo "  Outputs in: $OUT_DIR/"
echo "=========================================="
