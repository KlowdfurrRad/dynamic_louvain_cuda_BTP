#!/bin/bash
# Run the NetworkX (single-thread CPU) Louvain baseline on the synthetic-batch
# email-Enron graphs. nx_louvain.py parses the dynamic input and re-clusters the whole
# graph after every batch.
#
# Inputs:  graphs/<name>.txt  (initial graph + 5 batches). Missing inputs are skipped.
# Output:  outputs/networkx/<name>.txt
#
# Usage:
#   cd louvain/real_graphs/Enron_syn_batches && bash run_nx.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/networkx"

NX_SCRIPT="$ALG_DIR/nx_louvain.py"
SEED=42

GRAPHS=(email-Enron-5-2 email-Enron-5-5 email-Enron-5-10)

mkdir -p "$OUT_DIR"

if [ ! -f "$NX_SCRIPT" ]; then
    echo "ERROR: nx_louvain.py not found at $NX_SCRIPT"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}.txt"

    echo ""
    echo "=========================================="
    echo "  NetworkX on $name"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (run generate_batches.sh first)"
        continue
    fi

    echo "  Running NetworkX ..."
    python "$NX_SCRIPT" "$input" --seed "$SEED" \
        > "$OUT_DIR/${name}.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  NetworkX: OK  (log: $OUT_DIR/${name}.txt)"
    else
        echo "  NetworkX: FAILED (see $OUT_DIR/${name}.txt)"
    fi
done
