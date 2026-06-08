#!/bin/bash
# Run the NetworKit (parallel CPU PLM) Louvain baseline on the synthetic-batch
# com-DBLP graphs. networkit_louvain.py parses the dynamic input and re-clusters the
# whole graph after every batch.
#
# Inputs:  graphs/<name>.txt  (initial graph + 5 batches). Missing inputs are skipped.
# Output:  outputs/networkit/<name>.txt
#
# Requires NetworKit:  pip install networkit
#
# Usage:
#   cd louvain/real_graphs/DBLP_syn_batches && bash run_networkit.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/networkit"

NK_SCRIPT="$ALG_DIR/networkit_louvain.py"
SEED=42

GRAPHS=(com-dblp-5-2 com-dblp-5-5 com-dblp-5-10)

mkdir -p "$OUT_DIR"

if [ ! -f "$NK_SCRIPT" ]; then
    echo "ERROR: networkit_louvain.py not found at $NK_SCRIPT"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}.txt"

    echo ""
    echo "=========================================="
    echo "  NetworKit on $name"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (run generate_batches.sh first)"
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
