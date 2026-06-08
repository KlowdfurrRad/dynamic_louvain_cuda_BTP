#!/bin/bash
# Run the cuGraph (GPU) Louvain baseline on the synthetic-batch com-DBLP graphs.
# cugraph_louvain.py parses the dynamic input and re-clusters the whole graph after
# every batch.
#
# Inputs:  graphs/<name>.txt  (initial graph + 5 batches). Missing inputs are skipped.
# Output:  outputs/cugraph/<name>.txt
#
# Requires RAPIDS (cudf / cugraph / cupy) -- Linux/WSL/Colab with an NVIDIA GPU.
#
# Usage:
#   cd louvain/real_graphs/DBLP_syn_batches && bash run_cugraph.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/cugraph"

CUGRAPH_SCRIPT="$ALG_DIR/cugraph_louvain.py"

GRAPHS=(com-dblp-5-2 com-dblp-5-5 com-dblp-5-10 com-dblp-5-20)

mkdir -p "$OUT_DIR"

if [ ! -f "$CUGRAPH_SCRIPT" ]; then
    echo "ERROR: cugraph_louvain.py not found at $CUGRAPH_SCRIPT"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}.txt"

    echo ""
    echo "=========================================="
    echo "  cuGraph on $name"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (run generate_batches.sh first)"
        continue
    fi

    python "$CUGRAPH_SCRIPT" "$input" \
        > "$OUT_DIR/${name}.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  cuGraph: OK  (log: $OUT_DIR/${name}.txt)"
    else
        echo "  cuGraph: FAILED (see $OUT_DIR/${name}.txt)"
    fi
done
