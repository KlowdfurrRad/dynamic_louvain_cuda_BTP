#!/bin/bash
# Run the cuGraph (GPU) Louvain baseline on the large SNAP graphs in LargeSnap/.
#
# Inputs:  graphs/<name>_converted.txt  --- converted CUDA format ("n m" header,
#          "u v" edges, trailing "0" for n_batches). cugraph_louvain.py parses this
#          and, with 0 batches, performs a single static Louvain run. Missing inputs skipped.
# Output:  outputs/cugraph/<name>_cugraph.txt  (modularity / communities / time)
#
# Requires RAPIDS (cudf / cugraph / cupy) -- Linux/WSL/Colab with an NVIDIA GPU.
# NOTE: these graphs are large -- com-LiveJournal (~4.0M nodes, ~34.7M edges) and
# com-Orkut (~3.1M nodes, ~117M edges) -- and need several to tens of GB of GPU
# memory. Fine on cluster/Colab GPUs (e.g. A100); may OOM on a small card.
#
# Usage:
#   cd louvain/real_graphs/LargeSnap && bash run_cugraph.sh
#   Or:  bash real_graphs/LargeSnap/run_cugraph.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/cugraph"

CUGRAPH_SCRIPT="$ALG_DIR/cugraph_louvain.py"

GRAPHS=(com-LiveJournal com-Orkut)

mkdir -p "$OUT_DIR"

if [ ! -f "$CUGRAPH_SCRIPT" ]; then
    echo "ERROR: cugraph_louvain.py not found at $CUGRAPH_SCRIPT"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}_converted.txt"

    echo ""
    echo "=========================================="
    echo "  cuGraph on $name"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (convert it first)"
        continue
    fi

    python "$CUGRAPH_SCRIPT" "$input" \
        > "$OUT_DIR/${name}_cugraph.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  cuGraph: OK  (log: $OUT_DIR/${name}_cugraph.txt)"
    else
        echo "  cuGraph: FAILED (see $OUT_DIR/${name}_cugraph.txt)"
    fi
done
