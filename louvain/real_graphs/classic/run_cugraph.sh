#!/bin/bash
# Run the cuGraph (GPU) Louvain baseline on the classic small benchmark graphs
# (Zachary karate, dolphins).
#
# Requires RAPIDS (cudf / cugraph / cupy) -- a Linux/WSL/Colab box with an NVIDIA
# GPU; RAPIDS does not run on native Windows.
# Inputs:  graphs/<name>_converted.txt  (static CUDA input, 0 batches) from prepare_classic.py.
# Output:  outputs/cugraph/<name>.txt   (modularity / communities / time)
#
# Usage:
#   cd louvain/real_graphs/classic && bash run_cugraph.sh
#   Or:  bash real_graphs/classic/run_cugraph.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/cugraph"

CUGRAPH_SCRIPT="$ALG_DIR/cugraph_louvain.py"

GRAPHS=(karate dolphins)

mkdir -p "$OUT_DIR"

if [ ! -f "$CUGRAPH_SCRIPT" ]; then
    echo "ERROR: cugraph_louvain.py not found at $CUGRAPH_SCRIPT"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}_converted.txt"

    echo ""
    echo "=========================================="
    echo "  $name   (cuGraph)"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (run 'py -3 prepare_classic.py' first)"
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

echo ""
echo "=========================================="
echo "  cuGraph classic benchmarks complete."
echo "  Outputs in: $OUT_DIR/"
echo "=========================================="
