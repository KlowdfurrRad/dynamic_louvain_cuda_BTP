#!/bin/bash
# Run the cuGraph (GPU) Louvain baseline on the SNAP temporal graphs.
#
# Requires RAPIDS (cudf / cugraph / cupy) -- Linux/WSL/Colab with an NVIDIA GPU.
# Does NOT generate or convert graphs -- runs cugraph_louvain.py on the converted
# dynamic inputs in converted/ (initial graph + batches). cuGraph has no
# incremental mode, so it re-runs Louvain from scratch after each batch.
#
# Usage:
#   cd louvain/real_graphs/snap_temporal && bash run_cugraph.sh
#   Or:  bash real_graphs/snap_temporal/run_cugraph.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ALG_DIR="$LOUVAIN_DIR/algorithm"
CONVERTED_DIR="$SCRIPT_DIR/converted"
OUT_DIR="$SCRIPT_DIR/outputs/cugraph"

CUGRAPH_SCRIPT="$ALG_DIR/cugraph_louvain.py"

# Converted dynamic-input basenames to run (must already exist in converted/).
GRAPHS=(CollegeMsg sx-mathover sx-askubuntu sx-superuser)

mkdir -p "$OUT_DIR"

if [ ! -f "$CUGRAPH_SCRIPT" ]; then
    echo "ERROR: cugraph_louvain.py not found at $CUGRAPH_SCRIPT"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$CONVERTED_DIR/${name}.txt"

    echo ""
    echo "=========================================="
    echo "  $name   (cuGraph)"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (convert it first; this script does not convert)"
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
echo "  cuGraph temporal benchmarks complete."
echo "  Outputs in: $OUT_DIR/"
echo "=========================================="
