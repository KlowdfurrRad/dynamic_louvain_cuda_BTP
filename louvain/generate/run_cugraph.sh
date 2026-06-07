#!/bin/bash
# Run the cuGraph (GPU) Louvain baseline on the synthetic Erdos-Renyi dynamic
# graphs in generate/graphs/.
#
# Requires RAPIDS (cudf / cugraph / cupy) -- Linux/WSL/Colab with an NVIDIA GPU.
# Does NOT generate any graphs -- runs cugraph_louvain.py on the already-generated
# dynamic inputs (graphs/<name>.txt; initial graph + batches). cuGraph re-runs
# Louvain from scratch after each batch.
#
# Usage:
#   cd louvain/generate && bash run_cugraph.sh
#   Or:  bash generate/run_cugraph.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$SCRIPT_DIR")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/cugraph"

CUGRAPH_SCRIPT="$ALG_DIR/cugraph_louvain.py"

# Generated dynamic-input basenames to run (must already exist in graphs/).
# These match the names produced by run_benchmarks.sh's GRAPH_CONFIGS.
GRAPHS=(
    er_n100_p0.10_b5_bp0.001
    er_n500_p0.05_b5_bp0.001
    er_n1000_p0.02_b5_bp0.001
    er_n3000_p0.01_b5_bp0.001
    er_n10000_p0.005_b5_bp0.001
)

mkdir -p "$OUT_DIR"

if [ ! -f "$CUGRAPH_SCRIPT" ]; then
    echo "ERROR: cugraph_louvain.py not found at $CUGRAPH_SCRIPT"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}.txt"

    echo ""
    echo "=========================================="
    echo "  $name   (cuGraph)"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (generate it first; this script does not generate)"
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
echo "  cuGraph synthetic-graph benchmarks complete."
echo "  Outputs in: $OUT_DIR/"
echo "=========================================="
