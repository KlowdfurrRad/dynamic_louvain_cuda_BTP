#!/bin/bash
# Run the cuGraph (GPU) Louvain baseline on the static SNAP graphs in snap/.
#
# Requires RAPIDS (cudf / cugraph / cupy) -- Linux/WSL/Colab with an NVIDIA GPU.
# Does NOT generate or convert graphs -- runs cugraph_louvain.py on the already-
# converted *_converted.txt inputs in snap/graphs/. Unlike NetworkX, cuGraph runs
# on the GPU, so the large graphs (com-Youtube, web-Google) are included.
#
# Usage:
#   cd louvain/real_graphs/snap && bash run_cugraph.sh
#   Or:  bash real_graphs/snap/run_cugraph.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
SNAP_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/cugraph"

CUGRAPH_SCRIPT="$ALG_DIR/cugraph_louvain.py"

# Graph name -> converted input file name (must already exist in snap/graphs/).
declare -A GRAPH_FILES
GRAPH_FILES=(
    [ca-GrQc]="ca-GrQc_converted.txt"
    [facebook]="facebook_combined_converted.txt"
    [ca-HepTh]="ca-HepTh_converted.txt"
    [ca-HepPh]="ca-HepPh_converted.txt"
    [ca-AstroPh]="ca-AstroPh_converted.txt"
    [email-Enron]="email-Enron_converted.txt"
    [com-amazon]="com-amazon_converted.txt"
    [com-dblp]="com-dblp_converted.txt"
    [com-Youtube]="com-Youtube_converted.txt"
    [web-Google]="web-Google_converted.txt"
)

# Graphs to run (smallest first). Edit this list to subset.
GRAPHS=(ca-GrQc facebook ca-HepTh ca-HepPh ca-AstroPh email-Enron com-amazon com-dblp com-Youtube web-Google)

mkdir -p "$OUT_DIR"

if [ ! -f "$CUGRAPH_SCRIPT" ]; then
    echo "ERROR: cugraph_louvain.py not found at $CUGRAPH_SCRIPT"
    exit 1
fi

for graph in "${GRAPHS[@]}"; do
    input="$SNAP_DIR/${GRAPH_FILES[$graph]}"

    echo ""
    echo "=========================================="
    echo "  $graph   (cuGraph)"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found"
        continue
    fi

    python "$CUGRAPH_SCRIPT" "$input" \
        > "$OUT_DIR/${graph}.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  cuGraph: OK  (log: $OUT_DIR/${graph}.txt)"
    else
        echo "  cuGraph: FAILED (see $OUT_DIR/${graph}.txt)"
    fi
done

echo ""
echo "=========================================="
echo "  cuGraph SNAP benchmarks complete."
echo "  Outputs in: $OUT_DIR/"
echo "=========================================="
