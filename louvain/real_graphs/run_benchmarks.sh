#!/bin/bash
# Run the three CUDA Louvain implementations plus the NetworkX baseline on all
# SNAP graphs. Generates output logs and community assignment files.
#
# Usage: cd louvain/real_graphs && bash run_benchmarks.sh
#        Or: bash real_graphs/run_benchmarks.sh  (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$SCRIPT_DIR")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
SNAP_DIR="$SCRIPT_DIR/snap/graphs"
OUT_DIR="$SCRIPT_DIR/snap/outputs"

STATIC="$ALG_DIR/static_louvain.exe"
DYNAMIC="$ALG_DIR/dynamic_louvain.exe"
NODEBASED="$ALG_DIR/dynamic_louvain_nodebased.exe"
NX_SCRIPT="$ALG_DIR/nx_louvain.py"

SEED=42
NX_SKIP_ABOVE=2000000  # skip NetworkX above this many initial edges (too slow)

# Graph name -> converted file name
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
)

GRAPHS=(ca-GrQc facebook ca-HepTh ca-HepPh ca-AstroPh email-Enron com-amazon com-dblp)

# Create output directories
mkdir -p "$OUT_DIR/cuda_static" "$OUT_DIR/normal" "$OUT_DIR/node_based" "$OUT_DIR/networkx"

# Check executables exist
for exe in "$STATIC" "$DYNAMIC" "$NODEBASED"; do
    if [ ! -f "$exe" ]; then
        echo "ERROR: $exe not found. Compile first:"
        echo "  cd $ALG_DIR"
        echo "  nvcc -rdc=true -arch=sm_60 cuda_static_louvain.cu -o static_louvain"
        echo "  nvcc -rdc=true -arch=sm_60 cuda_dynamic_louvain.cu -o dynamic_louvain"
        echo "  nvcc -rdc=true -arch=sm_60 cuda_dynamic_louvain_nodebased.cu -o dynamic_louvain_nodebased"
        exit 1
    fi
done

if [ ! -f "$NX_SCRIPT" ]; then
    echo "ERROR: nx_louvain.py not found at $NX_SCRIPT"
    exit 1
fi

for graph in "${GRAPHS[@]}"; do
    input="$SNAP_DIR/${GRAPH_FILES[$graph]}"
    if [ ! -f "$input" ]; then
        echo "SKIP: $input not found"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  $graph"
    echo "=========================================="

    initial_edges=$(awk 'NR==1 { print $2 }' "$input")

    # # --- Static ---
    # echo "  [1/4] Static..."
    # "$STATIC" "$OUT_DIR/cuda_static/${graph}_communities.txt" \
    #     < "$input" \
    #     > "$OUT_DIR/cuda_static/${graph}.txt" 2>&1
    # if [ $? -eq 0 ]; then
    #     echo "  [1/4] Static: OK"
    # else
    #     echo "  [1/4] Static: FAILED (see $OUT_DIR/cuda_static/${graph}.txt)"
    # fi

    # # --- Dynamic (edge-based) ---
    # echo "  [2/4] Dynamic (edge)..."
    # "$DYNAMIC" "$OUT_DIR/normal/${graph}_communities.txt" \
    #     < "$input" \
    #     > "$OUT_DIR/normal/${graph}.txt" 2>&1
    # if [ $? -eq 0 ]; then
    #     echo "  [2/4] Dynamic (edge): OK"
    # else
    #     echo "  [2/4] Dynamic (edge): FAILED (see $OUT_DIR/normal/${graph}.txt)"
    # fi

    # # --- Dynamic (node-based) ---
    # echo "  [3/4] Dynamic (node)..."
    # "$NODEBASED" "$OUT_DIR/node_based/${graph}_communities.txt" \
    #     < "$input" \
    #     > "$OUT_DIR/node_based/${graph}.txt" 2>&1
    # if [ $? -eq 0 ]; then
    #     echo "  [3/4] Dynamic (node): OK"
    # else
    #     echo "  [3/4] Dynamic (node): FAILED (see $OUT_DIR/node_based/${graph}.txt)"
    # fi

    # --- NetworkX (CPU baseline; these SNAP inputs are static, 0 batches) ---
    if [ "$initial_edges" -gt "$NX_SKIP_ABOVE" ]; then
        echo "  [4/4] NetworkX: SKIP (initial edges $initial_edges > $NX_SKIP_ABOVE)"
    else
        echo "  [4/4] NetworkX..."
        python "$NX_SCRIPT" "$input" --seed "$SEED" \
            > "$OUT_DIR/networkx/${graph}.txt" 2>&1
        if [ $? -eq 0 ]; then
            echo "  [4/4] NetworkX: OK"
        else
            echo "  [4/4] NetworkX: FAILED (see $OUT_DIR/networkx/${graph}.txt)"
        fi
    fi
done

echo ""
echo "=========================================="
echo "  All benchmarks complete."
echo "  Outputs in: $OUT_DIR/{cuda_static,normal,node_based,networkx}/"
echo "  Run 'python results_table.py' for the comparison table."
echo "=========================================="