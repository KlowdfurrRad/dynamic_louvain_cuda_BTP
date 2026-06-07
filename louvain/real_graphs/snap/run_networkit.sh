#!/bin/bash
# Run the NetworKit (parallel CPU) Louvain baseline on the static SNAP graphs in snap/.
#
# Requires NetworKit:  pip install networkit
# Does NOT generate or convert graphs -- runs networkit_louvain.py on the already-
# converted *_converted.txt inputs in snap/graphs/. NetworKit's PLM is multi-threaded,
# so (unlike NetworkX) the large graphs (com-Youtube, web-Google) are included.
#
# Usage:
#   cd louvain/real_graphs/snap && bash run_networkit.sh
#   Or:  bash real_graphs/snap/run_networkit.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
SNAP_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/networkit"

NK_SCRIPT="$ALG_DIR/networkit_louvain.py"
SEED=42

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

if [ ! -f "$NK_SCRIPT" ]; then
    echo "ERROR: networkit_louvain.py not found at $NK_SCRIPT"
    exit 1
fi

for graph in "${GRAPHS[@]}"; do
    input="$SNAP_DIR/${GRAPH_FILES[$graph]}"

    echo ""
    echo "=========================================="
    echo "  $graph   (NetworKit)"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found"
        continue
    fi

    python "$NK_SCRIPT" "$input" --seed "$SEED" \
        > "$OUT_DIR/${graph}.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  NetworKit: OK  (log: $OUT_DIR/${graph}.txt)"
    else
        echo "  NetworKit: FAILED (see $OUT_DIR/${graph}.txt)"
    fi
done

echo ""
echo "=========================================="
echo "  NetworKit SNAP benchmarks complete."
echo "  Outputs in: $OUT_DIR/"
echo "=========================================="
