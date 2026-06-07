#!/bin/bash
# Run ONLY the from-scratch df_louvain binary on the static SNAP graphs in snap/.
#
# This script does NOT generate or convert any graphs. It runs df_louvain on the
# already-converted *_converted.txt inputs in snap/graphs/. If an input is
# missing, create it separately (e.g. with convert_snap_to_dynamic.py) and re-run.
#
# These converted inputs carry 0 batches, so df_louvain exercises its STATIC
# Louvain path here (useful to compare static quality/speed against the old
# static_louvain.exe and the NetworkX baseline). For dynamic DF/DS runs use
# snap_temporal/run_df.sh instead.
#
# Usage:
#   cd louvain/real_graphs/snap && bash run_df.sh
#   Or:  bash real_graphs/snap/run_df.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
SNAP_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/df"

# Locate the df_louvain binary (.exe on Windows).
DF="$ALG_DIR/df_louvain.exe"
[ -f "$DF" ] || DF="$ALG_DIR/df_louvain"

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

if [ ! -f "$DF" ]; then
    echo "ERROR: df_louvain not found at $ALG_DIR. Compile first:"
    echo "  cd $ALG_DIR"
    echo "  nvcc -O3 -arch=sm_86 df_louvain.cu -o df_louvain"
    exit 1
fi

for graph in "${GRAPHS[@]}"; do
    input="$SNAP_DIR/${GRAPH_FILES[$graph]}"

    echo ""
    echo "=========================================="
    echo "  $graph   (df_louvain only)"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (convert it first; this script does not convert)"
        continue
    fi

    echo "  Running df_louvain ..."
    "$DF" "$OUT_DIR/${graph}_communities.txt" \
        < "$input" \
        > "$OUT_DIR/${graph}.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  df_louvain: OK  (log: $OUT_DIR/${graph}.txt)"
    else
        echo "  df_louvain: FAILED (see $OUT_DIR/${graph}.txt)"
    fi
done

echo ""
echo "=========================================="
echo "  df_louvain static-graph benchmarks complete."
echo "  Outputs in: $OUT_DIR/"
echo "=========================================="
