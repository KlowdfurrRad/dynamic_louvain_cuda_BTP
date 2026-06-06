#!/bin/bash
# Run ONLY the from-scratch df_louvain binary on the synthetic Erdos-Renyi
# dynamic graphs in generate/graphs/.
#
# This script does NOT generate any graphs. It runs df_louvain on the already
# generated CUDA-ready dynamic input files (graphs/<name>.txt). If an input is
# missing, generate it separately (e.g. with run_benchmarks.sh / graphgen.py)
# and re-run this script.
#
# df_louvain reads the dynamic input (initial graph + batches), runs Static on
# the initial graph and then Naive-Dynamic / Dynamic-Frontier / Delta-Screening
# on each batch, printing Q / communities / time per method, and writes the
# final DF community assignment to <name>_communities.txt.
#
# Usage:
#   cd louvain/generate && bash run_df.sh
#   Or:  bash generate/run_df.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$SCRIPT_DIR")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/df"

# Locate the df_louvain binary (.exe on Windows).
DF="$ALG_DIR/df_louvain.exe"
[ -f "$DF" ] || DF="$ALG_DIR/df_louvain"

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

if [ ! -f "$DF" ]; then
    echo "ERROR: df_louvain not found at $ALG_DIR. Compile first:"
    echo "  cd $ALG_DIR"
    echo "  nvcc -O3 -arch=sm_86 df_louvain.cu -o df_louvain"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}.txt"

    echo ""
    echo "=========================================="
    echo "  $name   (df_louvain only)"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (generate it first; this script does not generate)"
        continue
    fi

    echo "  Running df_louvain ..."
    "$DF" "$OUT_DIR/${name}_communities.txt" \
        < "$input" \
        > "$OUT_DIR/${name}.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  df_louvain: OK  (log: $OUT_DIR/${name}.txt)"
    else
        echo "  df_louvain: FAILED (see $OUT_DIR/${name}.txt)"
    fi
done

echo ""
echo "=========================================="
echo "  df_louvain synthetic-graph benchmarks complete."
echo "  Outputs in: $OUT_DIR/"
echo "=========================================="
