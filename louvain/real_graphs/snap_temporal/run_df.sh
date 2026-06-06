#!/bin/bash
# Run ONLY the from-scratch df_louvain binary (Static + ND + DF + DS in one
# process) on the SNAP temporal graphs.
#
# This script does NOT generate or convert any graphs. It runs df_louvain on the
# already-converted dynamic input files in converted/. If an input is missing,
# create it separately (e.g. with run_benchmarks.sh / convert_snap_temporal_to_dynamic.py)
# and re-run this script.
#
# df_louvain reads the dynamic input (initial graph + batches), runs Static on
# the initial graph and then Naive-Dynamic / Dynamic-Frontier / Delta-Screening
# on each batch, printing Q / communities / time per method, and writes the
# final DF community assignment to <name>_communities.txt.
#
# Usage:
#   cd louvain/real_graphs/snap_temporal && bash run_df.sh
#   Or:  bash real_graphs/snap_temporal/run_df.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ALG_DIR="$LOUVAIN_DIR/algorithm"

CONVERTED_DIR="$SCRIPT_DIR/converted"
OUT_DIR="$SCRIPT_DIR/outputs/df"

# Locate the df_louvain binary (.exe on Windows).
DF="$ALG_DIR/df_louvain.exe"
[ -f "$DF" ] || DF="$ALG_DIR/df_louvain"

# Converted dynamic-input basenames to run (must already exist in converted/).
GRAPHS=(CollegeMsg sx-mathover sx-askubuntu sx-superuser)

mkdir -p "$OUT_DIR"

if [ ! -f "$DF" ]; then
    echo "ERROR: df_louvain not found at $ALG_DIR. Compile first:"
    echo "  cd $ALG_DIR"
    echo "  nvcc -O3 -arch=sm_86 df_louvain.cu -o df_louvain"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$CONVERTED_DIR/${name}.txt"

    echo ""
    echo "=========================================="
    echo "  $name   (df_louvain only)"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (convert it first; this script does not convert)"
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
echo "  df_louvain temporal benchmarks complete."
echo "  Outputs in: $OUT_DIR/"
echo "=========================================="
