#!/bin/bash
# Run df_louvain (Static + ND + DF + DS) on the synthetic-batch email-Enron graphs.
#
# Inputs:  graphs/<name>.txt  --- dynamic input (initial graph + 5 batches) produced by
#          generate_batches.sh. Missing inputs are skipped.
# Output:  outputs/df/<name>.txt  (per-batch Q / communities / time) and
#          outputs/df/<name>_communities.txt (final DF community assignment).
#
# Usage:
#   cd louvain/real_graphs/Enron_syn_batches && bash run_df.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/df"

# Locate the df_louvain binary (.exe on Windows).
DF="$ALG_DIR/df_louvain.exe"
[ -f "$DF" ] || DF="$ALG_DIR/df_louvain"

GRAPHS=(email-Enron-5-2 email-Enron-5-5 email-Enron-5-10)

mkdir -p "$OUT_DIR"

if [ ! -f "$DF" ]; then
    echo "ERROR: df_louvain not found at $ALG_DIR. Compile first:"
    echo "  cd $ALG_DIR && nvcc -O3 -arch=sm_86 df_louvain.cu -o df_louvain"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}.txt"

    echo ""
    echo "=========================================="
    echo "  df_louvain on $name"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (run generate_batches.sh first)"
        continue
    fi

    "$DF" "$OUT_DIR/${name}_communities.txt" \
        < "$input" \
        > "$OUT_DIR/${name}.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  df_louvain: OK  (log: $OUT_DIR/${name}.txt)"
    else
        echo "  df_louvain: FAILED (see $OUT_DIR/${name}.txt)"
    fi
done
