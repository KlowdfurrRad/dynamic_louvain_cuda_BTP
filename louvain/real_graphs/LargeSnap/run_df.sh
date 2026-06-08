#!/bin/bash
# Run the from-scratch df_louvain binary on the large SNAP graphs in LargeSnap/.
#
# Inputs:  graphs/<name>_converted.txt  --- converted CUDA format ("n m" header,
#          then "u v" edges, then a trailing "0" for n_batches), so df_louvain
#          exercises its STATIC Louvain path. Missing inputs are skipped.
# Output:  outputs/df/<name>_df.txt              (run log)
#          outputs/df/<name>_df_communities.txt  (final community assignment)
#
# NOTE: these graphs are large -- com-LiveJournal (~4.0M nodes, ~34.7M edges) and
# com-Orkut (~3.1M nodes, ~117M edges). Both are GPU-memory and time intensive and
# may exhaust GPU memory on smaller cards.
#
# Usage:
#   cd louvain/real_graphs/LargeSnap && bash run_df.sh
#   Or:  bash real_graphs/LargeSnap/run_df.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/df"

# Locate the df_louvain binary (.exe on Windows).
DF="$ALG_DIR/df_louvain.exe"
[ -f "$DF" ] || DF="$ALG_DIR/df_louvain"

GRAPHS=(com-Orkut)

mkdir -p "$OUT_DIR"

if [ ! -f "$DF" ]; then
    echo "ERROR: df_louvain not found at $ALG_DIR. Compile first:"
    echo "  cd $ALG_DIR && nvcc -O3 -arch=sm_86 df_louvain.cu -o df_louvain"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}_converted.txt"

    echo ""
    echo "=========================================="
    echo "  df_louvain on $name"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (convert it first)"
        continue
    fi

    "$DF" "$OUT_DIR/${name}_df_communities.txt" \
        < "$input" \
        > "$OUT_DIR/${name}_df.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  df_louvain: OK  (log: $OUT_DIR/${name}_df.txt)"
    else
        echo "  df_louvain: FAILED (see $OUT_DIR/${name}_df.txt)"
    fi
done
