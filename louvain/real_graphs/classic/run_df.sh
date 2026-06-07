#!/bin/bash
# Run df_louvain on the classic small benchmark graphs (Zachary karate, dolphins).
#
# Inputs:  graphs/<name>_converted.txt  (static CUDA input, 0 batches) produced by
#          prepare_classic.py. Run that first if the inputs are missing.
# Output:  outputs/df/<name>.txt              (run log)
#          outputs/df/<name>_communities.txt  (community assignment)
#
# Usage:
#   cd louvain/real_graphs/classic && bash run_df.sh
#   Or:  bash real_graphs/classic/run_df.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/df"

# Locate the df_louvain binary (.exe on Windows).
DF="$ALG_DIR/df_louvain.exe"
[ -f "$DF" ] || DF="$ALG_DIR/df_louvain"

GRAPHS=(karate dolphins)

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
    echo "  $name   (df_louvain)"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (run 'py -3 prepare_classic.py' first)"
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

echo ""
echo "=========================================="
echo "  df_louvain classic benchmarks complete."
echo "  Outputs in: $OUT_DIR/"
echo "=========================================="
