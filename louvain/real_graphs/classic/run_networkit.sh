#!/bin/bash
# Run the NetworKit (parallel CPU) Louvain baseline on the classic small benchmark
# graphs (Zachary karate, dolphins).
#
# Requires NetworKit:  pip install networkit
# Inputs:  graphs/<name>_converted.txt  (static CUDA input, 0 batches) from prepare_classic.py.
# Output:  outputs/networkit/<name>.txt  (modularity / communities / time)
#
# Usage:
#   cd louvain/real_graphs/classic && bash run_networkit.sh
#   Or:  bash real_graphs/classic/run_networkit.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/networkit"

NK_SCRIPT="$ALG_DIR/networkit_louvain.py"
SEED=42

GRAPHS=(karate dolphins)

mkdir -p "$OUT_DIR"

if [ ! -f "$NK_SCRIPT" ]; then
    echo "ERROR: networkit_louvain.py not found at $NK_SCRIPT"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}_converted.txt"

    echo ""
    echo "=========================================="
    echo "  $name   (NetworKit)"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (run 'py -3 prepare_classic.py' first)"
        continue
    fi

    python "$NK_SCRIPT" "$input" --seed "$SEED" \
        > "$OUT_DIR/${name}.txt" 2>&1
    if [ $? -eq 0 ]; then
        echo "  NetworKit: OK  (log: $OUT_DIR/${name}.txt)"
    else
        echo "  NetworKit: FAILED (see $OUT_DIR/${name}.txt)"
    fi
done

echo ""
echo "=========================================="
echo "  NetworKit classic benchmarks complete."
echo "  Outputs in: $OUT_DIR/"
echo "=========================================="
