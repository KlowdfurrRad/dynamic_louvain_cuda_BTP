#!/bin/bash
# Run the NetworKit (parallel CPU) Louvain baseline on the SNAP temporal graphs.
#
# Requires NetworKit:  pip install networkit
# Does NOT generate or convert graphs -- runs networkit_louvain.py on the converted
# dynamic inputs in converted/ (initial graph + batches). NetworKit's PLM has no
# incremental mode, so it re-runs Louvain from scratch after each batch.
#
# Usage:
#   cd louvain/real_graphs/snap_temporal && bash run_networkit.sh
#   Or:  bash real_graphs/snap_temporal/run_networkit.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ALG_DIR="$LOUVAIN_DIR/algorithm"
CONVERTED_DIR="$SCRIPT_DIR/converted"
OUT_DIR="$SCRIPT_DIR/outputs/networkit"

NK_SCRIPT="$ALG_DIR/networkit_louvain.py"
SEED=42

# Converted dynamic-input basenames to run (must already exist in converted/).
GRAPHS=(CollegeMsg sx-mathover sx-askubuntu sx-superuser)

mkdir -p "$OUT_DIR"

if [ ! -f "$NK_SCRIPT" ]; then
    echo "ERROR: networkit_louvain.py not found at $NK_SCRIPT"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$CONVERTED_DIR/${name}.txt"

    echo ""
    echo "=========================================="
    echo "  $name   (NetworKit)"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (convert it first; this script does not convert)"
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
echo "  NetworKit temporal benchmarks complete."
echo "  Outputs in: $OUT_DIR/"
echo "=========================================="
