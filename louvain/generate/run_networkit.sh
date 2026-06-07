#!/bin/bash
# Run the NetworKit (parallel CPU) Louvain baseline on the synthetic Erdos-Renyi
# dynamic graphs in generate/graphs/.
#
# Requires NetworKit:  pip install networkit
# Does NOT generate any graphs -- runs networkit_louvain.py on the already-generated
# dynamic inputs (graphs/<name>.txt; initial graph + batches). NetworKit re-runs
# Louvain from scratch after each batch.
#
# Usage:
#   cd louvain/generate && bash run_networkit.sh
#   Or:  bash generate/run_networkit.sh   (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$SCRIPT_DIR")"
ALG_DIR="$LOUVAIN_DIR/algorithm"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
OUT_DIR="$SCRIPT_DIR/outputs/networkit"

NK_SCRIPT="$ALG_DIR/networkit_louvain.py"
SEED=42

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

if [ ! -f "$NK_SCRIPT" ]; then
    echo "ERROR: networkit_louvain.py not found at $NK_SCRIPT"
    exit 1
fi

for name in "${GRAPHS[@]}"; do
    input="$GRAPHS_DIR/${name}.txt"

    echo ""
    echo "=========================================="
    echo "  $name   (NetworKit)"
    echo "=========================================="

    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found (generate it first; this script does not generate)"
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
echo "  NetworKit synthetic-graph benchmarks complete."
echo "  Outputs in: $OUT_DIR/"
echo "=========================================="
