#!/bin/bash
# Generate synthetic Erdos-Renyi graphs with dynamic batch updates and benchmark
# all Louvain implementations (static CUDA, dynamic CUDA edge-based,
# dynamic CUDA node-based, NetworkX).
# Mirror of real_graphs/run_benchmarks.sh, adapted for the generate/ directory.
#
# Each row of GRAPH_CONFIGS specifies one graph:
#   "<nodes> <prob> <n_batches> <batch_pct>"
# where batch_pct is the fraction of expected initial edges used for
# per-batch insertions (== per-batch deletions).
#
# Usage: cd louvain/generate && bash run_benchmarks.sh
#        Or: bash generate/run_benchmarks.sh  (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(dirname "$SCRIPT_DIR")"
ALG_DIR="$LOUVAIN_DIR/algorithm"

GEN_SCRIPT="$SCRIPT_DIR/graphgen.py"
NX_SCRIPT="$ALG_DIR/nx_louvain.py"
OUT_DIR="$SCRIPT_DIR/outputs"
GRAPHS_DIR="$SCRIPT_DIR/graphs"

STATIC="$ALG_DIR/static_louvain.exe"
DYNAMIC="$ALG_DIR/dynamic_louvain.exe"
NODEBASED="$ALG_DIR/dynamic_louvain_nodebased.exe"

# ============================================================================
# BENCHMARK CONFIGURATION — fill in per-graph values below
# ============================================================================
# Each row of GRAPH_CONFIGS specifies ONE graph:
#
#     "<nodes>  <prob>  <n_batches>  <batch_pct>"
#
#   nodes      — number of vertices in the initial Erdos-Renyi graph
#   prob       — edge probability for ER generation
#   n_batches  — number of dynamic batches to apply
#   batch_pct  — per-batch insertions = deletions = batch_pct * expected edges
#
# Dummy values (same across all rows) are filled in for now; tune per row as
# needed. Add or remove rows to change which graphs get benchmarked.
# ============================================================================
GRAPH_CONFIGS=(
    #  nodes    prob    n_batches   batch_pct
    "100   0.10   5  0.001"
    "500   0.05   5  0.001"
    "1000  0.02   5  0.001"
    "3000  0.01   5  0.001"
    "10000 0.005  5  0.001"
)
# ============================================================================

# Global parameters
SEED=42
NX_SKIP_ABOVE=20000 # skip NetworkX above this size (too slow)

# Create output and graph directories
mkdir -p "$OUT_DIR/cuda_static" "$OUT_DIR/normal" "$OUT_DIR/node_based" "$OUT_DIR/networkx" "$GRAPHS_DIR"

if [ ! -f "$GEN_SCRIPT" ]; then
    echo "ERROR: graphgen.py not found at $GEN_SCRIPT"
    exit 1
fi

if [ ! -f "$NX_SCRIPT" ]; then
    echo "ERROR: nx_louvain.py not found at $NX_SCRIPT"
    exit 1
fi

for cfg in "${GRAPH_CONFIGS[@]}"; do
    # Parse this row's parameters.
    read -r n PROB N_BATCHES BATCH_PCT <<< "$cfg"

    name="er_n${n}_p${PROB}_b${N_BATCHES}_bp${BATCH_PCT}"
    raw="$GRAPHS_DIR/${name}_raw.txt"
    input="$GRAPHS_DIR/${name}.txt"

    # Compute expected edge count for ER(n, p) and derive batch size.
    EXPECTED_EDGES=$(python -c "print(int($n * ($n - 1) // 2 * $PROB))")
    BATCH_SIZE=$(python -c "print(max(1, int($EXPECTED_EDGES * $BATCH_PCT)))")

    echo ""
    echo "=========================================="
    echo "  $name"
    echo "  nodes=$n  prob=$PROB  seed=$SEED"
    echo "  expected_edges=$EXPECTED_EDGES  batches=$N_BATCHES  per-batch del=ins=$BATCH_SIZE"
    echo "=========================================="

    # Generate a dynamic graph (initial ER(n, p) + N_BATCHES batches with
    # BATCH_SIZE deletions and BATCH_SIZE insertions each). graphgen writes BOTH
    # the program-compatible unweighted input ("$input", consumed by the CUDA
    # binaries, which read 'u v' and treat every weight as 1) and the raw
    # weighted view ("$raw") for reference.
    python "$GEN_SCRIPT" --nodes "$n" --prob "$PROB" \
        --batches "$N_BATCHES" \
        --deletions "$BATCH_SIZE" --insertions "$BATCH_SIZE" \
        --seed "$SEED" -o "$input" --raw-output "$raw" > /dev/null
    if [ ! -f "$input" ]; then
        echo "  ERROR: graph generation failed"
        continue
    fi

    initial_edges=$(awk 'NR==1 { print $2 }' "$input")
    echo "  Generated: $initial_edges initial undirected edges"

    # # --- Static CUDA (only runs on the initial graph; ignores batches) ---
    # echo "  [1/4] Static..."
    # "$STATIC" "$OUT_DIR/cuda_static/${name}_communities.txt" \
    #     < "$input" \
    #     > "$OUT_DIR/cuda_static/${name}.txt" 2>&1
    # if [ $? -eq 0 ]; then
    #     echo "  [1/4] Static: OK"
    # else
    #     echo "  [1/4] Static: FAILED (see $OUT_DIR/cuda_static/${name}.txt)"
    # fi

    # # --- Dynamic (edge-based) — runs initial + all N_BATCHES batches ---
    # echo "  [2/4] Dynamic (edge)..."
    # "$DYNAMIC" "$OUT_DIR/normal/${name}_communities.txt" \
    #     < "$input" \
    #     > "$OUT_DIR/normal/${name}.txt" 2>&1
    # if [ $? -eq 0 ]; then
    #     echo "  [2/4] Dynamic (edge): OK"
    # else
    #     echo "  [2/4] Dynamic (edge): FAILED (see $OUT_DIR/normal/${name}.txt)"
    # fi

    # # --- Dynamic (node-based) — same as above with the node-based kernel ---
    # echo "  [3/4] Dynamic (node)..."
    # "$NODEBASED" "$OUT_DIR/node_based/${name}_communities.txt" \
    #     < "$input" \
    #     > "$OUT_DIR/node_based/${name}.txt" 2>&1
    # if [ $? -eq 0 ]; then
    #     echo "  [3/4] Dynamic (node): OK"
    # else
    #     echo "  [3/4] Dynamic (node): FAILED (see $OUT_DIR/node_based/${name}.txt)"
    # fi

    # --- NetworkX (initial graph + all batches; re-runs Louvain per batch) ---
    # nx_louvain.py now parses the full dynamic input (initial graph + batches)
    # and re-runs Louvain from scratch after each batch update, so we feed it the
    # same "$input" file as the CUDA binaries (no truncated *_nx.txt needed).
    if [ "$n" -gt "$NX_SKIP_ABOVE" ]; then
        echo "  [4/4] NetworkX: SKIP (n_nodes > $NX_SKIP_ABOVE)"
    else
        echo "  [4/4] NetworkX..."
        python "$NX_SCRIPT" "$input" --seed "$SEED" \
            > "$OUT_DIR/networkx/${name}.txt" 2>&1
        if [ $? -eq 0 ]; then
            echo "  [4/4] NetworkX: OK"
        else
            echo "  [4/4] NetworkX: FAILED (see $OUT_DIR/networkx/${name}.txt)"
        fi
    fi
done

echo ""
echo "=========================================="
echo "  All benchmarks complete."
echo "  Generated graphs in: $GRAPHS_DIR/"
echo "  Outputs in: $OUT_DIR/{cuda_static,normal,node_based,networkx}/"
echo "=========================================="
