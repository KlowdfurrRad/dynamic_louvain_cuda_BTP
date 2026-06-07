#!/bin/bash
# Convert SNAP temporal graphs to CUDA dynamic Louvain input and benchmark all
# four implementations (static CUDA, dynamic CUDA edge-based, dynamic CUDA
# node-based, NetworkX) on them.
#
# Mirror of real_graphs/run_benchmarks.sh and generate/run_benchmarks.sh,
# adapted for temporal graph input.
#
# Usage:
#   cd louvain/real_graphs/snap_temporal && bash run_benchmarks.sh
#   Or: bash real_graphs/snap_temporal/run_benchmarks.sh  (from louvain/)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUVAIN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ALG_DIR="$LOUVAIN_DIR/algorithm"

CONVERT_SCRIPT="$SCRIPT_DIR/convert_snap_temporal_to_dynamic.py"
NX_SCRIPT="$ALG_DIR/nx_louvain.py"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
CONVERTED_DIR="$SCRIPT_DIR/converted"
OUT_DIR="$SCRIPT_DIR/outputs"

STATIC="$ALG_DIR/static_louvain.exe"
DYNAMIC="$ALG_DIR/dynamic_louvain.exe"
NODEBASED="$ALG_DIR/dynamic_louvain_nodebased.exe"

# ============================================================================
# BENCHMARK CONFIGURATION — one row per temporal graph
# ============================================================================
# Each row: "<graph_name>  <raw_file>  <initial_pct>  <n_batches>"
#
#   graph_name   — short name used for output file naming
#   raw_file     — path under graphs/ (the SNAP .txt file)
#   initial_pct  — fraction of earliest edges used as the initial graph
#   n_batches    — number of dynamic batches for remaining edges
# ============================================================================
GRAPH_CONFIGS=(
    #  name            raw_file                 initial_pct   n_batches
     "   CollegeMsg     CollegeMsg.txt           0.8           5"
     "   sx-mathover    sx-mathoverflow.txt      0.8           5"
     "   sx-askubuntu   sx-askubuntu.txt         0.8           5"
     "   sx-superuser   sx-superuser.txt         0.8           5"
)
# ============================================================================

SEED=42
NX_SKIP_ABOVE=2000000  # skip NetworkX above this many initial edges (too slow)

mkdir -p "$OUT_DIR/cuda_static" "$OUT_DIR/normal" "$OUT_DIR/node_based" "$OUT_DIR/networkx" "$CONVERTED_DIR"

if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "ERROR: converter not found at $CONVERT_SCRIPT"
    exit 1
fi

if [ ! -f "$NX_SCRIPT" ]; then
    echo "ERROR: nx_louvain.py not found at $NX_SCRIPT"
    exit 1
fi

for cfg in "${GRAPH_CONFIGS[@]}"; do
    read -r name raw_file initial_pct n_batches <<< "$cfg"

    raw="$GRAPHS_DIR/$raw_file"
    input="$CONVERTED_DIR/${name}.txt"
    nx_input="$CONVERTED_DIR/${name}.txt"

    echo ""
    echo "=========================================="
    echo "  $name"
    echo "  raw=$raw"
    echo "  initial_pct=$initial_pct  n_batches=$n_batches"
    echo "=========================================="

    if [ ! -f "$raw" ]; then
        echo "  SKIP: $raw not found (download it first into graphs/)"
        continue
    fi

    # Convert the temporal edge-stream into the dynamic CUDA input format.
    # This writes two files:
    #   $input    — full format (header + initial edges + n_batches + batch data)
    #   $nx_input — header + initial edges only, for the NetworkX baseline
    python "$CONVERT_SCRIPT" "$raw" "$input" \
        --initial-pct "$initial_pct" \
        --n-batches "$n_batches"
    if [ ! -f "$input" ] || [ ! -f "$nx_input" ]; then
        echo "  ERROR: conversion failed"
        continue
    fi

    initial_edges=$(awk 'NR==1 { print $2; exit }' "$input")
    echo "  Initial undirected edges: $initial_edges"

    # # --- Static CUDA (processes the initial graph only; ignores batches) ---
    # echo "  [1/4] Static..."
    # "$STATIC" "$OUT_DIR/cuda_static/${name}_communities.txt" \
    #     < "$input" \
    #     > "$OUT_DIR/cuda_static/${name}.txt" 2>&1
    # if [ $? -eq 0 ]; then
    #     echo "  [1/4] Static: OK"
    # else
    #     echo "  [1/4] Static: FAILED (see $OUT_DIR/cuda_static/${name}.txt)"
    # fi

    # # --- Dynamic (edge-based) — initial + all batches ---
    # echo "  [2/4] Dynamic (edge)..."
    # "$DYNAMIC" "$OUT_DIR/normal/${name}_communities.txt" \
    #     < "$input" \
    #     > "$OUT_DIR/normal/${name}.txt" 2>&1
    # if [ $? -eq 0 ]; then
    #     echo "  [2/4] Dynamic (edge): OK"
    # else
    #     echo "  [2/4] Dynamic (edge): FAILED (see $OUT_DIR/normal/${name}.txt)"
    # fi

    # # --- Dynamic (node-based) — same as edge-based but with node-parallel kernel ---
    # echo "  [3/4] Dynamic (node)..."
    # "$NODEBASED" "$OUT_DIR/node_based/${name}_communities.txt" \
    #     < "$input" \
    #     > "$OUT_DIR/node_based/${name}.txt" 2>&1
    # if [ $? -eq 0 ]; then
    #     echo "  [3/4] Dynamic (node): OK"
    # else
    #     echo "  [3/4] Dynamic (node): FAILED (see $OUT_DIR/node_based/${name}.txt)"
    # fi

    # --- NetworkX (initial graph only; NetworkX has no dynamic mode) ---
    if [ "$initial_edges" -gt "$NX_SKIP_ABOVE" ]; then
        echo "  [4/4] NetworkX: SKIP (initial edges > $NX_SKIP_ABOVE)"
    else
        echo "  [4/4] NetworkX..."
        python "$NX_SCRIPT" "$nx_input" --seed "$SEED" \
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
echo "  All temporal benchmarks complete."
echo "  Raw graphs in:       $GRAPHS_DIR/"
echo "  Converted inputs in: $CONVERTED_DIR/"
echo "  Outputs in:          $OUT_DIR/{cuda_static,normal,node_based,networkx}/"
echo "=========================================="
