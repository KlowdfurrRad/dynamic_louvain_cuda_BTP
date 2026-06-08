#!/bin/bash
# Generate synthetic dynamic-batch inputs for com-DBLP at three batch sizes.
#
# Source : graphs/com-dblp_converted.txt  (the static converted com-DBLP, |E| = 1,049,866)
# Output : graphs/com-dblp-5-<pct>.txt     (initial graph + 5 batches)
#
# Each of the 5 batches changes <pct>% of |E|, split 60% insertions / 40% deletions.
# No graph snapshot contains a duplicate edge (handled by gen_synthetic_batches.py).
#
#   com-dblp-5-2.txt  : 5 batches, each  2% of |E|  (~12.6k ins + ~8.4k del per batch)
#   com-dblp-5-5.txt  : 5 batches, each  5% of |E|  (~31.5k ins + ~21.0k del per batch)
#   com-dblp-5-10.txt : 5 batches, each 10% of |E|  (~63.0k ins + ~42.0k del per batch)
#
# Usage:  cd louvain/real_graphs/DBLP_syn_batches && bash generate_batches.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
GEN="$SCRIPT_DIR/../gen_synthetic_batches.py"
SRC="$GRAPHS_DIR/com-dblp_converted.txt"

SEED=42
N_BATCHES=5
INS_RATIO=0.6          # 60% insertions, 40% deletions

PY="python"

if [ ! -f "$GEN" ]; then
    echo "ERROR: generator not found at $GEN"; exit 1
fi
if [ ! -f "$SRC" ]; then
    echo "ERROR: source graph not found at $SRC"; exit 1
fi

# "<pct>:<fraction-of-|E|>"
for cfg in "2:0.02" "5:0.05" "10:0.10" "20:0.2"; do
    pct="${cfg%%:*}"
    frac="${cfg##*:}"
    out="$GRAPHS_DIR/com-dblp-5-${pct}.txt"

    echo ""
    echo "=========================================="
    echo "  com-dblp-5-${pct}: 5 batches x ${pct}% of |E| (60/40 ins/del)"
    echo "=========================================="
    $PY "$GEN" "$SRC" "$out" \
        --n-batches "$N_BATCHES" \
        --batch-frac "$frac" \
        --ins-ratio "$INS_RATIO" \
        --seed "$SEED"
done
