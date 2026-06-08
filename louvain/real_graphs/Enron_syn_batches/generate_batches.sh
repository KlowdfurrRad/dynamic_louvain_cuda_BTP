#!/bin/bash
# Generate synthetic dynamic-batch inputs for email-Enron at three batch sizes.
#
# Source : graphs/email-Enron_converted.txt  (static converted email-Enron, |E| = 183,831)
# Output : graphs/email-Enron-5-<pct>.txt      (initial graph + 5 batches)
#
# Each of the 5 batches changes <pct>% of |E|, split 60% insertions / 40% deletions.
# No graph snapshot contains a duplicate edge (handled by gen_synthetic_batches.py).
#
#   email-Enron-5-2.txt  : 5 batches, each  2% of |E|  (~2.2k ins + ~1.5k del per batch)
#   email-Enron-5-5.txt  : 5 batches, each  5% of |E|  (~5.5k ins + ~3.7k del per batch)
#   email-Enron-5-10.txt : 5 batches, each 10% of |E|  (~11.0k ins + ~7.4k del per batch)
#
# Usage:  cd louvain/real_graphs/Enron_syn_batches && bash generate_batches.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
GEN="$SCRIPT_DIR/../gen_synthetic_batches.py"
SRC="$GRAPHS_DIR/email-Enron_converted.txt"

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
for cfg in "2:0.02" "5:0.05" "10:0.10"; do
    pct="${cfg%%:*}"
    frac="${cfg##*:}"
    out="$GRAPHS_DIR/email-Enron-5-${pct}.txt"

    echo ""
    echo "=========================================="
    echo "  email-Enron-5-${pct}: 5 batches x ${pct}% of |E| (60/40 ins/del)"
    echo "=========================================="
    $PY "$GEN" "$SRC" "$out" \
        --n-batches "$N_BATCHES" \
        --batch-frac "$frac" \
        --ins-ratio "$INS_RATIO" \
        --seed "$SEED"
done
