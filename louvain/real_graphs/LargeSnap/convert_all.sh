#!/bin/bash
# Convert the raw SNAP edge lists in LargeSnap/graphs/ to the dynamic Louvain format
# (<name>_converted.txt). Accepts .txt or .txt.gz inputs -- if only the .gz is
# present, a kept copy is gunzipped first. Existing converted files are left as-is;
# delete one to force its regeneration.
#
# Uses the memory-safe streaming converter (convert_snap_streaming.py): these graphs
# are too large to de-duplicate in memory, and the SNAP com-* ungraph files are
# already simple undirected, so a two-pass streaming remap is enough.
#
# Usage:  cd louvain/real_graphs/LargeSnap && bash convert_all.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
CONVERT="$SCRIPT_DIR/../convert_snap_streaming.py"

# pick a python command (Linux/Colab: python3/python; Windows: py -3)
PY="python"

# "raw input file"   "output converted file"   (both inside graphs/)
GRAPHS=(
    "com-LiveJournal.ungraph.txt   com-LiveJournal_converted.txt"
    "com-orkut.ungraph.txt         com-Orkut_converted.txt"
)

if [ ! -f "$CONVERT" ]; then
    echo "ERROR: converter not found: $CONVERT"
    exit 1
fi

for entry in "${GRAPHS[@]}"; do
    read -r raw out <<< "$entry"
    rawpath="$GRAPHS_DIR/$raw"
    outpath="$GRAPHS_DIR/$out"

    echo ""
    echo "== $raw -> $out =="

    if [ -f "$outpath" ]; then
        echo "  already converted (delete $out to regenerate)"
        continue
    fi
    if [ ! -f "$rawpath" ] && [ -f "$rawpath.gz" ]; then
        echo "  gunzip $raw.gz"
        gunzip -k "$rawpath.gz"
    fi
    if [ ! -f "$rawpath" ]; then
        echo "  SKIP: neither $raw nor $raw.gz found"
        continue
    fi
    $PY "$CONVERT" "$rawpath" "$outpath"
done

echo ""
echo "Done. Converted files are in $GRAPHS_DIR/"
