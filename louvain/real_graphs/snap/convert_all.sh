#!/bin/bash
# Convert the raw SNAP edge lists in snap/graphs/ to the dynamic Louvain format
# (<name>_converted.txt). Accepts .txt or .txt.gz inputs -- if only the .gz is
# present, a kept copy is gunzipped first. Existing converted files are left as-is;
# delete one to force its regeneration.
#
# Uses convert_snap_to_dynamic.py, which de-duplicates and symmetrises edges (needed
# for the directed web-Google graph).
#
# Usage:  cd louvain/real_graphs/snap && bash convert_all.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAPHS_DIR="$SCRIPT_DIR/graphs"
CONVERT="$SCRIPT_DIR/../convert_snap_to_dynamic.py"

# pick a python command (Linux/Colab: python3/python; Windows: py -3)
if command -v python3 >/dev/null 2>&1; then PY="python3"
elif command -v python  >/dev/null 2>&1; then PY="python"
else PY="py -3"; fi

# "raw input file"   "output converted file"   (both inside graphs/)
GRAPHS=(
    "ca-GrQc.txt              ca-GrQc_converted.txt"
    "ca-HepTh.txt             ca-HepTh_converted.txt"
    "ca-HepPh.txt             ca-HepPh_converted.txt"
    "ca-AstroPh.txt           ca-AstroPh_converted.txt"
    "email-Enron.txt          email-Enron_converted.txt"
    "facebook_combined.txt    facebook_combined_converted.txt"
    "com-amazon.ungraph.txt   com-amazon_converted.txt"
    "com-dblp.ungraph.txt     com-dblp_converted.txt"
    "com-youtube.ungraph.txt  com-Youtube_converted.txt"
    "web-Google.txt           web-Google_converted.txt"
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
