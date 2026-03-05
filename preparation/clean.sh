#!/usr/bin/env bash
set -euo pipefail

dir="${1:?Usage: $0 <directory>}"

cd "$dir"

count=0
for f in *.png.txt; do
    [ -f "$f" ] || continue
    mv "$f" "${f%.png.txt}.txt"
    ((count++))
done

echo "Renamed $count files in $dir"
#./preparation/clean.sh out/doors_aligned_v1