#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: rename_pngs.sh -s <source_dir> -d <dest_dir> -p <prefix>

Finds all .png files in <source_dir>, renames them to <prefix><index>.png,
and moves them into <dest_dir>.

Example:
  ./rename_pngs.sh -s /tmp -d /tmp/renamed -p prefix_
EOF
}

SRC=""
DST=""
PREFIX=""

while getopts ":s:d:p:h" opt; do
  case "$opt" in
    s) SRC="$OPTARG" ;;
    d) DST="$OPTARG" ;;
    p) PREFIX="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Error: -$OPTARG requires an argument"; usage; exit 2 ;;
    \?) echo "Error: unknown option -$OPTARG"; usage; exit 2 ;;
  esac
done

if [[ -z "${SRC}" || -z "${DST}" || -z "${PREFIX}" ]]; then
  echo "Error: -s, -d, and -p are all required"
  usage
  exit 2
fi

if [[ ! -d "$SRC" ]]; then
  echo "Error: source directory does not exist: $SRC"
  exit 2
fi

# Create destination directory if needed
mkdir -p -- "$DST"

# Collect PNG files (case-insensitive). If none, exit cleanly.
shopt -s nullglob nocaseglob
files=("$SRC"/*.png)

if (( ${#files[@]} == 0 )); then
  echo "No PNG files found in $SRC"
  exit 0
fi

i=1
for f in "${files[@]}"; do
  new="$DST/${PREFIX}${i}.png"

  # Avoid overwriting existing files
  if [[ -e "$new" ]]; then
    echo "Skip (target exists): $new"
    ((i++))
    continue
  fi

  echo "Renaming+moving: $f -> $new"
  mv -- "$f" "$new"
  ((i++))
done
