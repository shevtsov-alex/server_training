#!/usr/bin/env python3
"""Reindex numbered file groups to a continuous 1-based sequence.

Each group consists of:
  - N.png        (main image)
  - N.txt        (caption)
  - N_1.png, N_2.png, ...  (sub-layers)

Gaps in the numbering are collapsed so the result is 1, 2, 3, ...
mapped_levels.json is rewritten to match.
"""
import argparse
import json
import re
import sys
from pathlib import Path

GROUP_FILE_RE = re.compile(r"^(\d+)(?:_(\d+))?\.(png|txt)$")


def discover_groups(directory: Path) -> dict[int, list[Path]]:
    """Return {group_index: [paths]} for every numbered file group."""
    groups: dict[int, list[Path]] = {}
    for path in directory.iterdir():
        m = GROUP_FILE_RE.match(path.name)
        if not m:
            continue
        idx = int(m.group(1))
        groups.setdefault(idx, []).append(path)
    return groups


def reindex(directory: Path, *, dry_run: bool = False) -> None:
    groups = discover_groups(directory)
    if not groups:
        print("No numbered file groups found.")
        return

    print(f"Found {len(groups)} groups")

    print(f"Groups: {groups[1]}")


    old_indices = sorted(groups)
    mapping: dict[int, int] = {}
    for new_idx, old_idx in enumerate(old_indices, start=1):
        mapping[old_idx] = new_idx

    already_sequential = all(o == n for o, n in mapping.items())
    if already_sequential:
        print("Files are already sequentially numbered -- nothing to do.")
        return

    gaps = [old for old, new in mapping.items() if old != new]
    print(f"Found {len(old_indices)} groups, first gap at index {gaps[0]}.")
    print(f"Renaming {len(gaps)} groups to close gaps.\n")

    # Phase 1: rename to temporary names to avoid collisions
    tmp_suffix = "__tmp_reindex"
    for old_idx in old_indices:
        new_idx = mapping[old_idx]
        if old_idx == new_idx:
            continue
        for path in groups[old_idx]:
            m = GROUP_FILE_RE.match(path.name)
            sub = f"_{m.group(2)}" if m.group(2) else ""
            ext = m.group(3)
            tmp_name = f"{new_idx}{sub}.{ext}{tmp_suffix}"
            tmp_path = path.with_name(tmp_name)
            print(f"  {path.name}  ->  {tmp_name}")
            if not dry_run:
                path.rename(tmp_path)

    # Phase 2: strip temporary suffix
    for path in directory.iterdir():
        if path.name.endswith(tmp_suffix):
            final_path = path.with_name(path.name.removesuffix(tmp_suffix))
            if not dry_run:
                path.rename(final_path)

    # Phase 3: update mapped_levels.json
    levels_path = directory / "mapped_levels.json"
    if levels_path.exists():
        with open(levels_path) as f:
            old_levels: dict[str, int] = json.load(f)

        new_levels: dict[str, int] = {}
        for old_key, level in old_levels.items():
            m = re.match(r"^(\d+)\.png$", old_key)
            if not m:
                new_levels[old_key] = level
                continue
            old_idx = int(m.group(1))
            new_idx = mapping.get(old_idx)
            if new_idx is None:
                continue
            new_levels[f"{new_idx}.png"] = level

        if not dry_run:
            with open(levels_path, "w") as f:
                json.dump(new_levels, f)
        print(f"\nUpdated mapped_levels.json ({len(new_levels)} entries).")
    else:
        print("\nNo mapped_levels.json found -- skipped.")

    print("Done." if not dry_run else "Dry run complete -- no files were changed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reindex file groups to a continuous sequence.")
    parser.add_argument("--input_dir", dest="input_dir",type=str, help="Directory containing numbered files")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without changing anything")
    args = parser.parse_args()

    target = Path(args.input_dir)
    if not target.is_dir():
        print(f"Error: {target} is not a directory", file=sys.stderr)
        sys.exit(1)

    reindex(target, dry_run=args.dry_run)
