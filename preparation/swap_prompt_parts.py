"""Swap the two parts around '>>' in [TAG] lines across all txt files."""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DELIMITER = " >> "


def swap_parts(line: str) -> str:
    """Swap text before and after '>>' within a [TAG] ... >> ... line."""
    if not line.startswith("[") or "]" not in line or DELIMITER not in line:
        return line

    tag_end = line.index("]") + 1
    prefix = line[:tag_end]
    body = line[tag_end:]

    parts = body.split(DELIMITER, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Expected exactly one '{DELIMITER}' delimiter in body: {body!r}")

    left = parts[0].strip()
    right = parts[1].strip()

    return f"{prefix} {right}{DELIMITER}{left}"


def process_file(path: Path, *, dry_run: bool) -> bool:
    """Process a single txt file. Returns True if content was modified."""
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)

    modified = False
    new_lines: list[str] = []
    for line in lines:
        new_line = swap_parts(line)
        if new_line != line:
            modified = True
        new_lines.append(new_line)

    if modified and not dry_run:
        path.write_text("".join(new_lines), encoding="utf-8")

    return modified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, help="Root directory with txt files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing",
    )
    args = parser.parse_args()

    root: Path = args.input_dir
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    txt_files = sorted(root.rglob("*.txt"))
    logger.info("Found %d txt files in %s", len(txt_files), root)

    changed = 0
    skipped = 0
    for path in txt_files:
        was_modified = process_file(path, dry_run=args.dry_run)
        if was_modified:
            changed += 1
            logger.info("%s %s", "Would modify" if args.dry_run else "Modified", path)
        else:
            skipped += 1

    logger.info(
        "Done. Changed: %d, Skipped (no [TAG] or no >>): %d",
        changed,
        skipped,
    )


if __name__ == "__main__":
    main()
