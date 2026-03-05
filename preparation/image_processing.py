from __future__ import annotations

import argparse, os 
from pathlib import Path
from typing import List, NamedTuple, Tuple
from tqdm import tqdm
from PIL import Image
import numpy as np

ALPHA_THRESHOLD = 4


class ContentBox(NamedTuple):
    """Bounding-box coordinates for the visible content region."""
    row_min: int
    row_max: int
    col_min: int
    col_max: int


def clean_alpha_noise(img: Image.Image) -> Image.Image:
    """Zero out pixels whose alpha is at or below the noise threshold."""
    rgba = np.array(img.convert("RGBA"))
    rgba[rgba[:, :, 3] <= ALPHA_THRESHOLD] = 0
    return Image.fromarray(rgba, mode="RGBA")


def compute_content_box(img: Image.Image, margin: int = 20) -> ContentBox:
    """Find the bounding box of visible pixels in an RGBA image.

    Pixels with alpha <= ALPHA_THRESHOLD are treated as transparent.
    A ``margin`` of extra pixels is kept on each side (clamped to canvas).

    Raises ``ValueError`` if the image is fully transparent.
    """
    rgba = np.array(clean_alpha_noise(img).convert("RGBA"))
    h, w = rgba.shape[:2]
    visible = rgba[:, :, 3] > ALPHA_THRESHOLD

    rows_with_content = np.any(visible, axis=1)
    cols_with_content = np.any(visible, axis=0)

    if not rows_with_content.any():
        raise ValueError("Image is fully transparent -- nothing to crop")

    row_min = int(np.argmax(rows_with_content))
    row_max = int(len(rows_with_content) - 1 - np.argmax(rows_with_content[::-1]))
    col_min = int(np.argmax(cols_with_content))
    col_max = int(len(cols_with_content) - 1 - np.argmax(cols_with_content[::-1]))

    return ContentBox(
        row_min=max(row_min - margin, 0),
        row_max=min(row_max + margin, h - 1),
        col_min=max(col_min - margin, 0),
        col_max=min(col_max + margin, w - 1),
    )


def crop_to_box(img: Image.Image, box: ContentBox) -> Image.Image:
    """Crop an RGBA image to the region described by ``box``, cleaning alpha noise."""
    rgba = np.array(clean_alpha_noise(img).convert("RGBA"))
    rgba[rgba[:, :, 3] <= ALPHA_THRESHOLD] = 0
    return Image.fromarray(
        rgba[box.row_min:box.row_max + 1, box.col_min:box.col_max + 1],
        mode="RGBA",
    )


def crop_to_content(img: Image.Image, margin: int = 20) -> tuple[Image.Image, ContentBox]:
    """Crop an RGBA image to the bounding box of its visible pixels plus a margin.

    Convenience wrapper: computes the content box and applies it in one call.
    Returns ``(cropped_image, content_box)``.
    """
    box = compute_content_box(img, margin)
    return crop_to_box(img, box), box


def clear_noise(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    rgba = np.array(img)
    rgba[rgba[:, :, 3] == 0, :3] = 0  # zero RGB where alpha is 0
    img = Image.fromarray(rgba)
    return img


def find_invisible_nonempty_rgb(img: Image.Image) -> None:
    """
    Returns a list of ((x, y), (r, g, b, a)) for pixels where:
      a == 0 AND (r, g, b) != (0, 0, 0)
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    w, h = img.size
    px = img.load()
    total_pixels = w * h
    noise_pixels = 0

    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a == 0 and (r != 0 or g != 0 or b != 0):
                #print(f" X:{x}, Y:{y} ({r}, {g}, {b}, {a}")
                noise_pixels += 1

    #print(f"Total pixels: {total_pixels}, Noise pixels: {noise_pixels}, Noise percentage: {noise_pixels / total_pixels * 100:.2f}%")
    return noise_pixels / total_pixels * 100

def compute_noise_percentage(input_dir: str) -> float:
    perc = []
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".png"):
            img = Image.open(f"{input_dir}/{filename}")
            perc.append(find_invisible_nonempty_rgb(img))
    
    perc.sort()
    print(f' low: {perc[0]}, high: {perc[-1]} avg: {sum(perc) / len(perc)} median: {perc[len(perc) // 2]}')

def main(input_dir: str, output_dir: str):
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".png"):
            img = Image.open(f"{input_dir}/{filename}")
            img = clear_noise(img)
            img.save(f"{output_dir}/{filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, dest="input_dir", required=True)
    parser.add_argument("--output_dir", type=str, dest="output_dir", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_dir) or not os.path.exists(args.output_dir):
        raise FileNotFoundError(f"Output/Input directory does not exist or is not a directory: {args.output_dir} or {args.input_dir}")
    if args.input_dir.endswith('/'):
        args.input_dir = args.input_dir[:-1]
    if args.output_dir.endswith('/'):
        args.output_dir = args.output_dir[:-1]
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print('Do you want to proceed? (y/n)')
    answer = input()
    if answer.lower().strip() != 'y':
        import sys
        sys.exit(-1)

    #compute_noise_percentage(args.input_dir)
    main(args.input_dir, args.output_dir)