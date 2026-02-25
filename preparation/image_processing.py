from __future__ import annotations

import argparse, os 
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image
import numpy as np

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

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, dest="input_dir", required=True)
parser.add_argument("--output_dir", type=str, dest="output_dir", required=True)
args = parser.parse_args()

if __name__ == "__main__":
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