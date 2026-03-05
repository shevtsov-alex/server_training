#!/usr/bin/env python3
"""
Script to extract layers from PSD and PSB files in src folder as PNG images with RGBA.
"""
import argparse
import subprocess
import hashlib, os, sys
from pathlib import Path
from psd_tools import PSDImage
from PIL import Image, ImageDraw
from image_processing import compute_content_box, crop_to_box
from anotation import process_file
import json
from collections import defaultdict


Base_prompt = """
Identify and isolate every character in the scene as an individual layer. Ensure all character edges are sharp and clean.
Keep internal shadows that define the character's body and clothes. Exclude shadows cast by the character onto the ground or walls; these must stay in the separate layer.

Layer Hierarchy: Characters (Each person isolated with their handheld objects), Foreground (Any object blocking or overlapping a character must be its own separate layer) and Background (The remaining environment including all floor and wall shadows). 
"""


Prompts_by_level = ["L1_ENTITY ISOLATE CLEAN_EDGE character person environment split",
                    "L2_LIMB ISOLATE CLEAN_EDGE anatomy head torso arms legs",
                    "L3_PARTS ISOLATE CLEAN_EDGE body parts",
                    "L4_DETAILED ISOLATE CLEAN_EDGE detailed micro parts"]

OUTPUT_FILE_COUNT = 0

ARROW_WIDTH = 400
ARROW_PAD = 60
ARROW_HEAD_HALF_H = 120
ARROW_SHAFT_HALF_H = 60
ARROW_MARGIN = 80
STRIP_HEIGHT = 200

def collect_all_layers(layer_container, layer_list: list):
    """Recursively collect all extractable layers in z-order (bottom to top)."""
    if not hasattr(layer_container, '_layers'):
        return
    
    for layer in reversed(layer_container._layers):
        if hasattr(layer, 'topil') and layer.visible and layer.parent.visible:
            layer_image = layer.topil()
            if layer_image is not None:
                layer_list.append(layer)
        
        if hasattr(layer, '_layers'):
            collect_all_layers(layer, layer_list)


def get_image_hash(image: Image.Image) -> str:
    """Calculate hash of an image for duplicate detection."""
    return hashlib.md5(image.tobytes()).hexdigest()


def extract_layers_from_psd(psd_path: Path, output_dir: Path, total_layers: int, strip_dir: bool) -> int:
    global OUTPUT_FILE_COUNT
    """
    Extract all layers from a PSD or PSB file as PNG images with RGBA.
    
    Args:
        psd_path: Path to the PSD or PSB file
        output_dir: Directory to save PNG files
        
    Returns:
        Number of layers extracted
        
    Raises:
        FileNotFoundError: If PSD/PSB file doesn't exist
        ValueError: If file cannot be read as PSD/PSB
    """
    
    if not psd_path.exists():
        raise FileNotFoundError(f"PSD/PSB file does not exist: {psd_path}")
    
    try:
        psd = PSDImage.open(psd_path)
    except Exception as e:
        raise ValueError(f"Failed to open PSD/PSB file {psd_path}: {e}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    canvas_width = psd.width
    canvas_height = psd.height
    base_name = psd_path.stem
    layers_extracted = []
    has_stored_layers = False

    psd_level = int(base_name[-1])
    filename = ""

    # Remove PNG files related to the current PSD file in the output folder
    for file in output_dir.glob(f"{base_name}_*.png"):
        try:
            file.unlink()
        except Exception as e:
            print(f"Warning: Failed to remove file {file}: {e}")

    all_layers = []
    collect_all_layers(psd, all_layers)
    all_layers = all_layers[::-1]
    all_layers = [layer for layer in all_layers if layer.visible and layer.topil() is not None ]
    print(f"Psd: {psd_path.name} - Total layers: {len(all_layers)}")
    
    layer_hashes = {}
    duplicate_groups = {}
    layer_file_index = 1
    unique_layers = []
    
    for layer_index, layer in enumerate(all_layers):
        layer_image = layer.topil()
        #if layer_image is None or layer.visible == False:
        #    continue
        
        if layer_image.mode != 'RGBA':
            layer_image = layer_image.convert('RGBA')
        
        canvas_image = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        
        layer_left = layer.left if hasattr(layer, 'left') else 0
        layer_top = layer.top if hasattr(layer, 'top') else 0
        
        canvas_image.paste(layer_image, (layer_left, layer_top), layer_image)
        
        image_hash = get_image_hash(canvas_image)
        layer_name = layer.name if hasattr(layer, 'name') and layer.name else f"Layer_{layer_index}"
        
        if image_hash in layer_hashes:
            if image_hash not in duplicate_groups:
                first_index, first_name = layer_hashes[image_hash]
                duplicate_groups[image_hash] = [(first_index, first_name)]
            duplicate_groups[image_hash].append((layer_index, layer_name))
        else:
            if has_stored_layers == False:
                OUTPUT_FILE_COUNT += 1
                has_stored_layers = True
                
            layer_hashes[image_hash] = (layer_index, layer_name)
            unique_layers.append(canvas_image)
    if unique_layers:
        content_box = compute_content_box(unique_layers[-1])
        cropped_layers = [crop_to_box(img, content_box) for img in unique_layers]

        if not strip_dir:

            for i, cropped in enumerate(cropped_layers):
                if i == len(cropped_layers) - 1:
                    output_path = output_dir / f"{OUTPUT_FILE_COUNT}.png"
                    filename = f"{OUTPUT_FILE_COUNT}.png"
                else:
                    output_path = output_dir / f"{OUTPUT_FILE_COUNT}_{i + 1}.png"
                cropped.save(output_path, 'PNG')

            composite_path = output_dir / f"{OUTPUT_FILE_COUNT}.png"
            """
            level = int(base_name[-1])
            keep, caption = process_file(level, composite_path)

            if keep:
                with open(output_dir / f"{OUTPUT_FILE_COUNT}.txt", 'w+') as f:
                    f.write(caption)
            """

        composite = cropped_layers[-1]
        sub_layers = cropped_layers[:-1]
        if sub_layers and strip_dir:
            crop_w, crop_h = composite.size
            scale = STRIP_HEIGHT / crop_h
            scaled_w = int(crop_w * scale)

            sub_layers = [img.resize((scaled_w, STRIP_HEIGHT), Image.LANCZOS) for img in sub_layers]

            separator_w = 4
            total_w = scaled_w * len(sub_layers) + separator_w * (len(sub_layers) - 1)
            strip = Image.new('RGBA', (total_w, STRIP_HEIGHT), (0, 0, 0, 0))

            for i, layer_img in enumerate(sub_layers):
                x = i * (scaled_w + separator_w)
                strip.paste(layer_img, (x, 0))
                if i < len(sub_layers) - 1:
                    strip.paste(Image.new('RGBA', (separator_w, STRIP_HEIGHT), (0, 0, 0, 255)), (x + scaled_w, 0))

            strip.save(output_dir / f"{base_name}_strip.png", 'PNG')

            composite_scaled = composite.resize((scaled_w, STRIP_HEIGHT), Image.LANCZOS)
            composite_scaled.save(output_dir / f"{base_name}_strip_1.png", 'PNG')

    return filename, psd_level


def main(src_dir: str, output_dir: str, total_layers: int, strip_dir: bool):
    """Main function to process all PSD and PSB files from src folder."""
    src_dir = Path(src_dir)
    output_dir = Path(output_dir)
    archive = defaultdict(lambda : -1)
    
    psd_files = sorted(list(src_dir.glob("*.psd")) + list(src_dir.glob("*.psb")))
    
    if not psd_files:
        print(f"No PSD or PSB files found in {src_dir}")
        return
    
    print(f"Found {len(psd_files)} PSD/PSB file(s)\n")
    
    for psd_file in psd_files:
        print(f"Processing: {psd_file.name}")
        try:
            filename, psd_level = extract_layers_from_psd(psd_file, output_dir, total_layers, strip_dir)
            archive[filename] = psd_level
        except Exception as e:
            print(f"Error processing {psd_file.name}: {e}\n")
    
    print(f"All layers saved to: {output_dir}")
    with open(output_dir / "mapped_levels.json", 'w+') as f:
        json.dump(archive, f)

parser = argparse.ArgumentParser(description="Extract layers from PSD and PSB files")
parser.add_argument("--total_layers", type=int, default=1, help="Total layers to extract")
parser.add_argument("--src_dir", type=str, default="./src", help="Source directory")
parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
parser.add_argument("--strip", action="store_true", default=False, help="Create strip files")


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.src_dir) or not os.path.isdir(args.src_dir):
        raise FileNotFoundError(f"Source directory does not exist or is not a directory: {args.src_dir}")
    if not os.path.exists(args.output_dir) or not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Output directory does not exist or is not a directory: {args.output_dir}")
    if args.total_layers < 1:
        raise ValueError(f"Total layers must be at least 1: {args.total_layers}")
    main(args.src_dir, args.output_dir, args.total_layers, args.strip)
