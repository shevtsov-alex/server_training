#!/usr/bin/env python3
"""
Script to extract layers from PSD and PSB files in src folder as PNG images with RGBA.
"""
import argparse
import subprocess
import hashlib, os, sys
from pathlib import Path
from psd_tools import PSDImage
from PIL import Image


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


def extract_layers_from_psd(psd_path: Path, output_dir: Path, total_layers: int) -> int:
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

    # Remove PNG files related to the current PSD file in the output folder
    for file in output_dir.glob(f"{base_name}_*.png"):
        try:
            file.unlink()
        except Exception as e:
            print(f"Warning: Failed to remove file {file}: {e}")

    # Save the composite (total/flattened) image
    """
    try:
        composite_image = psd.topil()
        if composite_image is not None:
            if composite_image.mode != 'RGBA':
                composite_image = composite_image.convert('RGBA')
            composite_path = output_dir / f"{base_name}.png"
            composite_image.save(composite_path, 'PNG')
            print(f"Saved composite image: {composite_path.name}")
    except Exception as e:
        print(f"Warning: Failed to extract composite image: {e}")
    """

    all_layers = []
    collect_all_layers(psd, all_layers)
    all_layers = all_layers[::-1]
    all_layers = [layer for layer in all_layers if layer.visible and layer.topil() is not None ]
    print(f"Psd: {psd_path.name} - Total layers: {len(all_layers)}")
    
    layer_hashes = {}
    duplicate_groups = {}
    layer_file_index = 1
    
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
                output_path = output_dir / f"{OUTPUT_FILE_COUNT}.png"
                with open(output_dir / f"{OUTPUT_FILE_COUNT}.txt", 'w+') as f:
                    level = int(base_name[-1])
                    if level not in range(0, len(Prompts_by_level)):
                        raise ValueError(f"Level {level} is not valid in file {base_name}")
                    f.write(f"{Prompts_by_level[level]}\n{Base_prompt}")
            layer_hashes[image_hash] = (layer_index, layer_name)
            if layer_file_index == len(all_layers):
                output_path = output_dir / f"{OUTPUT_FILE_COUNT}.png"
            else:
                output_path = output_dir / f"{OUTPUT_FILE_COUNT}_{layer_file_index}.png"
            canvas_image.save(output_path, 'PNG')
            layer_file_index += 1
            #layers_extracted.append(canvas_image)

    """
    extracted_count = layer_file_index - 1
    if extracted_count < total_layers:
        empty_canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        for _ in range(total_layers - extracted_count):
            output_path = output_dir / f"{base_name}_{layer_file_index}.png"
            empty_canvas.save(output_path, 'PNG')
            layer_file_index += 1
        print(
            f"Added {total_layers - extracted_count} empty layer(s) "
            f"for {base_name} to reach {total_layers}"
        )
    """


def main(src_dir: str, output_dir: str, total_layers: int):
    """Main function to process all PSD and PSB files from src folder."""
    src_dir = Path(src_dir)
    output_dir = Path(output_dir)
    
    psd_files = sorted(list(src_dir.glob("*.psd")) + list(src_dir.glob("*.psb")))
    
    if not psd_files:
        print(f"No PSD or PSB files found in {src_dir}")
        return
    
    print(f"Found {len(psd_files)} PSD/PSB file(s)\n")
    
    for psd_file in psd_files:
        print(f"Processing: {psd_file.name}")
        try:
            count = extract_layers_from_psd(psd_file, output_dir, total_layers)
            print(f"Extracted {count} layer(s) from {psd_file.name}\n")
        except Exception as e:
            print(f"Error processing {psd_file.name}: {e}\n")
    
    print(f"All layers saved to: {output_dir}")

parser = argparse.ArgumentParser(description="Extract layers from PSD and PSB files")
parser.add_argument("--total_layers", type=int, default=1, help="Total layers to extract")
parser.add_argument("--src_dir", type=str, default="./src", help="Source directory")
parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.src_dir) or not os.path.isdir(args.src_dir):
        raise FileNotFoundError(f"Source directory does not exist or is not a directory: {args.src_dir}")
    if not os.path.exists(args.output_dir) or not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Output directory does not exist or is not a directory: {args.output_dir}")
    if args.total_layers < 1:
        raise ValueError(f"Total layers must be at least 1: {args.total_layers}")
    main(args.src_dir, args.output_dir, args.total_layers)
