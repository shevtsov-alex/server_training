#!/usr/bin/env python3
#export HF_HOME="/workspace/layered/hf_cache/"
import os
import time
import torch
from PIL import Image
from model import QIL
from filtering import contains_character
from helper import scale_down_image, crop_to_content, restore_from_crop, create_psd
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

PROMPTS = ["L1_ENTITY ISOLATE CLEAN_EDGE character person environment split",
    "L2_LIMB ISOLATE CLEAN_EDGE anatomy head torso arms legs",
    "L3_PARTS ISOLATE CLEAN_EDGE body parts",
    "L4_DETAILED ISOLATE CLEAN_EDGE detailed micro parts"]
PROMPTS_BASE = """Identify and isolate every character in the scene as an individual layer. Ensure all character edges are sharp and clean.
Keep internal shadows that define the character's body and clothes. Exclude shadows cast by the character onto the ground or walls; these must stay in the separate layer.

Layer Hierarchy: Characters (Each person isolated with their handheld objects), Foreground (Any object blocking or overlapping a character must be its own separate layer) and Background (The remaining environment including all floor and wall shadows). 
"""

MODEL = QIL(device="cuda", dtype=torch.bfloat16)
SCALE_MAX_SIDE = 640


def process_single_level(original_image: Image.Image, images: list[Image.Image], level_num: int) -> tuple[list[Image.Image], list[Image.Image]]:
    prompt = PROMPTS[level_num - 1] +  "\n" + PROMPTS_BASE
    original_scaled = scale_down_image(original_image, SCALE_MAX_SIDE)
    ready_layers = []
    character_layers = []
    for image in tqdm(images):
        for layer in MODEL.inference(image, prompt):
            has_character = contains_character(original_scaled, scale_down_image(layer, SCALE_MAX_SIDE))
            if has_character is None:
                raise ValueError(f"Error: contains_character returned None for image")
            if has_character:
                character_layers.append(layer)
            else:
                ready_layers.append(layer)

    return ready_layers, character_layers


def main(input_image: str, level_num: int):
    print("Loading input image...")
    original_img = Image.open(input_image).convert("RGBA")
    print(f"Input image loaded: {original_img.size}")
    
    ready_layers = [original_img]
    total_start = time.time()

    if level_num == 0:
        level_todo_images = [original_img]
        for i in range(1, 4):
            print(f"Processing level {i} with total of {len(level_todo_images)} images...")
            level_start = time.time()
            level_ready_layers, level_character_layers = process_single_level(original_image = original_img, images = level_todo_images, level_num = i)
            level_elapsed = time.time() - level_start
            print(f"Level {i} done in {level_elapsed:.2f}s")
            ready_layers.extend(level_ready_layers)
            level_todo_images = level_character_layers
        
        ready_layers.extend(level_todo_images)
    else:
        level_start = time.time()
        level_ready_layers, level_character_layers = process_single_level(original_image = original_img, images = [original_img], level_num = level_num)
        level_elapsed = time.time() - level_start
        print(f"Level {level_num} done in {level_elapsed:.2f}s")
        ready_layers = level_ready_layers + level_character_layers

    total_elapsed = time.time() - total_start
    print(f"Ready layers: {len(ready_layers)}")
    print(f"Total processing time: {total_elapsed:.2f}s")
    
    create_psd(ready_layers, input_image.split(".png")[0] + "_layers.psd")

            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, dest="input_image", required=True)
    parser.add_argument("--level_num", type=str, dest="level_num", default=0, required=False)
    return parser.parse_args()

args = parse_args()
if __name__ == "__main__":
    if not os.path.exists(args.input_image):
        raise FileNotFoundError(f"Input image not found: {args.input_image}")
    if args.level_num == 0:
        print(f"Level number is not set, execution of full level pipeline will be performed")
    elif args.level_num < 0 or args.level_num > 3:
        raise ValueError(f"Level number must be between 0 and 3: {args.level_num}")

    main(input_image=args.input_image, level_num=args.level_num)