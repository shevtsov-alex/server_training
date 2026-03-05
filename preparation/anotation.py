"""
Dataset Captioner for Qwen-Image-Layered Training
==================================================
4-Level Hierarchical Decomposition Caption Generator
"""

import os
import sys
import json
import time
import shutil
import argparse
import re
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
if not GEMINI_MODEL:
    raise RuntimeError("GEMINI_MODEL is not set in .env file")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env file")

# ══════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════

TRIGGERS = [
    "[ISOLATE_SUBJECT]",
    "[DECOMPOSE_COMPONENTS]",
    "[DECOMPOSE_OBJECTS]",
    "[DECOMPOSE_DETAILS]",
]

LEVEL_DESCRIPTIONS = ["Background isolation — extract object from scene",
                      "Main component separation — primary structural parts",
                      "Sub-object separation — components into smaller pieces",
                      "Detail separation — small objects into granular elements"]
MIN_PIXEL_AREA = 300

# ══════════════════════════════════════════════════════════
#  GEMINI PROMPTS
# ══════════════════════════════════════════════════════════

GEMINI_SYSTEM = """You are a dataset annotation expert for hierarchical image segmentation.
Analyze image groups and return structured JSON annotations.
STRICT: respond with valid JSON only. No markdown, no backticks, no explanation.
Describe only what is clearly visible. Never invent parts.
Flag scattered particles, shadow-layers, and micro-elements as quality issues."""

PROMPTS_BY_LEVEL = ["""Analyze this LEVEL 0 pair: Image1=original scene, Image2=isolated object.
Return this JSON exactly:
{
  "object_type": "building|vehicle|character|prop|structure|vegetation|effect",
  "object_subtype": "specific e.g. medieval_gate or police_car",
  "object_description": "2-3 sentences: identity, primary material, dominant colors, viewing angle, notable surface features",
  "background_description": "1 sentence describing the background",
  "boundary_complexity": "simple|medium|complex",
  "boundary_notes": "describe edge challenges: cables, foliage, transparency, shadows",
  "isolation_quality": "clean|acceptable|problematic",
  "decomposition_instruction": "Output a single foreground layer containing only [describe subject] with full alpha transparency on the background. [boundary notes if any.]",
  "quality_flags": []
}
quality_flags values: blurry, too_dark, too_small_object, ambiguous_subject, background_bleeds_into_subject, partial_crop""",
"""Analyze this LEVEL 1 set: Image1=complete object, Images2+=each main component.
Main components are the PRIMARY STRUCTURAL PARTS, the largest fundamental divisions including object shadows and lighting effects.
Return this JSON exactly:
{
  "object_type": "category",
  "object_subtype": "specific type",
  "object_description": "2-3 sentences: full identity, materials, orientation, complexity",
  "components": [
    {"index": 0, "name": "name_with_underscores", "material": "material", "description": "1 sentence", "estimated_pixel_area": 0, "is_valid": true}
  ],
  "structural_logic": "1 sentence: rule that determines component divisions",
  "cross_material_variety": true,
  "decomposition_instruction": "Segment the following main components: [names]. Each is a distinct primary structural element. Output spatially non-overlapping layers.",
  "quality_flags": []
}
quality_flags: too_few_components, components_too_small, inconsistent_logic, scattered_elements_as_components, shadows_included_as_component, ambiguous_split""",
"""Analyze this LEVEL 2 set: Image1=one component from Level1, Images2+=smaller individual objects.
Sub-objects are mid-level distinct pieces that make up the component.
Return this JSON exactly:
{
  "parent_object_type": "what object this component came from",
  "component_name": "name of component being decomposed",
  "component_description": "1-2 sentences: component appearance",
  "sub_objects": [
    {"index": 0, "name": "name_with_underscores", "description": "1 sentence", "estimated_pixel_area": 0, "is_repeating_unit": false, "is_valid": true}
  ],
  "decomposition_pattern": "symmetric_split|repeating_units|distinct_sub_objects|layered_depth",
  "structural_logic": "1 sentence: why these are the natural subdivision units",
  "decomposition_instruction": "Segment the following distinct objects: [names]. [Pattern note.] Output non-overlapping.",
  "quality_flags": []
}
quality_flags: sub_objects_too_small, scattered_particles, shadows_as_objects, no_clear_boundary, ambiguous_logic, micro_details_not_sub_objects""",
"""Analyze this LEVEL 3 set: Image1=small individual object from Level2, Images2+=fine detail elements.
Details are the most granular visible features: surface decorations, functional micro-parts, texture regions.
Return this JSON exactly:
{
  "parent_component": "Level2 object name",
  "object_description": "1-2 sentences: precise description including material and finish",
  "details": [
    {"index": 0, "name": "name_with_underscores", "type": "structural|decorative|functional|surface_region", "description": "1 sentence", "estimated_pixel_area": 0, "is_valid": true}
  ],
  "detail_logic": "1 sentence: principle determining detail boundaries",
  "requires_closeup_crop": false,
  "decomposition_instruction": "Segment the following fine detail elements: [names]. Boundaries follow [structural/material/functional] divisions. Preserve sharp edges between adjacent details.",
  "worth_training": true,
  "quality_flags": []
}
Set worth_training=false if any area < 300px, if scattered particles, or if object too small.
quality_flags: elements_too_small, no_structural_logic, particle_noise, micro_shadows, object_too_small_for_level3, detail_count_too_low"""
]


# ══════════════════════════════════════════════════════════
#  CAPTION ASSEMBLY
# ══════════════════════════════════════════════════════════

def build_caption(level: int, analysis: dict) -> str:
    trigger = TRIGGERS[level]

    if level == 0:
        content = (
            analysis.get("object_description", "").strip() + " " +
            analysis.get("background_description", "").strip()
        )
        instruction = "Output a single foreground layer containing only the subject with full alpha transparency on the background."

    elif level == 1:
        comps = [
            c.get("name", "").replace("_", " ")
            for c in analysis.get("components", [])
            if c.get("is_valid", True)
        ]
        content = (
            analysis.get("object_description", "").strip() + " " +
            "Primary structural components: " + ", ".join(comps) + "."
        )
        instruction = f"Segment the following main components: " +\
            f"{', '.join(comps)} and their shadows/lighting effects. " +\
            "Each is a distinct primary structural element. Output spatially non-overlapping layers."

    elif level == 2:
        subs = [
            s.get("name", "").replace("_", " ")
            for s in analysis.get("sub_objects", [])
            if s.get("is_valid", True)
        ]
        pattern = analysis.get("decomposition_pattern", "").replace("_", " ")
        content = (
            analysis.get("component_description", "").strip() + " " +
            f"Component: {analysis.get('component_name', '')}. " +
            f"Decomposition pattern: {pattern}. " +
            f"Sub-objects: {', '.join(subs)}."
        )
        instruction =f"Segment the following distinct objects: " + \
            f"{', '.join(subs)} and their shadows/lighting effects. Output non-overlapping."

    elif level == 3:
        dets = [
            d.get("name", "").replace("_", " ")
            for d in analysis.get("details", [])
            if d.get("is_valid", True)
        ]
        content = (
            analysis.get("object_description", "").strip() + " " +
            f"Parent: {analysis.get('parent_component', '')}. " +
            f"Fine detail elements: {', '.join(dets)}."
        )
        instruction = f"Segment the following fine detail elements: " + \
            f"{', '.join(dets)} and their shadows/lighting effects. Preserve sharp edges between adjacent details."
        

    else:
        content = str(analysis)
        instruction = "Decompose as shown."

    content     = " ".join(content.split()).rstrip(".")
    instruction = " ".join(instruction.split())
    instruction = re.sub(r"(?i)\binto\s+\d+\s+", "the following ", instruction)

    return f"{trigger} {content} >> {instruction}"


# ══════════════════════════════════════════════════════════
#  QUALITY GATE
# ══════════════════════════════════════════════════════════

HARD_SKIP_FLAGS = {
    "too_small_object", "scattered_elements_as_components",
    "scattered_particles", "shadows_as_objects",
    "particle_noise", "micro_shadows",
    "no_structural_logic", "no_clear_boundary",
    "object_too_small_for_level3",
}

def should_skip(level: int, analysis: dict):
    flags = set(analysis.get("quality_flags", []))
    triggered = flags & HARD_SKIP_FLAGS
    #if triggered:
    #    return True, f"Hard quality flags: {triggered}"

    if level == 0:
        if analysis.get("isolation_quality") == "problematic":
            return True, "isolation_quality = problematic"

    if level == 3:
        #if not analysis.get("worth_training", True):
        #    return True, "worth_training = false"
        if analysis.get("requires_closeup_crop", False):
            return True, (
                "requires_closeup_crop=true -- create a cropped close-up of this object "
                "and re-run it as a separate level3 entry"
            )

    return False, ""


# ══════════════════════════════════════════════════════════
#  GEMINI API
# ══════════════════════════════════════════════════════════

def read_image_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def image_mime(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

def call_gemini(client: genai.Client, level: int, image_path: Path, retries: int = 3):
    image_bytes = read_image_bytes(image_path)
    mime = image_mime(image_path)
    contents = [
        {"inline_data": {"mime_type": mime, "data": image_bytes}},
        PROMPTS_BY_LEVEL[level],
    ]

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
            return json.loads(raw.strip())

        except json.JSONDecodeError as e:
            print(f"      [WARN] JSON parse error attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(2)

        except Exception as e:
            err = str(e)
            if "RATE_LIMIT" in err or "429" in err:
                wait = 10 * (attempt + 1)
                print(f"      [RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"      [WARN] API error attempt {attempt+1}: {e}")
                if attempt < retries - 1:
                    time.sleep(3)

    return None


def process_file(level: int, filename: str):
    client = genai.Client(api_key=GEMINI_API_KEY)

    analysis = call_gemini(client, level, Path(filename))

    if analysis is None:
        print("[ERROR]")
        return False, "Error calling Gemini API"

    #skip, reason = should_skip(level, analysis)
    #if skip:
    #    print(f"[SKIP] {reason}")
    #    return False, json.dumps(analysis, indent=2) + f"\n\n[SKIP] {reason}"

    caption = build_caption(level, analysis)
    return True, caption


def main(input_dir: str):
    input_dir = Path(input_dir)

    try:
        with open(input_dir / "mapped_levels.json", 'r') as f:
            mapped_levels = json.load(f)
    except Exception as e:
        print(f"Error loading mapped_levels.json: {e}")
        return
    descriptions = set([item.stem for item in input_dir.glob("*.txt")])
    for file in tqdm(input_dir.glob("*.png")):
        filename = file.name
        if '_' in filename:
            continue
        if filename.split(".")[0] in descriptions:
            continue
        print(f"Processing: {filename} with level {mapped_levels[filename]}")
        keep, caption = process_file(mapped_levels[filename], input_dir / filename)
        if keep:
            with open(input_dir / f"{filename.replace('.png', '.txt')}", 'w+') as f:
                f.write(caption)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./output", help="Input directory")
    args = parser.parse_args()
    main(args.input_dir)