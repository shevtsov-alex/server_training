#!/usr/bin/env python3
################################################################################
#------------------------------------Commands------------------------------------
#export HF_HOME="/workspace/layered/hf_cache/"
#python3 new_inference.py --pretrained_path Qwen/Qwen-Image-Layered --dit_checkpoint_path /workspace/layered/model/layered-buckfalse-w00-const-all-layers-notaligned-4e6-500-2026_02_13__18_33_38/layered-buckfalse-w00-const-all-layers-4e6-notaligned-000400.safetensors --dtype bf16 --device cuda
#----------------------------------------------------------------
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import re, math
from pathlib import Path
from typing import Dict, Any
from PIL import Image
from contextlib import contextmanager
import torch
import numpy as np
from diffusers import QwenImageLayeredPipeline
import functools

def parse_version(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version)
    if not parts:
        raise ValueError(f"Cannot parse version: {version}")
    return tuple(int(part) for part in parts)


def require_transformers(min_version: str) -> None:
    installed = importlib.metadata.version("transformers")
    if parse_version(installed) < parse_version(min_version):
        raise RuntimeError(
            "Transformers is too old for Qwen-Image-Layered. "
            f"Installed: {installed}. Required: >= {min_version}. "
            "Upgrade with: pip install -U transformers"
        )

def require_torchvision() -> None:
    if importlib.util.find_spec("torchvision") is None:
        raise RuntimeError(
            "torchvision is required for Qwen-Image-Layered AutoProcessor. "
            "Install it with: pip install torchvision"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Qwen-Image-Layered pipeline and update DiT weights."
    )
    parser.add_argument(
        "--pretrained_path",
        required=True,
        help="Hugging Face repo id or local path for Qwen-Image-Layered pipeline.",
    )
    parser.add_argument(
        "--dit_checkpoint_path",
        required=True,
        help="Local checkpoint path for DiT weights (.safetensors/.pt/.pth/.bin).",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=("bf16", "fp16", "fp32"),
        help="Torch dtype to use for loading the pipeline.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to move the pipeline to (e.g. cuda, cpu, cuda:0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=972047012108803,
        help="Seed for the random number generator.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if not path.is_file():
        raise ValueError(f"Checkpoint path is not a file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(path.as_posix())
    elif suffix in {".pt", ".pth", ".bin"}:
        state = torch.load(path.as_posix(), map_location="cpu")
    else:
        raise ValueError(
            "Unsupported checkpoint extension. "
            "Expected .safetensors, .pt, .pth, or .bin."
        )

    if not isinstance(state, dict):
        raise TypeError("Checkpoint did not contain a state dict.")

    if "state_dict" in state and isinstance(state["state_dict"], dict):
        print("Checkpoint has nested 'state_dict'; using that for DiT weights.")
        return state["state_dict"]

    return state


def pad_rgba_to_multiple(img: Image.Image, multiple: int) -> tuple[Image.Image, tuple[int, int]]:
    """
    Pads (not resizes) to make width/height divisible by `multiple`.
    Returns (padded_image, original_size).
    """
    img = img.convert("RGBA")
    ow, oh = img.size

    nw = ((ow + multiple - 1) // multiple) * multiple
    nh = ((oh + multiple - 1) // multiple) * multiple

    if (nw, nh) == (ow, oh):
        return img, (ow, oh)

    canvas = Image.new("RGBA", (nw, nh), (0, 0, 0, 0))
    canvas.paste(img, (0, 0), img)
    return canvas, (ow, oh)

def pad_rgba_to_multiple_edge(img: Image.Image, multiple: int):
    w, h = img.size
    new_w = math.ceil(w / multiple) * multiple
    new_h = math.ceil(h / multiple) * multiple
    if (new_w, new_h) == (w, h):
        return img, (w, h)

    arr = np.array(img.convert("RGBA"))
    pad_w = new_w - w
    pad_h = new_h - h

    # edge padding = повторяем крайние пиксели
    arr2 = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    out = Image.fromarray(arr2, mode="RGBA")
    return out, (w, h)

@contextmanager
def force_qwen_layered_dimensions(target_w: int, target_h: int):
    """
    Monkey-patches diffusers.pipelines.qwenimage.pipeline_qwenimage_layered.calculate_dimensions
    so the pipeline uses (target_w, target_h) instead of bucketed dims.
    """
    mod = importlib.import_module("diffusers.pipelines.qwenimage.pipeline_qwenimage_layered")
    orig = mod.calculate_dimensions

    def _forced_dims(_target_area, _ratio):
        # calculate_dimensions returns (width, height)
        return int(target_w), int(target_h)

    mod.calculate_dimensions = _forced_dims
    try:
        yield
    finally:
        mod.calculate_dimensions = orig

@contextmanager
def fixed_flow_shift(pipe, shift: float = 1.0, ignore_mu: bool = True):
    sched = pipe.scheduler

    old_dyn = getattr(sched.config, "use_dynamic_shifting", None)
    old_shift = getattr(sched, "_shift", None)
    old_set_timesteps = sched.set_timesteps

    if ignore_mu:
        @functools.wraps(old_set_timesteps)
        def set_timesteps_no_mu(*args, **kwargs):
            kwargs.pop("mu", None)  # ключевое
            return old_set_timesteps(*args, **kwargs)

    try:
        if old_dyn is not None:
            sched.config.use_dynamic_shifting = False

        if hasattr(sched, "set_shift"):
            sched.set_shift(float(shift))
        else:
            sched._shift = float(shift)

        if ignore_mu:
            sched.set_timesteps = set_timesteps_no_mu

        yield
    finally:
        if ignore_mu:
            sched.set_timesteps = old_set_timesteps
        if old_dyn is not None:
            sched.config.use_dynamic_shifting = old_dyn
        if old_shift is not None:
            if hasattr(sched, "set_shift"):
                sched.set_shift(old_shift)
            else:
                sched._shift = old_shift




def main() -> None:
    args = parse_args()
    require_transformers("4.51.3")
    require_torchvision()
    dtype = resolve_dtype(args.dtype)

    dit_checkpoint_path = Path(args.dit_checkpoint_path)

    print("Loading Qwen-Image-Layered pipeline...")
    pipe = QwenImageLayeredPipeline.from_pretrained(
        args.pretrained_path, torch_dtype=dtype
    )

    print(f"Loading DiT checkpoint: {dit_checkpoint_path}")
    dit_state_dict = load_checkpoint(dit_checkpoint_path)

    print("Updating DiT weights...")
    if hasattr(pipe, "dit"):
        dit_component = pipe.dit
        component_name = "dit"
    elif hasattr(pipe, "transformer"):
        dit_component = pipe.transformer
        component_name = "transformer"
    elif hasattr(pipe, "unet"):
        dit_component = pipe.unet
        component_name = "unet"
    else:
        available = ", ".join(sorted(pipe.components.keys()))
        raise AttributeError(
            "Cannot find DiT component in pipeline. "
            f"Available components: {available}"
        )
    
    print(f"Loading state dict into pipeline.{component_name}...")
    dit_component.load_state_dict(dit_state_dict, strict=True)

    pipe = pipe.to(device="cuda", dtype=dtype)
    pipe.vae.to(device="cuda", dtype=dtype)

    print("Pipeline is ready.")

    try:
        pipe.vae.enable_slicing()
        print("Enabled VAE slicing.")
    except Exception:
        # fallback for older codepaths
        if hasattr(pipe.vae, "use_slicing"):
            pipe.vae.use_slicing = True
            print("Enabled VAE slicing via use_slicing=True.")
        else:
            print("[warn] Could not enable VAE slicing; attribute not found.")

    print("No VAE tiling ")
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        
    prompt = """L1_ENTITY ISOLATE CLEAN_EDGE character person environment split 
Identify and isolate every character in the scene as an individual layer. Ensure all character edges are sharp and clean.
Keep internal shadows that define the character's body and clothes. Exclude shadows cast by the character onto the ground or walls; these must stay in the separate layer.
Layer Hierarchy: Characters (Each person isolated with their handheld objects), Foreground (Any object blocking or overlapping a character must be its own separate layer) and Background (The           remaining environment including all floor and wall shadows). 
"""
    # The pipeline itself uses multiple_of = self.vae_scale_factor * 2 and floors sizes to it :contentReference[oaicite:2]{index=2}
    multiple_of = int(pipe.vae_scale_factor * 2)

    img = Image.open("/workspace/layered/control_images/2.png").convert("RGB")
    img.putalpha(255)

    img_padded, orig_size = pad_rgba_to_multiple_edge(img, multiple_of)
    
    width, height = img.size
    print(f'Original image size: {width} x {height}')
    target_w, target_h = img_padded.size
    print(f'Padded image size: {target_w} x {target_h}')

    gen = torch.Generator(device="cuda").manual_seed(args.seed)

    # Force pipeline to "calculate" our native dims; it will still call resize(),
    # but since calculated dims == current dims, it's effectively a no-op. :contentReference[oaicite:3]{index=3}
    #with force_qwen_layered_dimensions(target_w, target_h):
    with force_qwen_layered_dimensions(target_w, target_h), fixed_flow_shift(pipe, shift=1.0):
        out = pipe(
            image=img_padded,
            prompt=prompt,
            negative_prompt="",
            true_cfg_scale=2.5,
            num_inference_steps=10,
            num_images_per_prompt=1,
            layers=2,
            resolution=1024, 
            cfg_normalize=False,
            use_en_prompt=True,
            generator=gen,
        )

    layers = out.images[0]  # list of RGBA PIL images

    # If we padded, optionally crop back to original exact input size
    ow, oh = orig_size
    for i, layer in enumerate(layers):
        layer_native = layer.crop((0, 0, ow, oh))
        layer_native.save(f"/workspace/layered/out/t1_layer_{i}.png")


if __name__ == "__main__":
    main()

