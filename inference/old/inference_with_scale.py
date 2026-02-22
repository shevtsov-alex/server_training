#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import re
from pathlib import Path
from typing import Dict, Any

import torch
from PIL import Image
from diffusers import QwenImageLayeredPipeline


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
        description="Option A: run Qwen-Image-Layered at bucket size, then resize output layers back to original input size."
    )
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--dit_checkpoint_path", required=True)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--device", default="cuda")

    
    parser.add_argument("--out_dir", default="/workspace/layered/out", help="Where to save layers.")
    parser.add_argument("--seed", type=int, default=712398737)

    # Inference knobs
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--negative_prompt", type=str, default=" ")
    parser.add_argument("--cfg_normalize", action="store_true", default=False)
    parser.add_argument("--use_en_prompt", action="store_true", default=True)
    parser.add_argument("--resolution", type=int, default=1024, choices=(640, 1024),
                        help="Bucket gate used by the pipeline (640 or 1024).")

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
            "Unsupported checkpoint extension. Expected .safetensors, .pt, .pth, or .bin."
        )

    if not isinstance(state, dict):
        raise TypeError("Checkpoint did not contain a state dict.")

    if "state_dict" in state and isinstance(state["state_dict"], dict):
        print("Checkpoint has nested 'state_dict'; using that for DiT weights.")
        return state["state_dict"]

    return state


def main() -> None:
    args = parse_args()
    require_transformers("4.51.3")
    require_torchvision()

    dtype = resolve_dtype(args.dtype)
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen-Image-Layered pipeline...")
    pipe = QwenImageLayeredPipeline.from_pretrained(args.pretrained_path, torch_dtype=dtype)

    print(f"Loading DiT checkpoint: {args.dit_checkpoint_path}")
    dit_state_dict = load_checkpoint(Path(args.dit_checkpoint_path))

    print("Updating DiT weights...")
    pipe.transformer.load_state_dict(dit_state_dict, strict=True)

    pipe = pipe.to(device=device, dtype=dtype)

    # VAE decode memory helpers (usually safe at bucket sizes too)
    try:
        pipe.vae.enable_slicing()
        print("Enabled VAE slicing.")
    except Exception:
        if hasattr(pipe.vae, "use_slicing"):
            pipe.vae.use_slicing = True
            print("Enabled VAE slicing via use_slicing=True.")

    print('No VAE tiling')
    print("Pipeline is ready.")

    prompt = """L1_ENTITY ISOLATE CLEAN_EDGE character person environment split 
Identify and isolate every character in the scene as an individual layer. Ensure all character edges are sharp and clean.
Keep internal shadows that define the character's body and clothes. Exclude shadows cast by the character onto the ground or walls; these must stay in the separate layer.
Layer Hierarchy: Characters (Each person isolated with their handheld objects), Foreground (Any object blocking or overlapping a character must be its own separate layer) and Background (The remaining environment including all floor and wall shadows). 
"""

    img = Image.open("/workspace/layered/control_images/2.png").convert("RGBA")
    ow, oh = img.size
    print(f"Original input size: {ow} x {oh}")

    gen = torch.Generator(device=device).manual_seed(args.seed)

    # IMPORTANT: no forced sizing here.
    # Pipeline will internally resize the input to a bucket size derived from `resolution` + aspect ratio.
    out = pipe(
        image=img,
        prompt=prompt,
        negative_prompt=" ",
        true_cfg_scale=4.0,
        num_inference_steps=10,
        num_images_per_prompt=1,
        layers=8,
        resolution=1024,
        cfg_normalize=False,
        use_en_prompt=True,
        generator=gen,
    )

    layers = out.images[0]  # list of RGBA PIL images
    bw, bh = layers[0].size
    print(f"Bucket output size: {bw} x {bh}")

    # Resize each layer back to original input size
    for i, layer in enumerate(layers):
        layer_back = layer.resize((ow, oh), resample=Image.Resampling.LANCZOS)
        layer_back.save(f"/workspace/layered/out/t1_scaled_layer_{i}.png")

    print(f"Saved {len(layers)} layers resized back to {ow}x{oh} into: {out_dir}")


if __name__ == "__main__":
    main()
