#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

from contextlib import contextmanager
from diffusers import QwenImageLayeredPipeline
import importlib
import functools

from helper import pad_rgba_to_multiple_edge, shards, clean_alpha_noise, crop_to_content, restore_from_crop

from dotenv import load_dotenv
import os

# Load .env file from current directory
load_dotenv(override=True)

BASE_PATH = os.getenv("BASE_PATH")
if not BASE_PATH:
    raise RuntimeError("BASE_PATH environment variable is not set")
BASE_PATH = BASE_PATH.rstrip("/")

"""pipeline based on the DiffSynth Studio"""
class DiffSynth:
    def __init__(self, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        self.multiple_of = 16
        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=dtype,
            device=device,
            model_configs=[
                ModelConfig(path=f"{BASE_PATH}/transformer/diffusion_pytorch_model.safetensors"),
                ModelConfig(path=shards(f"{BASE_PATH}/text_encoder/model*.safetensors")),
                ModelConfig(path=f"{BASE_PATH}/vae/diffusion_pytorch_model.safetensors"),
            ],
            tokenizer_config=ModelConfig(path=f"{BASE_PATH}/tokenizer/"),
            processor_config=None,
        )
    
    """
    Run the DiffSynth inference for the given image and prompt. Images are aligned (padded to higher multiple) to the model multiple and return back of the original size.
    """
    def inference(self, image: Image.Image, prompt: str, 
                  num_inference_steps: int = 10, cfg_scale: float = 2.5, 
                  layer_num: int = 9) -> list[Image.Image]:
        
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        img_padded, orig_size = pad_rgba_to_multiple_edge(image, self.multiple_of)
        target_w, target_h = img_padded.size


        images: Iterable[Image.Image] = self.pipe(prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        height=target_h ,
                        width=target_w,
                        cfg_scale=cfg_scale,
                        layer_input_image=img_padded,
                        layer_num=layer_num,
                        tiled=True,
                        tile_size=256,   
                        tile_stride=128)

        ow, oh = orig_size
        """Remove image padding for the model usage"""
        return [img.crop((0, 0, ow, oh)) for img in images]

"""pipeline based on the HF Diffuser"""
class HFDiffuser:

    def __init__(self, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        self.pipe = None

        #dit_checkpoint_path = Path(args.dit_checkpoint_path)

        print("Loading Qwen-Image-Layered pipeline...")
        self.pipe = QwenImageLayeredPipeline.from_pretrained( BASE_PATH, torch_dtype=dtype)
        self.pipe = self.pipe.to(device=device, dtype=dtype)
        self.pipe.vae.to(device=device, dtype=dtype)

        try:
            self.pipe.vae.enable_slicing()
            print("Enabled VAE slicing.")
        except Exception:
            # fallback for older codepaths
            if hasattr(self.pipe.vae, "use_slicing"):
                self.pipe.vae.use_slicing = True
                print("Enabled VAE slicing via use_slicing=True.")
            else:
                print("[warn] Could not enable VAE slicing; attribute not found.")

    @contextmanager
    def force_qwen_layered_dimensions(self,target_w: int, target_h: int):
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
    def fixed_flow_shift(self, shift: float = 1.0, ignore_mu: bool = True):
        sched = self.pipe.scheduler

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
    
    def inference(self, image: Image.Image, prompt: str, num_inference_steps: int = 10, cfg_scale: float = 2.5, layer_num: int = 9) -> List[Image.Image]:
        torch.cuda.empty_cache()
        
        # The pipeline itself uses multiple_of = self.vae_scale_factor * 2 and floors sizes to it :contentReference[oaicite:2]{index=2}
        multiple_of = int(self.pipe.vae_scale_factor * 2)

        img_padded, orig_size = pad_rgba_to_multiple_edge(image, multiple_of)
        
        width, height = image.size
        target_w, target_h = img_padded.size

        with self.force_qwen_layered_dimensions(target_w, target_h), self.fixed_flow_shift(shift=1.0):
            out = self.pipe(
                image=img_padded,
                prompt=prompt,
                negative_prompt="",
                true_cfg_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                layers=layer_num,
                resolution=1024, 
                cfg_normalize=False,
                use_en_prompt=True
            )

        layers = out.images[0]  # list of RGBA PIL images

        # If we padded, optionally crop back to original exact input size
        ow, oh = orig_size
        return [layer.crop((0, 0, ow, oh)) for layer in layers]


class QIL:
    def __init__(self, device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16, model_type: str = "diffsynth"):
        self.model = DiffSynth(dtype, device) if model_type == "diffsynth" else HFDiffuser(dtype, device)
    
    """
    Run the selected model inference for the given image and prompt. 
    Images are aligned (padded to higher multiple) to the model multiple and return back of the original size.
    """
    def inference(self, image: Image.Image, prompt: str, 
        num_inference_steps: int = 10, cfg_scale: float = 2.5, 
        layer_num: int = 9) -> List[Image.Image]:
        
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        
        image, crop_box = crop_to_content(image)

        layers: list[Image.Image] = self.model.inference(image, prompt, num_inference_steps, cfg_scale, layer_num)


        ready: List[Image.Image] = []
        for img in layers:
            """Reverse crop the image to the original size"""
            img = restore_from_crop(img, crop_box)
            """Remove small alpha noise after generation"""
            ready.append(clean_alpha_noise(img))
        return ready
