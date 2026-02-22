#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from PIL import Image
from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

from helper import pad_rgba_to_multiple_edge, shards, clean_alpha_noise, crop_to_content, restore_from_crop

from dotenv import load_dotenv
import os

# Load .env file from current directory
load_dotenv(override=True)

BASE_PATH = os.getenv("BASE_PATH")
if not BASE_PATH:
    raise RuntimeError("BASE_PATH environment variable is not set")
BASE_PATH = BASE_PATH.rstrip("/")


class QIL:
    def __init__(self, device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,):
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
        layer_num: int = 9) -> List[Image.Image]:
        
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        
        image, crop_box = crop_to_content(image)

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

        ready: List[Image.Image] = []
        ow, oh = orig_size
        for img in images:
            """Remove image padding for the model usage"""
            img = img.crop((0, 0, ow, oh))
            """Reverse crop the image to the original size"""
            img = restore_from_crop(img, crop_box)
            """Remove small alpha noise"""
            ready.append(clean_alpha_noise(img))
        return ready
