#!/usr/bin/env python3
#export HF_HOME="/workspace/layered/hf_cache/"
import os, glob
import torch, math
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import numpy as np
import code

def pad_rgba_to_multiple_edge(img: Image.Image, multiple: int):
    w, h = img.size
    new_w = math.ceil(w / multiple) * multiple
    new_h = math.ceil(h / multiple) * multiple
    if (new_w, new_h) == (w, h):
        return img, (w, h)

    arr = np.array(img)#img.convert("RGBA"))
    pad_w = new_w - w
    pad_h = new_h - h

    # edge padding = повторяем крайние пиксели
    arr2 = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    out = Image.fromarray(arr2, mode="RGBA")
    return out, (w, h)

BASE = "/workspace/layered/qwen_layered_base"
def shards(pat: str):
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(pat)
    return files

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=f"{BASE}/transformer/diffusion_pytorch_model.safetensors"),
        # Qwen2.5-VL TE is correct for DiffSynth's QwenImageTextEncoder
        #ModelConfig(path="/workspace/layered/model/default/qwen_2.5_vl_7b_bf16.safetensors"),#shards(f"{BASE}/text_encoder/model*.safetensors")),
        ModelConfig(path=shards(f"{BASE}/text_encoder/model*.safetensors")),
        #ModelConfig(path="/workspace/layered/model/default/qwen_image_layered_vae.safetensors"),#/vae/diffusion_pytorch_model.safetensors"),
        ModelConfig(path=f"{BASE}/vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(path=f"{BASE}/tokenizer/"),
    processor_config=None,#ModelConfig(path=f"{BASE}/processor/"),
)

img = Image.open("/workspace/layered/control_images/1_crop.png").convert("RGBA")

multiple_of = 16
print(f'Mitplier of: {multiple_of}')
img_padded, orig_size = pad_rgba_to_multiple_edge(img, multiple_of)

width, height = img.size
print(f'Original image size: {width} x {height}')
target_w, target_h = img_padded.size
print(f'Padded image size: {target_w} x {target_h}')

prompt = """L1_ENTITY ISOLATE CLEAN_EDGE character person environment split"""

images = pipe(
    prompt,
    #seed=972047012108803,
    num_inference_steps=50,
    height=target_h ,
    width=target_w,
    cfg_scale=2.5,
    #exponential_shift_mu=1.0,
    #denoising_strength=0.95,       # << anchor to the input
    layer_input_image=img_padded,
    layer_num=2,     # IMPORTANT: keep small at 2048
    tiled=False,      # helps VAE memory
    #tile_size=256,   # you can try 128/256
    #tile_stride=128,
)



# If we padded, optionally crop back to original exact input size
for i, img in enumerate(images):
    ow, oh = orig_size
    img_native = img.crop((0, 0, ow, oh))
    img_native.save(f"/workspace/layered/out/df1_raw_img_1_down_s2_not_tiled_50steps_layer_{i}.png")

#code.interact(local=dict(globals(), **locals()))
