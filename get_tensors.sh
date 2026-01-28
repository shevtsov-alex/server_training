#!/bin/bash
cd /workspace/layered/model/default/

echo 'Removing old tensors ....'
rm *.safetensors*

echo 'Downloading the vae tensors ....'

hf download Comfy-Org/Qwen-Image-Layered_ComfyUI split_files/vae/qwen_image_layered_vae.safetensors   --local-dir /workspace/layered/model/default

echo 'Downloading DiT ....'

hf download Comfy-Org/Qwen-Image-Layered_ComfyUI split_files/diffusion_models/qwen_image_layered_bf16.safetensors    --local-dir /workspace/layered/model/default

mv model/default/split_files/diffusion_models/qwen_image_layered_bf16.safetensors model/default 
mv model/default/split_files/vae/qwen_image_layered_vae.safetensors model/default 