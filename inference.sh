cd /workspace/musubi-tuner/

python3.10 -m venv .venv
source .venv/bin/activate
#    --dit /workspace/layered/model/newfull/layered-full.safetensors \
#    --dit /workspace/layered/model/default/qwen_image_layered_bf16.safetensors \
#layered-lora
python /workspace/musubi-tuner/src/musubi_tuner/qwen_image_generate_image.py \
    --dit /workspace/layered/model/default/qwen_image_layered_bf16.safetensors \
    --vae /workspace/layered/model/default/qwen_image_layered_vae.safetensors \
    --text_encoder /workspace/models/qwen_2.5_vl_7b_bf16.safetensors \
    --control_image_path /workspace/layered/control_images/2.png \
    --output_layers 3 \
    --prompt "Separate characters from the background of the image" \
    --negative_prompt " " \
    --image_size 1072 1584\
    --infer_steps 10 \
    --guidance_scale 4.0 \
    --save_path /workspace/layered/control_images_result/ \
    --output_type images \
    --seed 1234  \
    --model_version layered \
    --resize_control_to_image_size
#    --attn_mode sdpa \
#    --lora_multiplier 1.0 --lora_weight /workspace/layered/model/newfull/layered-lora.safetensors \