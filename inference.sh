cd /workspace/musubi-tuner/

python3.10 -m venv .venv
source .venv/bin/activate
CONTROL_IMAGES="/workspace/layered/control_images"
RESULT_PATH="/workspace/layered/control_images_result"
RESULT_TMP="$RESULT_PATH/tmp"


mkdir -p -- "$RESULT_TMP"

shopt -s nullglob nocaseglob
files=("$CONTROL_IMAGES"/*.png)



#    --dit /workspace/layered/model/newfull/layered-full.safetensors \
#    --dit /workspace/layered/model/default/qwen_image_layered_bf16.safetensors \
#layered-lora
#    --attn_mode sdpa \
#    --lora_multiplier 1.0 --lora_weight /workspace/layered/model/newfull/layered-lora.safetensors \
i=1
for f in "${files[@]}"; do

  python /workspace/musubi-tuner/src/musubi_tuner/qwen_image_generate_image.py \
    --dit /workspace/layered/model/newfull/layered-full.safetensors \
    --vae /workspace/layered/model/default/qwen_image_layered_vae.safetensors \
    --text_encoder /workspace/models/qwen_2.5_vl_7b_bf16.safetensors \
    --control_image_path "$f" \
    --output_layers 3 \
    --prompt "Separate characters from the background of the image" \
    --negative_prompt " " \
    --image_size 1072 1584\
    --infer_steps 10 \
    --guidance_scale 4.0 \
    --save_path "$RESULT_TMP" \
    --output_type images \
    --seed 1234  \
    --model_version layered \
    --resize_control_to_image_size

  bash /workspace/layered/temp_files_management.sh -s "$RESULT_TMP" -d "$RESULT_PATH" -p "full_300_${i}_"

  ((i++))
done
