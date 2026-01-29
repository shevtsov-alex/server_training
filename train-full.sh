#!/bin/bash

# Generate timestamp-based folder name
EPOCHS=150
TIMESTAMP=$(date +"%Y_%m_%d__%H_%M_%S")
RUN_NAME="layered-fullscale-fullhd_bck_false-${EPOCHS}-${TIMESTAMP}"
OUTPUT_BASE="/workspace/layered/model"
RUN_DIR="${OUTPUT_BASE}/${RUN_NAME}"

# Create the run directory
mkdir -p "${RUN_DIR}"
mkdir -p "${RUN_DIR}/logs"

cd /workspace/musubi-tuner/

python3.10 -m venv .venv
source .venv/bin/activate

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 /workspace/musubi-tuner/src/musubi_tuner/qwen_image_train.py \
    --sdpa \
    --dit /workspace/layered/model/default/qwen_image_layered_bf16.safetensors \
    --vae /workspace/layered/model/default/qwen_image_layered_vae.safetensors \
    --text_encoder /workspace/models/qwen_2.5_vl_7b_bf16.safetensors \
    --mixed_precision bf16 \
    --full_bf16 --fused_backward_pass \
    --timestep_sampling qwen_shift \
    --weighting_scheme none \
    --optimizer_type adafactor --learning_rate 4e-6 --gradient_checkpointing\
    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" "weight_decay=0.01" \
    --max_grad_norm 0 --lr_scheduler constant \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --max_train_epochs ${EPOCHS} --save_every_n_epochs 25 --seed 42 \
    --dataset_config /workspace/layered/dataset.toml \
    --output_dir "${RUN_DIR}" \
    --output_name layered-full \
    --model_version layered \
    --logging_dir "${RUN_DIR}/logs" \
    --log_prefix "${RUN_NAME}_" \
    --log_with tensorboard \
    --remove_first_image_from_target

# Print completion message
echo "========================================"
echo "Training completed!"
echo "Models saved in: ${RUN_DIR}"
echo "Logs saved in: ${RUN_DIR}/logs"
echo "Latest run symlink: ${OUTPUT_BASE}/latest"
echo "========================================"
