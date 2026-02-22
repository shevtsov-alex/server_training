#!/bin/bash

# Generate timestamp-based folder name
EPOCHS=500
EPOCHS_SAVE=50
TIMESTAMP=$(date +"%Y_%m_%d__%H_%M_%S")
LR=2e-5
NAME_PREFIX="layered-buckfalse-w00-const-all-layers-notaligned-${LR}-${EPOCHS}"
RUN_NAME="${NAME_PREFIX}-${TIMESTAMP}"
DATASET_FILE="/workspace/layered/dataset-v2_2.toml"
OUTPUT_BASE="/workspace/layered/model"
RUN_DIR="${OUTPUT_BASE}/${RUN_NAME}"

GRAD_ACC=1
GRAD_PARAM=""

if [ ${GRAD_ACC} -gt 1 ]; then
    GRAD_PARAM="--max_grad_norm 1.0 --gradient_accumulation_steps ${GRAD_ACC} "
else
    GRAD_PARAM="--max_grad_norm 0"
fi
echo ${GRAD_PARAM}

echo "========================================"
echo "Training parameters:"
echo "  EPOCHS:            ${EPOCHS}"
echo "  EPOCHS_SAVE:       ${EPOCHS_SAVE}"
echo "  LR:                ${LR}"
echo "  NAME_PREFIX:       ${NAME_PREFIX}"
echo "  RUN_NAME:          ${RUN_NAME}"
echo "  DATASET_FILE:      ${DATASET_FILE}"
echo "  OUTPUT_BASE:       ${OUTPUT_BASE}"
echo "  RUN_DIR:           ${RUN_DIR}"
echo "  GRAD_PARAM:        ${GRAD_PARAM}"
echo "========================================"
echo -n "Proceed with these settings? (y/n): "
read -r CONFIRM
if [ "${CONFIRM}" != "y" ]; then
    echo "Aborted by user."
    exit 1
fi


# Create the run directory
mkdir -p "${RUN_DIR}"
mkdir -p "${RUN_DIR}/logs"

cd /workspace/layered/musubi-tuner/

python3.10 -m venv .venv
source .venv/bin/activate

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 /workspace/layered/musubi-tuner/src/musubi_tuner/qwen_image_train.py \
    --sdpa \
    --dit /workspace/layered/model/default/qwen_image_layered_bf16.safetensors \
    --vae /workspace/layered/model/default/qwen_image_layered_vae.safetensors \
    --text_encoder /workspace/layered/model/default/qwen_2.5_vl_7b_bf16.safetensors \
    --mixed_precision bf16 \
    --full_bf16 --fused_backward_pass \
    --timestep_sampling qwen_shift \
    --weighting_scheme none \
    --optimizer_type adafactor --learning_rate ${LR} --gradient_checkpointing\
    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" "weight_decay=0.0" \
    --lr_scheduler constant \
    ${GRAD_PARAM} \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --max_train_epochs ${EPOCHS} --save_every_n_epochs ${EPOCHS_SAVE} --seed 42 \
    --dataset_config "${DATASET_FILE}" \
    --output_dir "${RUN_DIR}" \
    --output_name "${NAME_PREFIX}" \
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
