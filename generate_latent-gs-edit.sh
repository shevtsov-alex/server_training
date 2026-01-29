#!/usr/bin/env bash
cd /workspace
set -e  # завершить при ошибке

# Проверка существования папки musubi-tuner
if [ ! -d "musubi-tuner" ]; then
    echo "Ошибка: папка musubi-tuner не найдена! Необходимо запустить команду clone_musubi-tuner.sh"
    exit 1
fi

# Параметры
DATASET_FILE="${1:-dataset.toml}"   # первый аргумент, по умолчанию dataset.toml
BATCH_SIZE="${2:-1}"               # второй аргумент, по умолчанию 1

echo "Используется dataset: ${DATASET_FILE}"

cd /workspace/musubi-tuner

# venv
python3.10 -m venv .venv
source .venv/bin/activate

python /workspace/musubi-tuner/src/musubi_tuner/qwen_image_cache_latents.py \
    --dataset_config "/workspace/layered/${DATASET_FILE}" \
    --vae /workspace/layered/model/default/qwen_image_layered_vae.safetensors \
    --model_version layered

python /workspace/musubi-tuner/src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
    --dataset_config "/workspace/layered/${DATASET_FILE}" \
    --text_encoder /workspace/models/qwen_2.5_vl_7b_bf16.safetensors \
    --model_version layered


