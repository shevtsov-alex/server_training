cd /workspace/musubi-tuner/

python3.10 -m venv .venv
source .venv/bin/activate


accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 /workspace/musubi-tuner/src/musubi_tuner/qwen_image_train_network.py \
    --sdpa \
    --dit /workspace/layered/model/default/qwen_image_layered_bf16.safetensors \
    --vae /workspace/layered/model/default/qwen_image_layered_vae.safetensors \
    --text_encoder /workspace/models/qwen_2.5_vl_7b_bf16.safetensors \
    --timestep_sampling qwen_shift \
    --weighting_scheme none \
    --optimizer_type adafactor --learning_rate 2e-6 --gradient_checkpointing \
    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" "weight_decay=0" \
    --max_train_epochs 50 --save_every_n_epochs 10 --seed 42 \
    --dataset_config /workspace/layered/dataset.toml \
    --output_dir /workspace/layered/model/newfull \
    --output_name layered-lora \
    --model_version layered \
    --network_module networks.lora_qwen_image \
    --network_dim 16 \
    --remove_first_image_from_target \
    --max_grad_norm 0 --lr_scheduler cosine \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers
