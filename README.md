Scripts used to train gen-1 Qwen Image Layered model including the inference pipeline

Huggingface [link](https://huggingface.co/playrix/Qwen-Image-Layered)

parameters:
 - mixed_precision bf16 
 - sdpa 
 - full_bf16 
 - fused_backward_pass 
 - timestep_sampling qwen_shift 
 - weighting_scheme none 
 - optimizer_type adafactor 
 - learning_rate 2e-5 
 - gradient_checkpointing 
 - gradient_accumulation_steps 8 
 - optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" "weight_decay=0.0" 
 - max_grad_norm 1.0 
 - lr_scheduler constant 
 - max_data_loader_n_workers 2 
 - persistent_data_loader_workers 
 - max_train_epochs 400 
 - save_every_n_epochs 50 
 - seed 42 
 - model_version layered 
 - log_with tensorboard 
 - remove_first_image_from_target