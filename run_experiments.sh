#!/bin/bash

# Run 1: baseline - horizon 49, action steps 8
python train_wrapper.py \
  --dataset.repo_id=jjr1007/5may_lorenzo_merged_1-4_6-12 \
  --output_dir=./outputs/run1_baseline \
  --batch_size=32 \
  --steps=10000 \
  --save_freq=2000 \
  --log_freq=100 \
  --eval_freq=1000 \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.horizon=49 \
  --policy.n_action_steps=8 \
  --policy.objective=flow_matching \
  --policy.timestep_sampling_strategy=beta \
  --policy.timestep_sampling_alpha=1.5 \
  --policy.timestep_sampling_beta=1.0 \
  --policy.timestep_sampling_s=0.999 \
  --policy.num_integration_steps=100 \
  --policy.integration_method=euler \
  --policy.sigma_min=0.0 \
  --policy.num_layers=6 \
  --policy.hidden_dim=512 \
  --policy.vision_encoder_name=openai/clip-vit-base-patch16 \
  --policy.image_resize_shape='[224,224]' \
  --policy.image_crop_shape='[224,224]' \
  --policy.repo_id="jjr1007/multitask-dit-run1-baseline" \
  --policy.use_amp=true \
  --wandb.enable=true \
  --wandb.project=multitask-dit-experiments \
  --wandb.run_id=run1_baseline_h49_a8

# Run 2: larger action steps
python train_wrapper.py \
  --dataset.repo_id=jjr1007/5may_lorenzo_merged_1-4_6-12 \
  --output_dir=./outputs/run2_larger_steps \
  --batch_size=32 \
  --steps=10000 \
  --save_freq=2000 \
  --log_freq=100 \
  --eval_freq=1000 \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.horizon=49 \
  --policy.n_action_steps=16 \
  --policy.objective=flow_matching \
  --policy.timestep_sampling_strategy=beta \
  --policy.timestep_sampling_alpha=1.5 \
  --policy.timestep_sampling_beta=1.0 \
  --policy.timestep_sampling_s=0.999 \
  --policy.num_integration_steps=100 \
  --policy.integration_method=euler \
  --policy.sigma_min=0.0 \
  --policy.num_layers=6 \
  --policy.hidden_dim=512 \
  --policy.vision_encoder_name=openai/clip-vit-base-patch16 \
  --policy.image_resize_shape='[224,224]' \
  --policy.image_crop_shape='[224,224]' \
  --policy.repo_id="jjr1007/multitask-dit-run2-larger-steps" \
  --policy.use_amp=true \
  --wandb.enable=true \
  --wandb.project=multitask-dit-experiments \
  --wandb.run_id=run2_larger_steps_h49_a16

# Run 3: smaller model
python train_wrapper.py \
  --dataset.repo_id=jjr1007/5may_lorenzo_merged_1-4_6-12 \
  --output_dir=./outputs/run3_small_model \
  --batch_size=32 \
  --steps=10000 \
  --save_freq=2000 \
  --log_freq=100 \
  --eval_freq=1000 \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.horizon=49 \
  --policy.n_action_steps=8 \
  --policy.objective=flow_matching \
  --policy.timestep_sampling_strategy=beta \
  --policy.timestep_sampling_alpha=1.5 \
  --policy.timestep_sampling_beta=1.0 \
  --policy.timestep_sampling_s=0.999 \
  --policy.num_integration_steps=100 \
  --policy.integration_method=euler \
  --policy.sigma_min=0.0 \
  --policy.num_layers=4 \
  --policy.hidden_dim=256 \
  --policy.vision_encoder_name=openai/clip-vit-base-patch16 \
  --policy.image_resize_shape='[224,224]' \
  --policy.image_crop_shape='[224,224]' \
  --policy.repo_id="jjr1007/multitask-dit-run3-small-model" \
  --policy.use_amp=true \
  --wandb.enable=true \
  --wandb.project=multitask-dit-experiments \
  --wandb.run_id=run3_small_model_h49_a8

echo "All runs done"