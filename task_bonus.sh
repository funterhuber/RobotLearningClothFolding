source /opt/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
source /home/team16/RobotLearningClothFolding/.env
 

python -m lerobot.scripts.lerobot_rollout     --robot.type=so101_follower     --robot.port=/dev/ttyACM0     --robot.id=my_awesome_follower_arm     --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"     --strategy.type=dagger     --policy.path=jjr1007/diffusion_dino_dagger_ft --policy.num_inference_steps=10     --device=cuda     --compile_warmup_inferences=5     --fps=20     --interpolation_multiplier=3    --teleop.type=so101_leader     --teleop.id=my_awesome_leader_arm     --teleop.port=/dev/ttyACM1     --strategy.num_episodes=50     --dataset.repo_id=jjr1007/rollout_delete --strategy.record_autonomous=true 