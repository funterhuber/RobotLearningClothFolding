from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("jjr1007/may7_first16_last48_newMethod", root="/Users/ferdinandunterhuber/.cache/huggingface/lerobot")
dataset.push_to_hub()