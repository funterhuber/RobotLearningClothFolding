import numpy as np
from datasets import load_dataset
from huggingface_hub import login
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import recompute_stats


# 2. Load your Hugging Face dataset
# Because we logged in above, this will successfully bypass the 404 and download from the Hub
repo_id = "jjr1007/may7_first16_last48_newMethod" 
print(f"Fetching {repo_id} from the Hub...")
dataset = LeRobotDataset(repo_id)

# 3. Edit the dataset
print("Recomputing stats...")
recompute_stats(dataset, relative_action=True, chunk_size=50, relative_exclude_joints=["gripper"])

# 4. Push it back to the Hub
print("Pushing updates back to the Hub...")
dataset.repo_id = repo_id + "_relative"
dataset.push_to_hub()
print("Done!")