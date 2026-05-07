import numpy as np
from datasets import load_dataset

# Load your Hugging Face dataset
repo_id = "jjr1007/5may_lorenzo_merged_1-4_6-12" 
dataset = load_dataset(repo_id, split="train")

episode_frozen_stats = {}

# Group by episode
for i in range(dataset.features['episode_index'].num_classes if hasattr(dataset.features['episode_index'], 'num_classes') else max(dataset['episode_index']) + 1):
    ep_data = dataset.filter(lambda x: x['episode_index'] == i)
    actions = np.array(ep_data['action'])
    
    # Calculate difference between consecutive actions
    action_diffs = np.diff(actions, axis=0)
    
    # Check where the difference is effectively zero across all joints/motors
    frozen_frames = np.all(np.abs(action_diffs) < 1e-5, axis=1)
    
    frozen_count = np.sum(frozen_frames)
    total_frames = len(actions)
    
    if frozen_count > 0:
         episode_frozen_stats[i] = (frozen_count, total_frames, frozen_count / total_frames * 100)

# Print the full list
for ep, (frozen, total, pct) in episode_frozen_stats.items():
    print(f"Episode {ep}: {frozen}/{total} frames frozen ({pct:.1f}%)")
    
import pickle

with open("episode_frozen_stats.pkl", "wb") as f:
    pickle.dump(episode_frozen_stats, f)