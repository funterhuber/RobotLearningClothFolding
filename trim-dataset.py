from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

N_FIRST = 50
N_LAST = 10
SOURCE = "jjr1007/may7_merged"
TARGET = "jjr1007/may7_merged_trimmed_part3"
START_EPISODE = 100
END_EPISODE = 169
TASK = "Fold the cloth twice, one from the closest corner and the second from the second closest corner"

# Use local cache to avoid re-downloading ~2GB
LOCAL_ROOT = Path.home() / ".cache/huggingface/lerobot"

print(f"Loading source dataset (episodes {START_EPISODE}-{END_EPISODE})...")
episodes_to_process = list(range(START_EPISODE, END_EPISODE + 1))
source = LeRobotDataset(
    SOURCE,
    episodes=episodes_to_process,
    video_backend="pyav",
    root=LOCAL_ROOT,
)
print(f"Loaded {len(source)} frames across {len(episodes_to_process)} episodes")

# Build episode -> frame indices map (one pass)
episode_to_indices = {}
for i in range(len(source)):
    ep_idx = int(source.hf_dataset[i]["episode_index"])
    episode_to_indices.setdefault(ep_idx, []).append(i)

# Create new dataset
print(f"\nCreating target dataset: {TARGET}")
new_dataset = LeRobotDataset.create(
    repo_id=TARGET,
    fps=source.fps,
    robot_type=source.meta.robot_type,
    features=source.features,
)

total_saved = 0
for ep_idx in episodes_to_process:
    frames = episode_to_indices.get(ep_idx, [])
    if len(frames) <= N_FIRST + N_LAST:
        print(f"Episode {ep_idx}: only {len(frames)} frames, skipping (too short to trim)")
        continue

    trimmed = frames[N_FIRST : len(frames) - N_LAST]
    print(f"Episode {ep_idx}: {len(frames)} → {len(trimmed)} frames", end="", flush=True)

    for i in trimmed:
        sample = source[i]
        new_dataset.add_frame({
            "action": sample["action"],
            "observation.state": sample["observation.state"],
            "observation.images.front": sample["observation.images.front"].permute(1, 2, 0),  # CHW -> HWC
            "task": TASK,
        })

    new_dataset.save_episode()
    total_saved += 1
    print(f" saved")

print(f"\nAll {total_saved} episodes saved locally.")

# CRITICAL: finalize writes the parquet footers — skipping this was the bug
print("Finalizing dataset (writing parquet footers)...")
new_dataset.finalize()

# Push once, after finalize
print("Pushing to HuggingFace Hub...")
new_dataset.push_to_hub()
print("Done! part3 complete.")
