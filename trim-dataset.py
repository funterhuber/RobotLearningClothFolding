from lerobot.datasets.lerobot_dataset import LeRobotDataset

N_FIRST = 50
N_LAST = 10
SOURCE = "jjr1007/may7_merged"
TARGET = "jjr1007/may7_merged_trimmed_part2"
START_EPISODE = 101
END_EPISODE = 169

dataset = LeRobotDataset(SOURCE, video_backend="pyav")

new_dataset = LeRobotDataset.create(
    repo_id=TARGET,
    fps=dataset.fps,
    robot_type=dataset.meta.robot_type,
    features=dataset.features,
)

print(f"Processing episodes {START_EPISODE} to {END_EPISODE}")

for episode_idx in range(START_EPISODE, END_EPISODE + 1):
    all_indices = [
        i for i in range(len(dataset))
        if dataset.hf_dataset[i]["episode_index"] == episode_idx
    ]
    trimmed_indices = all_indices[N_FIRST:-N_LAST]
    print(f"Episode {episode_idx}: {len(all_indices)} → {len(trimmed_indices)} frames")
    for idx in trimmed_indices:
        sample = dataset[idx]
        frame = {
            "action": sample["action"],
            "observation.state": sample["observation.state"],
            "observation.images.front": sample["observation.images.front"].permute(1, 2, 0),
            "task": "Fold the cloth twice, one from the closest corner and the second from the second closest corner",
        }
        new_dataset.add_frame(frame)
    new_dataset.save_episode()
    print(f"  Episode {episode_idx} saved ✅")
    if (episode_idx + 1) % 20 == 0:
        print(f"Pushing checkpoint...")
        new_dataset.push_to_hub()

print("Pushing final...")
new_dataset.push_to_hub()
print("Done! part1 complete")

