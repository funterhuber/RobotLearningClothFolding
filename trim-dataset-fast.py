"""Trim episodes from a LeRobot dataset using ffmpeg stream-copy for video.

Approach: do not decode + re-encode video. Cut the source MP4s with `ffmpeg -c copy`
at GOP-aligned boundaries, then hand the pre-trimmed segments to LeRobot's writer
internals so it just moves them into place (and stream-copy-concatenates them into
chunks). Action / state trimming is done with a single batched parquet read per
episode. End result: per-episode work goes from "decode + encode entire episode"
to "ffmpeg stream copy + one parquet write".

The GOP-alignment is the price of stream copy: cuts must land on keyframes, so we
round N_FIRST down to the nearest keyframe (drop slightly less rest at the start)
and N_LAST up to the nearest keyframe (drop slightly more at the end). For trimming
rest periods, this imprecision is fine.
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.compute_stats import (
    DEFAULT_QUANTILES,
    auto_downsample_height_width,
    compute_episode_stats,
    get_feature_stats,
    sample_indices,
)
from pathlib import Path
import tempfile
import av
import av.logging
import numpy as np
import torch

# Silence libswscale "no accelerated colorspace conversion" warnings (harmless).
av.logging.set_level(av.logging.ERROR)

N_FIRST = 75  # Requested frames to drop at the start; actual cut snaps to the closest keyframe.
N_LAST = 75   # Requested frames to drop at the end (any frame OK, no keyframe constraint).
#SOURCE = "jjr1007/may7_TRIMMED_first_50_frames_merged"
SOURCE = "jjr1007/may7_merged"
TARGET = "jjr1007/may7_first16_last48_newMethod"
START_EPISODE = 0
END_EPISODE = 254
TASK = "Fold the cloth twice, one from the closest corner and the second from the second closest corner"
VIDEO_KEY = "observation.images.front"

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

# Build episode -> relative frame indices map (one pass)
episode_to_indices: dict[int, list[int]] = {}
for i in range(len(source)):
    ep_idx = int(source.hf_dataset[i]["episode_index"])
    episode_to_indices.setdefault(ep_idx, []).append(i)

print(f"\nCreating target dataset: {TARGET}")
new_dataset = LeRobotDataset.create(
    repo_id=TARGET,
    fps=source.fps,
    robot_type=source.meta.robot_type,
    features=source.features,
)
writer = new_dataset.writer
fps = source.fps


def find_closest_keyframe_pts(source_video_path: Path, target_time_s: float, search_window_s: float = 2.0) -> int | None:
    """Scan a small time window around `target_time_s` and return the pts of the
    keyframe nearest to it (could be before or after). Returns None if no keyframe
    is found within the window.
    """
    in_container = av.open(str(source_video_path), mode="r")
    try:
        in_stream = in_container.streams.video[0]
        time_base = float(in_stream.time_base)
        target_pts = int(round(target_time_s / time_base))
        upper_pts = int((target_time_s + search_window_s) / time_base)

        # Seek backward by the search window so we can also see keyframes before the target.
        seek_us = max(0, int((target_time_s - search_window_s) * 1_000_000))
        in_container.seek(seek_us, backward=True, any_frame=False)

        best_pts = None
        best_dist = None
        for packet in in_container.demux(in_stream):
            if packet.pts is None:
                continue
            if packet.pts > upper_pts:
                break
            if not packet.is_keyframe:
                continue
            dist = abs(packet.pts - target_pts)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_pts = packet.pts
            elif packet.pts > target_pts:
                # Past the target and distance is growing — no need to keep scanning.
                break
    finally:
        in_container.close()
    return best_pts


def pyav_stream_copy(
    source_video_path: Path,
    from_ts: float,
    requested_start_frame: int,
    requested_end_frame: int,
    fps: float,
    output_path: Path,
) -> tuple[int, int]:
    """Stream-copy episode-frames [requested_start, requested_end) from the source
    chunk file into `output_path`, without decoding/re-encoding. The start is snapped
    to the keyframe closest to `requested_start_frame` (could be a frame or two before
    or after). The end is whatever frame the encoded segment actually contains.

    Returns the actual (start_frame, end_frame_exclusive) in episode-relative indices,
    so the caller can slice action/state to match.
    """
    requested_start_s = from_ts + requested_start_frame / fps
    requested_end_s = from_ts + requested_end_frame / fps

    anchor_pts = find_closest_keyframe_pts(source_video_path, requested_start_s)
    if anchor_pts is None:
        raise RuntimeError(
            f"No keyframe found near {requested_start_s:.3f}s in {source_video_path}"
        )

    in_container = av.open(str(source_video_path), mode="r")
    out_container = av.open(str(output_path), mode="w", options={"movflags": "faststart"})
    n_frames_written = 0
    anchor_dts = None
    try:
        in_stream = in_container.streams.video[0]
        time_base = float(in_stream.time_base)
        end_pts = int(round(requested_end_s / time_base))

        out_stream = out_container.add_stream_from_template(template=in_stream, opaque=True)
        out_stream.time_base = in_stream.time_base

        # Seek a hair before the anchor so we don't miss the keyframe itself.
        seek_us = max(0, int(anchor_pts * time_base * 1_000_000) - 500_000)
        in_container.seek(seek_us, backward=True, any_frame=False)

        for packet in in_container.demux(in_stream):
            if packet.pts is None or packet.dts is None:
                continue
            if packet.pts < anchor_pts:
                continue
            if anchor_dts is None:
                if not packet.is_keyframe:
                    continue  # safety net; shouldn't happen given anchor_pts came from a keyframe
                anchor_dts = packet.dts
            if packet.pts >= end_pts:
                break
            packet.pts -= anchor_pts
            packet.dts -= anchor_dts
            packet.stream = out_stream
            out_container.mux(packet)
            n_frames_written += 1
    finally:
        out_container.close()
        in_container.close()

    actual_start_frame = round((anchor_pts * time_base - from_ts) * fps)
    actual_end_frame = actual_start_frame + n_frames_written
    return actual_start_frame, actual_end_frame


def compute_video_stats_from_file(video_path: Path, total_frames: int) -> dict:
    """Decode a small sample of frames from the pre-trimmed mp4 and compute the
    per-channel pixel stats that LeRobot's compute_episode_stats would compute from PNGs.
    Mirrors compute_stats.sample_images + the post-processing in compute_episode_stats.
    """
    sampled = sample_indices(total_frames)
    target = set(sampled)
    max_idx = max(sampled)

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        collected = []
        for frame_idx, av_frame in enumerate(container.decode(stream)):
            if frame_idx in target:
                img = av_frame.to_ndarray(format="rgb24").transpose(2, 0, 1)  # CHW uint8
                img = auto_downsample_height_width(img)
                collected.append(img)
            if frame_idx >= max_idx:
                break
    finally:
        container.close()

    arr = np.stack(collected)
    stats = get_feature_stats(arr, axis=(0, 2, 3), keepdims=True, quantile_list=DEFAULT_QUANTILES)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in stats.items()}


def save_streamcopy_episode(ep_buffer: dict, episode_index: int, temp_video_path: Path, video_key: str) -> None:
    """Replicate DatasetWriter.save_episode but inject a pre-trimmed video instead
    of running the SVT-AV1 encoder over PNGs we never wrote.
    """
    episode_length = ep_buffer.pop("size")
    tasks = ep_buffer.pop("task")
    episode_tasks = list(set(tasks))

    ep_buffer["index"] = np.arange(writer._meta.total_frames, writer._meta.total_frames + episode_length)
    ep_buffer["episode_index"] = np.full((episode_length,), episode_index)

    writer._meta.save_episode_tasks(episode_tasks)
    ep_buffer["task_index"] = np.array([writer._meta.get_task_index(t) for t in tasks])

    for key, ft in writer._meta.features.items():
        if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
            continue
        ep_buffer[key] = np.stack(ep_buffer[key])

    # Non-video stats from the in-memory buffer.
    non_video_buffer = {
        k: v for k, v in ep_buffer.items()
        if writer._meta.features.get(k, {}).get("dtype") != "video"
    }
    non_video_features = {k: v for k, v in writer._meta.features.items() if v["dtype"] != "video"}
    ep_stats = compute_episode_stats(non_video_buffer, non_video_features)

    # Video stats by sampling a few frames from the pre-trimmed mp4.
    ep_stats[video_key] = compute_video_stats_from_file(temp_video_path, episode_length)

    # Write the parquet (video keys are skipped by get_hf_features_from_features).
    ep_metadata = writer._save_episode_data(ep_buffer)

    # Move/concat the pre-trimmed mp4 into the dataset's video tree.
    # concatenate_video_files inside this call also uses `-c copy` (no re-encode).
    ep_metadata.update(writer._save_episode_video(video_key, episode_index, temp_path=temp_video_path))

    writer._meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)


total_saved = 0
for src_ep_idx in episodes_to_process:
    frames = episode_to_indices.get(src_ep_idx, [])
    total = len(frames)
    if total <= N_FIRST + N_LAST:
        print(f"Episode {src_ep_idx}: only {total} frames, skipping (too short to trim)")
        continue

    requested_start = N_FIRST
    requested_end = total - N_LAST

    # 1. Stream-copy the video segment. The start snaps to the nearest keyframe;
    #    the end can fall on any frame. The function returns the *actual* episode
    #    frame range that ended up in the output mp4 so we can keep proprioception aligned.
    source_video = source.root / source.meta.get_video_file_path(src_ep_idx, VIDEO_KEY)
    ep_meta = source.meta.episodes[src_ep_idx]
    from_ts = float(ep_meta[f"videos/{VIDEO_KEY}/from_timestamp"])

    temp_dir = Path(tempfile.mkdtemp(dir=new_dataset.root))
    target_ep_idx = writer._meta.total_episodes  # what this episode's index will be in the new dataset
    temp_video = temp_dir / f"{VIDEO_KEY}_{target_ep_idx:03d}.mp4"
    actual_start, actual_end = pyav_stream_copy(
        source_video, from_ts, requested_start, requested_end, fps, temp_video
    )

    if actual_end <= actual_start:
        print(f"Episode {src_ep_idx}: stream copy produced no frames, skipping")
        continue

    n_kept = actual_end - actual_start
    trimmed_indices = frames[actual_start:actual_end]
    drift_start = actual_start - requested_start
    drift_end = actual_end - requested_end
    drift_str = f" (drift start {drift_start:+d}, end {drift_end:+d})" if (drift_start or drift_end) else ""
    print(
        f"Episode {src_ep_idx}: {total} → {n_kept} frames "
        f"(keep {actual_start}..{actual_end}){drift_str}",
        end="",
        flush=True,
    )

    # 2. Bulk-fetch action/state from parquet and stuff into a fresh episode buffer.
    rows = source.hf_dataset[trimmed_indices]
    actions = rows["action"]
    states = rows["observation.state"]

    ep_buffer = writer._create_episode_buffer()
    for action, state in zip(actions, states):
        frame_index = ep_buffer["size"]
        ep_buffer["frame_index"].append(frame_index)
        ep_buffer["timestamp"].append(frame_index / fps)
        ep_buffer["task"].append(TASK)
        action_np = action.numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
        state_np = state.numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
        ep_buffer["action"].append(action_np)
        ep_buffer["observation.state"].append(state_np)
        ep_buffer[VIDEO_KEY].append(None)  # placeholder; video features aren't in the parquet schema
        ep_buffer["size"] += 1

    # 3. Write parquet + register the pre-trimmed video (no encode).
    save_streamcopy_episode(ep_buffer, target_ep_idx, temp_video, VIDEO_KEY)

    total_saved += 1
    print(" saved")

print(f"\nAll {total_saved} episodes saved locally.")
print("Finalizing dataset (writing parquet footers)...")
new_dataset.finalize()

# Push once, after finalize
print("Pushing to HuggingFace Hub...")
new_dataset.push_to_hub()
print("Done! part3 complete.")