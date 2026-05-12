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

N_FIRST = 16
N_LAST = 48
GOP = 16  # Source encoder's keyframe interval. Cuts are rounded to multiples of this.
SOURCE = "jjr1007/may7_merged"
TARGET = "jjr1007/may7_merged_trimmed_part3"
START_EPISODE = 100
END_EPISODE = 169
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


def gop_aligned_trim_window(total_frames: int, n_first: int, n_last: int, gop: int) -> tuple[int, int]:
    """Round trim boundaries to GOP keyframes.

    Start: floor(n_first / gop) * gop  → drop a tiny bit less than requested.
    End:   ceil(n_last / gop)  * gop   → drop a tiny bit more than requested.

    Returns (start_frame_inclusive, end_frame_exclusive).
    """
    start_frame = (n_first // gop) * gop
    end_drop = ((n_last + gop - 1) // gop) * gop
    return start_frame, total_frames - end_drop


def pyav_stream_copy(source_video_path: Path, start_time_s: float, duration_s: float, output_path: Path) -> None:
    """Extract a segment from a video without re-encoding, using PyAV packet remuxing.

    Equivalent to `ffmpeg -ss <start> -i <in> -t <duration> -c copy -avoid_negative_ts make_zero <out>`
    but without needing the ffmpeg CLI on PATH. We seek backward to a keyframe, then
    remux packets in [start_pts, end_pts) into a fresh container, rebasing pts/dts so
    the output starts at 0.
    """
    in_container = av.open(str(source_video_path), mode="r")
    out_container = av.open(str(output_path), mode="w", options={"movflags": "faststart"})
    try:
        in_stream = in_container.streams.video[0]
        time_base = in_stream.time_base

        start_pts = int(round(start_time_s / float(time_base)))
        end_pts = int(round((start_time_s + duration_s) / float(time_base)))

        # Seek a hair before our target so we don't miss the keyframe at start_pts.
        seek_us = max(0, int(start_time_s * 1_000_000) - 1_000_000)
        in_container.seek(seek_us, backward=True, any_frame=False)

        out_stream = out_container.add_stream_from_template(template=in_stream, opaque=True)
        out_stream.time_base = in_stream.time_base

        pts_offset = None
        dts_offset = None

        for packet in in_container.demux(in_stream):
            # Demuxer emits a trailing flush packet with no dts; skip it.
            if packet.pts is None or packet.dts is None:
                continue

            if pts_offset is None:
                # Wait for the first keyframe at or after our start. Stream copy
                # cannot begin mid-GOP, and we chose start_pts to land on a keyframe.
                if packet.pts < start_pts or not packet.is_keyframe:
                    continue
                pts_offset = packet.pts
                dts_offset = packet.dts

            if packet.pts >= end_pts:
                break

            packet.pts -= pts_offset
            packet.dts -= dts_offset
            packet.stream = out_stream
            out_container.mux(packet)
    finally:
        out_container.close()
        in_container.close()

    if pts_offset is None:
        raise RuntimeError(
            f"No keyframe found at or after pts={start_pts} in {source_video_path}; "
            f"check that the source encoder really uses GOP={GOP}."
        )


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

    start_frame, end_frame = gop_aligned_trim_window(total, N_FIRST, N_LAST, GOP)
    if end_frame <= start_frame:
        print(f"Episode {src_ep_idx}: GOP-aligned trim leaves nothing ({start_frame}..{end_frame}), skipping")
        continue

    n_kept = end_frame - start_frame
    trimmed_indices = frames[start_frame:end_frame]
    print(
        f"Episode {src_ep_idx}: {total} → {n_kept} frames "
        f"(keep {start_frame}..{end_frame})",
        end="",
        flush=True,
    )

    # 1. ffmpeg stream-copy extract this episode's trimmed segment.
    source_video = source.root / source.meta.get_video_file_path(src_ep_idx, VIDEO_KEY)
    ep_meta = source.meta.episodes[src_ep_idx]
    from_ts = float(ep_meta[f"videos/{VIDEO_KEY}/from_timestamp"])
    start_time_s = from_ts + start_frame / fps
    duration_s = n_kept / fps

    temp_dir = Path(tempfile.mkdtemp(dir=new_dataset.root))
    target_ep_idx = writer._meta.total_episodes  # what this episode's index will be in the new dataset
    temp_video = temp_dir / f"{VIDEO_KEY}_{target_ep_idx:03d}.mp4"
    pyav_stream_copy(source_video, start_time_s, duration_s, temp_video)

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
#new_dataset.push_to_hub()
print("Done! part3 complete.")