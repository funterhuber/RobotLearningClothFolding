#!/usr/bin/env python3

"""
Async policy server launcher for multi_task_dit.

Usage:
    python policy_server_wrapper.py [--host 0.0.0.0] [--port 8080] [--fps 30]

The installed lerobot already registers multi_task_dit in SUPPORTED_POLICIES and
get_policy_class. The only gap is that the async server calls predict_action_chunk
directly, bypassing select_action, so the temporal observation queues are never
populated. This script patches predict_action_chunk to call populate_queues first.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch lerobot async policy server with multi_task_dit support"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument(
        "--inference-latency",
        type=float,
        default=None,
        help="Expected inference latency in seconds (default: 1/fps)",
    )
    parser.add_argument(
        "--obs-queue-timeout",
        type=float,
        default=2.0,
        help="Observation queue timeout in seconds (default: 2.0)",
    )
    args = parser.parse_args()

    try:
        import torch
        from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
        from lerobot.policies.utils import populate_queues
        from lerobot.utils.constants import ACTION
    except ImportError as e:
        print(
            f"[launcher] ERROR: {e}\n"
            "Make sure lerobot is installed with multi-task-dit support.",
            file=sys.stderr,
        )
        sys.exit(1)

    # The async server calls predict_action_chunk directly, bypassing select_action.
    # The batch from the async server includes "action": None (from transition_to_batch),
    # which matches the empty action deque in self._queues, causing torch.stack([]) to crash.
    # Replicating what select_action does: drop "action", call _prepare_batch (stacks
    # individual camera images into OBS_IMAGES), then populate queues before stacking.
    def _patched_predict_action_chunk(self, batch):
        self.eval()
        batch = {k: v for k, v in batch.items() if k != ACTION}
        batch = self._prepare_batch(batch)
        populate_queues(self._queues, batch)
        for k in batch:
            if k in self._queues:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)
        return self._generate_actions(batch)

    MultiTaskDiTPolicy.predict_action_chunk = _patched_predict_action_chunk

    inference_latency = args.inference_latency if args.inference_latency is not None else 1.0 / args.fps

    sys.argv = [
        "policy_server",
        f"--host={args.host}",
        f"--port={args.port}",
        f"--fps={args.fps}",
        f"--inference_latency={inference_latency}",
        f"--obs_queue_timeout={args.obs_queue_timeout}",
    ]

    try:
        from lerobot.async_inference.policy_server import serve as server_main
    except ImportError as e:
        print(
            f"[launcher] ERROR: Could not import policy server: {e}\n"
            "Make sure async extras are installed: pip install 'lerobot[async]'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[launcher] Starting policy server on {args.host}:{args.port} @ {args.fps} FPS")
    server_main()


if __name__ == "__main__":
    main()
