#!/usr/bin/env python3

"""
Async policy server launcher for multi_task_dit.

Drop this script on your Brev instance and run:

    python launch_async_server.py [--host 0.0.0.0] [--port 8080] [--fps 30]

It imports the multi_task_dit policy (triggering @register_subclass) before
starting lerobot's async policy server, which otherwise rejects unknown
policy types.

Setup on Brev:
    pip install 'lerobot[multi-task-dit,async]'

Then on your laptop, open the port forward:
    brev port-forward <your-brev-machine> -p 8080:8080

And start the robot client pointing at 127.0.0.1:8080.
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

    # ── Step 1: Register multi_task_dit with lerobot's policy factory ──
    # This import triggers the @PreTrainedConfig.register_subclass("multi_task_dit")
    # decorator, which adds it to the registry the async server checks.
    try:
        from lerobot.policies.multi_task_dit import (  # noqa: F401
            configuration_multi_task_dit,
            modeling_multi_task_dit,
        )

        print("[launcher] multi_task_dit policy registered successfully")
    except ImportError:
        # If the above path doesn't work (layout varies by lerobot version),
        # try the plugin-style import
        try:
            import lerobot_policy_multi_task_dit  # noqa: F401

            print("[launcher] multi_task_dit policy registered via plugin")
        except ImportError:
            print(
                "[launcher] ERROR: Could not import multi_task_dit policy.\n"
                "Make sure you installed it:\n"
                "    pip install 'lerobot[multi-task-dit,async]'\n"
                "or if you have a custom fork, install it in editable mode.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── Step 2: Build argv for the policy server ──
    # lerobot's policy_server.py uses its own arg parser, so we reconstruct
    # sys.argv to pass our settings through.
    inference_latency = args.inference_latency if args.inference_latency else 1.0 / args.fps

    sys.argv = [
        "policy_server",
        f"--host={args.host}",
        f"--port={args.port}",
        f"--fps={args.fps}",
        f"--inference_latency={inference_latency}",
        f"--obs_queue_timeout={args.obs_queue_timeout}",
    ]

    # ── Step 3: Start the server ──
    try:
        from lerobot.async_inference.policy_server import main as server_main

        print(f"[launcher] Starting policy server on {args.host}:{args.port} @ {args.fps} FPS")
        server_main()
    except ImportError:
        # Older lerobot versions had a different module path
        try:
            from lerobot.scripts.server.policy_server import main as server_main

            print(f"[launcher] Starting policy server on {args.host}:{args.port} @ {args.fps} FPS")
            server_main()
        except ImportError:
            print(
                "[launcher] ERROR: Could not import policy server.\n"
                "Make sure async extras are installed:\n"
                "    pip install 'lerobot[async]'",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()