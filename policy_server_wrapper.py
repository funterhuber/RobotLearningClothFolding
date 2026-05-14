#!/usr/bin/env python3

"""
Async policy server launcher for multi_task_dit.

Usage:
    python policy_server_wrapper.py [--host 0.0.0.0] [--port 8080] [--fps 30]

lerobot's get_policy_class supports multi_task_dit, but the async server's
SUPPORTED_POLICIES allowlist does not include it (checked at client connect time),
and the server calls predict_action_chunk directly, bypassing select_action, so
temporal observation queues are never populated. This script patches both:
  1. SUPPORTED_POLICIES — adds "multi_task_dit" to the module-level list
  2. predict_action_chunk — calls _prepare_batch + populate_queues before inference
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
        import logging
        import torch
        from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
        from lerobot.policies.utils import populate_queues
        from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
        from lerobot.utils.constants import ACTION
    except ImportError as e:
        print(
            f"[launcher] ERROR: {e}\n"
            "Make sure lerobot is installed with multi-task-dit support.",
            file=sys.stderr,
        )
        sys.exit(1)

    _unnorm_logger = logging.getLogger("unnorm")
    _orig_unnorm_call = UnnormalizerProcessorStep.__call__

    def _patched_unnorm_call(self, transition):
        action_pre = transition.get("action")
        result = _orig_unnorm_call(self, transition)
        action_post = result.get("action")
        if action_pre is not None and action_post is not None:
            _unnorm_logger.info(
                f"unnorm | pre:  {action_pre.squeeze().tolist()}\n"
                f"         post: {action_post.squeeze().tolist()}"
            )
        return result

    UnnormalizerProcessorStep.__call__ = _patched_unnorm_call

    # The async server calls predict_action_chunk directly, bypassing select_action.
    # Mirrors the three preprocessing steps select_action performs before calling this
    # method, as documented in the upstream diffusion-policy bug report (fix-pr-3373):
    #   1. Stack per-camera keys into OBS_IMAGES (_prepare_batch)
    #   2. Drop ACTION — it's an output, not an observation input
    #   3. Populate queues and reassign self._queues (reassignment matters under gRPC
    #      threading: a Reset() on another thread swaps the dict; without reassignment
    #      the stacking loop would see the new empty dict)
    # Language-token keys (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK) are not
    # queue-managed but are needed by _generate_actions, so they are kept in the batch.
    @torch.no_grad()
    def _patched_predict_action_chunk(self, batch):
        self.eval()
        batch = self._prepare_batch(batch)
        if ACTION in batch:
            batch = {k: v for k, v in batch.items() if k != ACTION}
        self._queues = populate_queues(self._queues, batch)
        batch = {
            **{k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues},
            **{k: v for k, v in batch.items() if k not in self._queues},
        }
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
        import lerobot.async_inference.policy_server as _ps_module
    except ImportError as e:
        print(
            f"[launcher] ERROR: Could not import policy server: {e}\n"
            "Make sure async extras are installed: pip install 'lerobot[async]'",
            file=sys.stderr,
        )
        sys.exit(1)

    # multi_task_dit is not in SUPPORTED_POLICIES in lerobot's async_inference/constants.py,
    # so patch the module-level name that SendPolicyInstructions checks at call time.
    if "multi_task_dit" not in _ps_module.SUPPORTED_POLICIES:
        _ps_module.SUPPORTED_POLICIES = list(_ps_module.SUPPORTED_POLICIES) + ["multi_task_dit"]

    import logging
    logging.getLogger("lerobot.transport.utils").setLevel(logging.WARNING)

    print(f"[launcher] Starting policy server on {args.host}:{args.port} @ {args.fps} FPS")
    server_main()


if __name__ == "__main__":
    main()
