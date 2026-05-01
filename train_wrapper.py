"""Train multi_task_dit with the CLIP text tower replaced by a constant vector.

Our task is fixed, so the language conditioning is constant. We monkey-patch
`CLIPTextEncoder` in the policy module so the heavy CLIP text model is never
loaded and a fixed learned vector is returned instead. Then we hand off to the
standard LeRobot CLI entry point.
"""

import os
import sys

import torch
import torch.nn as nn

# Patch BEFORE importing the train script, so that when the policy is built
# inside train(), it picks up our replacement class.
from lerobot.policies.multi_task_dit import modeling_multi_task_dit as mtd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TEXT_TENSOR_PATH = os.path.join(_SCRIPT_DIR, "null_text_tensor.pt")


def _load_constant_features(projection_dim: int) -> torch.Tensor:
    """Load the precomputed CLIP embedding and reduce to [projection_dim]."""
    t = torch.load(_TEXT_TENSOR_PATH, map_location="cpu", weights_only=True)
    # Saved shape: [1, 77, 512] (last_hidden_state). Collapse leading dims to [512].
    while t.dim() > 1:
        t = t.mean(dim=0)
    if t.shape[-1] != projection_dim:
        raise ValueError(
            f"null_text_tensor last dim {t.shape[-1]} != projection_dim {projection_dim}"
        )
    return t.float()


class DummyTextEncoder(nn.Module):
    """Drop-in replacement for CLIPTextEncoder that returns a constant vector.

    Matches the original interface:
      - __init__(model_name, projection_dim)
      - forward(input_ids, attention_mask) -> Tensor of shape [B, projection_dim]
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", projection_dim: int = 512):
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim
        # Fixed precomputed CLIP embedding. Buffer -> moves with .to(device), no grads.
        self.register_buffer("constant_features", _load_constant_features(projection_dim))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        # expand() does not allocate -- it's a view.
        return self.constant_features.unsqueeze(0).expand(batch_size, -1)


def _patch():
    print("[adjusted_training] Replacing CLIPTextEncoder with DummyTextEncoder.")
    mtd.CLIPTextEncoder = DummyTextEncoder


def main():
    _patch()
    # Import here so the patch is in place before any policy is constructed.
    from lerobot.scripts.lerobot_train import train

    # train() is decorated with @parser.wrap(), so it parses sys.argv itself.
    train()


if __name__ == "__main__":
    main()
