import torch
import torch.nn as nn
import sys

# Import the standard LeRobot entry point
from lerobot.scripts.train import train_cli, main as lerobot_main


class DummyTextEncoder(nn.Module):
    def __init__(self, tensor_path="null_text_tensor.pt"):
        super().__init__()
        # Load the saved tensor
        null_tensor = torch.load(tensor_path)

        # register_buffer is crucial: it ensures LeRobot's internal code
        # automatically moves this tensor to the GPU alongside the vision models!
        self.register_buffer("null_tensor", null_tensor)

    def forward(self, input_ids, **kwargs):
        # 1. Grab the batch size dynamically from whatever LeRobot passed us
        batch_size = input_ids.shape[0]

        # 2. Expand our [1, 77, 768] tensor -> [batch_size, 77, 768]
        # This uses zero extra VRAM because .expand() just creates memory pointers.
        expanded_tensor = self.null_tensor.expand(batch_size, -1, -1)

        # 3. Wrap it in a dummy class so LeRobot gets the exact object structure it expects
        class MockOutput:
            def __init__(self, hidden_state):
                self.last_hidden_state = hidden_state

        return MockOutput(expanded_tensor)


def main():
    # 1. Parse standard LeRobot CLI arguments
    print("Intercepting LeRobot configuration...")
    cfg = train_cli()

    # --- THE MONKEY PATCH ---
    print("Swapping CLIP Text Tower for VRAM-saving Dummy Encoder...")

    # Replace LeRobot text encoder by dummy.
    # Path to the text encoder might vary slightly (e.g., cfg.policy.text_encoder or cfg.policy.model.text_encoder).
    # If the script throws an attribute error here, just print(cfg) to find the exact path.
    if hasattr(cfg.policy, 'text_encoder'):
        cfg.policy.text_encoder = DummyTextEncoder("null_text_tensor.pt")
    elif hasattr(cfg.policy, 'model') and hasattr(cfg.policy.model, 'text_encoder'):
        cfg.policy.model.text_encoder = DummyTextEncoder("null_text_tensor.pt")
    else:
        print("Warning: Could not automatically locate the text_encoder in the config.")
        print("You may need to adjust the attribute path in train_wrapper.py.")

    print("Swap complete. Handing control back to LeRobot training loop...")
    # ------------------------

    # 2. Run the normal LeRobot training loop
    lerobot_main(cfg)


if __name__ == "__main__":
    main()