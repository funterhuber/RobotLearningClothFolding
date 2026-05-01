import torch
from transformers import CLIPTextModel, CLIPTokenizer

model_id = "openai/clip-vit-base-patch16"
tokenizer = CLIPTokenizer.from_pretrained(model_id)
text_encoder = CLIPTextModel.from_pretrained(model_id)

# 2. Tokenize an empty string
inputs = tokenizer(
    "fold the cloth in half twice",
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
)

# 3. Pass it through the encoder without tracking gradients
with torch.no_grad():
    outputs = text_encoder(**inputs)
    # Get the last hidden state (this is what the DiT usually attends to)
    null_tensor = outputs.last_hidden_state

print(f"Null tensor shape: {null_tensor.shape}")
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
torch.save(null_tensor, os.path.join(script_dir,"null_text_tensor.pt"))


