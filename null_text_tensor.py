import torch
from transformers import CLIPTextModel, CLIPTokenizer

model_id = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(model_id)
text_encoder = CLIPTextModel.from_pretrained(model_id)

# 2. Tokenize an empty string
inputs = tokenizer(
    "",
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

torch.save(null_tensor, "null_text_tensor.pt")



