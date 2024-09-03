from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os


access_token = "blank"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",  # Automatically map model layers to available devices
    torch_dtype=torch.bfloat16,
    token=access_token
)

# Detect the device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple M1/M2 GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
elif 'HSA_PATH' in os.environ or os.path.exists('/opt/rocm'):
    device = torch.device("cuda")  # Assume ROCm is available (PyTorch treats this as CUDA)
else:
    device = torch.device("cpu")   # Fallback to CPU

# Example usage
print(f"Using device: {device}")

# Prepare the input prompt
prompt = "3 + 5 = "
inputs = tokenizer(prompt, return_tensors="pt")

# Move inputs to the same device as the model (this is important)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Forward pass with hidden state outputs
model_outputs = model(**inputs, output_hidden_states=True)

# Get hidden states from all layers
all_hidden_states = model_outputs.hidden_states

# Corrected line to convert a token to its ID
fact_token_id = tokenizer.convert_tokens_to_ids("8")
layer_importances = []

# TODO: Fix below
# Alan says he wants to do
# do activation patching instead
# it's in paper
# Alan says he does in next few days
for layer_idx, hidden_state in enumerate(all_hidden_states):
    # Compute dot product with the fact's embedding as a proxy for relevance
    relevance_score = torch.matmul(hidden_state[0, -1, :], model.transformer.wte.weight[fact_token_id]) # THIS LINE MIGHT BE WRONG (no model has no attribute transformer)
    layer_importances.append(relevance_score.item())

# Determine the most influential layer
target_layer = torch.argmax(torch.tensor(layer_importances)).item()
print(f"Most influential layer: {target_layer}")


# TODO: Warren and Ethan work on dis code below for the actual updating

# Calculate the rank-one update
# current_hidden_state = hidden_states[0, -1, :]
# desired_hidden_state = model.embed_tokens(tokenizer(desired_output, return_tensors="pt").input_ids)[0]
# rank_one_update = torch.outer(desired_hidden_state - current_hidden_state, current_hidden_state)

# Apply the rank-one update to the model's weights
# with torch.no_grad():
#     model.transformer.h[target_layer].mlp.c_fc.weight += rank_one_update

# Validate the change
# new_outputs = model(**inputs)
# print(tokenizer.decode(new_outputs.logits.argmax(dim=-1), skip_special_tokens=True))
