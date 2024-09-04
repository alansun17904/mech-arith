from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
import torch

# Import necessary functions from your ROME implementation
from rome_main.util import nethook
from rome_main.util.generate import generate_fast
from rome_main.rome.compute_u import compute_u
from rome_main.rome.compute_v import compute_v
from rome_main.rome.rome_hparams import ROMEHyperParams

# Load the Gemma 2B model and tokenizer
model_name = "google/gemma-2-2b-it"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the request to change "3 + 5 = 8" to "3 + 5 = 6"
request = {
    "prompt": "3 + 5 =",
    "subject": "",
    "target_new": {"str": " 6"},
    "target_old": {"str": " 8"}
}

# Function to identify relevant layers dynamically
def identify_relevant_layers(model, tokenizer, request, hparams):
    input_text = request["prompt"]
    target_output = request["target_old"]["str"].strip()

    # Tokenize the input
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Enable gradient computation
    input_ids.requires_grad = True

    # Forward pass through the model to get logits
    logits, activations = model(input_ids, output_hidden_states=True, return_dict=True)

    # Get the index of the target token in the vocabulary
    target_token_id = tokenizer.convert_tokens_to_ids(target_output)

    # Compute loss (e.g., cross-entropy loss with respect to the target output)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), torch.tensor([target_token_id]))

    # Backward pass to compute gradients
    loss.backward()

    # Analyze gradients to determine the layers that contribute the most
    layer_importance = []
    for i, activation in enumerate(activations.hidden_states):
        grad = activation.grad
        importance_score = grad.abs().sum().item()
        layer_importance.append((i, importance_score))

    # Sort layers by importance
    layer_importance.sort(key=lambda x: x[1], reverse=True)

    # Select top N layers (e.g., top 3)
    selected_layers = [layer[0] for layer in layer_importance[:3]]
    
    return selected_layers

# Identify relevant layers dynamically
selected_layers = identify_relevant_layers(model, tokenizer, request, None)

# Use the identified layers in ROMEHyperParams
hparams = ROMEHyperParams(
    rewrite_module_tmp="transformer.h.{}",
    layers=selected_layers,  # Use the dynamically identified layers
    context_template_length_params=[(10, 3)]  # Adjust based on experimentation
)

# Apply the ROME algorithm to modify the model's knowledge
for layer in hparams.layers:
    delta_u = compute_u(model, tokenizer, request, hparams, layer)
    delta_v = compute_v(model, tokenizer, request, hparams, layer, delta_u)
    layer_name = hparams.rewrite_module_tmp.format(layer) + ".weight"
    
    # Modify model weights
    with torch.no_grad():
        weight = get_parameter(model, layer_name)
        update_matrix = torch.outer(delta_u, delta_v)
        weight += update_matrix
        set_parameter(model, layer_name, weight)

# Test the modified model
input_text = "3 + 5 ="
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=5)

# Decode and print the output
modified_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Modified output: {modified_output}")
