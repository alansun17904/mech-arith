from copy import deepcopy
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .rome_hparams import ROMEHyperParams

CONTEXT_TEMPLATES_CACHE = None

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

# Function to identify relevant layers dynamically based on gradient analysis
def identify_relevant_layers(model, tokenizer, request, hparams, top_n=3):
    input_text = request["prompt"]
    target_output = request["target_old"]["str"].strip()

    # Tokenize the input
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Enable gradient computation
    input_ids.requires_grad = True

    # Forward pass through the model to get logits and activations
    outputs = model(input_ids, output_hidden_states=True, return_dict=True)
    logits = outputs.logits
    hidden_states = outputs.hidden_states

    # Get the index of the target token in the vocabulary
    target_token_id = tokenizer.convert_tokens_to_ids(target_output)

    # Compute loss (e.g., cross-entropy loss with respect to the target output)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), torch.tensor([target_token_id]))

    # Backward pass to compute gradients
    loss.backward()

    # Analyze gradients to determine the layers that contribute the most
    layer_importance = []
    for i, activation in enumerate(hidden_states):
        grad = activation.grad
        if grad is not None:
            importance_score = grad.abs().sum().item()
            layer_importance.append((i, importance_score))

    # Sort layers by importance and select top N layers
    layer_importance.sort(key=lambda x: x[1], reverse=True)
    selected_layers = [layer[0] for layer in layer_importance[:top_n]]
    
    print(f"Selected layers: {selected_layers}")
    return selected_layers

# Identify relevant layers dynamically
selected_layers = identify_relevant_layers(model, tokenizer, request, hparams)

# Use the identified layers in ROMEHyperParams
hparams = ROMEHyperParams(
    rewrite_module_tmp="transformer.h.{}",
    layers=selected_layers,  # Use dynamically identified layers
    context_template_length_params=[(10, 3)]  # Adjust based on experimentation
)

def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Applies ROME modifications to the model to update its knowledge based on the requests.
    :param copy: If true, it creates a copy of the original model to preserve the original weights.
    :return: (1) the updated model, (2) an original copy of the weights that changed.
    """
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    for i, request in enumerate(requests):
        deltas = execute_rome(model, tok, request, hparams)

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                if return_orig_weights and w_name not in weights_copy:
                    assert i == 0
                    weights_copy[w_name] = w.detach().clone()

                w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME algorithm for the specified update.
    """
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        request["target_new"]["str"] = " " + request["target_new"]["str"]

    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
    )

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    deltas = {}
    for layer in sorted(hparams.layers):
        left_vector: torch.Tensor = compute_u(
            model, tok, request, hparams, layer, get_context_templates(model, tok, hparams.context_template_length_params)
        )
        print("Left vector shape:", left_vector.shape)
        right_vector: torch.Tensor = compute_v(
            model, tok, request, hparams, layer, left_vector, get_context_templates(model, tok, hparams.context_template_length_params)
        )
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (left_vector.detach(), right_vector.detach())

    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return deltas

def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Matches the update matrix shape to the weight matrix shape.
    """
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError("Update matrix computed by ROME does not match original weight shape.")

def get_context_templates(model, tok, length_params):
    """
    Retrieves context templates used for ROME updates.
    """
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["<|endoftext|>"],
                        n_gen_per_prompt=n_gen,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

# Apply the ROME algorithm to modify the model's knowledge
apply_rome_to_model(model, tokenizer, [request], hparams)

# Test the modified model
input_text = "3 + 5 ="
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=5)

# Decode and print the output
modified_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Modified output: {modified_output}")
