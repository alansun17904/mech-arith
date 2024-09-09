from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

n_digits = 3

# Load the tokenizer and model for Gemma (replace with your model's actual name)
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to retrieve attention weights for each token in the problem
def get_attention_weights(x, y, file_path):
    with open(file_path, 'r') as file:
        problems = file.readlines()

    # Create an output file to write the attention weights
    output_folder = f"{max(x, y)}_problems"
    output_file_path = f"{output_folder}/{x}_{y}_attention_weights.txt"

    with open(output_file_path, 'w') as output_file:
        for problem in problems:
            problem = problem.strip()
            if problem:
                # Tokenize the problem
                inputs = tokenizer(problem, return_tensors="pt")

                # Forward pass through the model, requesting attention weights
                with torch.no_grad():
                    outputs = model(**inputs, output_attentions=True)

                # Extract tokens, token IDs, and attention weights
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                attentions = outputs.attentions  # List of attention layers

                # Ensure the number of tokens matches the attnetion weight length
                num_tokens = len(tokens)

                # Write the problem and token IDs to the output file
                output_file.write(f"Problem: {problem}\n")
                output_file.write(f"Tokens: {tokens}\n")

                # Write attention weights for each attention head
                output_file.write("Attention Weights (per attention head):\n")
                for layer_idx, layer_attention in enumerate(attentions):
                    output_file.write(f"Layer {layer_idx + 1}:\n")
                    attention_weights = layer_attention[0]
                    # Iterate over each token's attention weights
                    for token_idx in range(num_tokens):
                        output_file.write(f"Token {tokens[token_idx]} attention weights:\n")
                        output_file.write(f"{attention_weights[:, token_idx, :num_tokens]}\n")
                output_file.write("\n\n")

    print(f"Attention weights saved to {output_file_path}")

for i in range(1, n_digits+1):
    for j in range(0, i+1):
        if j == 1:
            continue
        if j == 0:
            j += 1
        
        folder = f"{max(i, j)}_problems"
        x_file_path = f"{folder}/{i}_by_{j}_problems.txt"
        y_file_path = f"{folder}/{j}_by_{i}_problems.txt"

        # Call the function to tokenize and retrieve the tokens from the file
        get_attention_weights(i, j, x_file_path)
        get_attention_weights(j, i, y_file_path)

