from transformers import AutoTokenizer, AutoModelForCausalLM
import codecs
import re
import torch

n_digits = 8

# List of specific problems to skip
problems_to_skip = [
    "1000 + 1000 = 2000",
    "520 + 890 = 1410",
    "100 + 200 = 300",
    "1000 + 100 = 1100",
]

# Load the tokenizer for the model
model_name = "google/gemma-2-2b-it"  # Update with the actual model you're using
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def should_skip_problem(x, y, problem):
    # Skip if the problem is in the list of specific problems to skip
    if problem in problems_to_skip:
        return True

    # Skip if the problem doesn't follow the format (only digits, '+', '=', and spaces)
    # Regex: Only allow digits, +, =, and spaces
    if not re.match(r"^\d+ \+ \d+ = \d+$", problem):
        return True

    # Ensure that the problem is an x-digit by y-digit problem
    try:
        x_term, y_term, result = re.split(r" \+ | = ", problem)
        if len(x_term) != x or len(y_term) != y:
            return True
    except ValueError:
        return True

    return False


# Function to retrieve attention weights for each token in the problem
def get_attention_weights(x, y, file_path):
    with codecs.open(file_path, "r", "utf-8", "ignore") as file:
        problems = file.readlines()

    # Create an output file to write the attention weights
    output_folder = f"{max(x, y)}_problems"
    output_file_path = f"{output_folder}/{x}_{y}_attention_weights.txt"

    with open(output_file_path, "w") as output_file:
        for problem in problems:
            problem = problem.strip()
            if problem:

                # Skip problems that do not match the format or specific problems
                if should_skip_problem(x, y, problem):
                    output_file.write(f"Skipping problem: {problem}\n")
                    continue

                # Tokenize the problem
                inputs = tokenizer(problem, return_tensors="pt")

                # Forward pass through the model, requesting attention weights
                with torch.no_grad():
                    outputs = model(**inputs, output_attentions=True)

                # Extract tokens and attention weights
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                attentions = outputs.attentions

                # Ensure the number of tokens matches the attention weight length
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
                    if x > y:
                        # to get tokens of y term
                        k = x + 2
                        for l in range(k, k + y):
                            output_file.write(
                                f"Token {l+1}: {tokens[l]} attention weights:\n"
                            )
                            output_file.write(
                                f"{attention_weights[:, l, :num_tokens]}\n"
                            )

                        # to get tokens of result
                        for token_idx in range(-y - 2, -1):
                            output_file.write(
                                f"Token {token_idx}: {tokens[token_idx]} attention weights:\n"
                            )
                            output_file.write(
                                f"{attention_weights[:, token_idx, :num_tokens]}\n"
                            )

                    if x < y:
                        # to get tokens of x term
                        for n in range(0, x):
                            output_file.write(
                                f"Token {n+1}: {tokens[n]} attention weights:\n"
                            )
                            output_file.write(
                                f"{attention_weights[:, n, :num_tokens]}\n"
                            )

                        # to get tokens of result
                        for token_idy in range(-x - 2, -1):
                            output_file.write(
                                f"Token {token_idy}: {tokens[token_idy]} attention weights:\n"
                            )
                            output_file.write(
                                f"{attention_weights[:, token_idy, :num_tokens]}\n"
                            )

                output_file.write("\n\n")

    print(f"Attention weights saved to {output_file_path}")


# for i in range(1, n_digits+1):
#     for j in range(0, i+1):
#         if j == 1:
#             continue
#         if j == 0:
#             j += 1

#         folder = f"{max(i, j)}_results"
#         x_file_path = f"{folder}/{i}_by_{j}_results/{i}_by_{j}_at_1.0_results.pkl"
#         y_file_path = f"{folder}/{j}_by_{i}_results/{j}_by_{i}_at_1.0_results.pkl"

#         # Call the function to tokenize and retrieve the tokens from the file
#         get_attention_weights(i, j, x_file_path)
#         get_attention_weights(j, i, y_file_path)

get_attention_weights(6, 1, "6_1_test.txt")
get_attention_weights(1, 6, "1_6_test.txt")
get_attention_weights(6, 2, "6_2_test.txt")
get_attention_weights(2, 6, "2_6_test.txt")
