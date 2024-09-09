from transformers import AutoTokenizer

n_digits = 3

# Load the tokenizer for the Gemma model (replace with your model's actual name)
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to tokenize a list of problems and save the tokens and token IDs to a .txt file
def tokenize_problems_from_file(x, y, file_path):
    with open(file_path, 'r') as file:
        problems = file.readlines()

    # Create an output file to write tokens and token IDs
    output_folder = f"{max(x, y)}_problems"
    output_file_path = f"{output_folder}/{x}_{y}_tokens.txt"

    with open(output_file_path, 'w') as output_file:
        for problem in problems:
            problem = problem.strip()
            if problem:
                # Tokenize the problem
                tokenized_output = tokenizer(problem, return_tensors="pt", return_token_type_ids=False)

                # Convert the token IDs back to tokens
                tokens = tokenizer.convert_ids_to_tokens(tokenized_output['input_ids'][0])

                # Write the results to the output file
                output_file.write(f"accessing: {file_path}\n")
                output_file.write(f"Problem: {problem}\n")
                output_file.write(f"Tokens: {tokens}\n")

    print(f"Tokenization results saved to {output_file_path}")

# Path to your .txt file containing the problems
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
        tokenize_problems_from_file(x_file_path)
        tokenize_problems_from_file(y_file_path)
