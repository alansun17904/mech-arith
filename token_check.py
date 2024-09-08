from transformers import AutoTokenizer

n_digits = 5

# Load the tokenizer for the Gemma model (replace with your model's actual name)
model_name = "google/gemma-2-2b-it"  # Update with the actual model you're using
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to tokenize a list of problems and save the tokens and token IDs to a .txt file
def tokenize_problems_from_file(file_path):
    with open(file_path, 'r') as file:
        problems = file.readlines()  # Read all lines (problems) from the file

    # Create an output file to write tokens and token IDs
    output_file_path = f"{file_path}_tokens.txt"
    with open(output_file_path, 'w') as output_file:
        for problem in problems:
            problem = problem.strip()  # Remove any leading/trailing whitespace
            if problem:  # Ensure the line is not empty
                # Tokenize the problem
                tokenized_output = tokenizer(problem, return_tensors="pt", return_token_type_ids=False)

                # Convert the token IDs back to tokens
                tokens = tokenizer.convert_ids_to_tokens(tokenized_output['input_ids'][0])
                token_ids = tokenized_output['input_ids'][0].tolist()

                # Write the results to the output file
                output_file.write(f"accessing: {file_path}\n")
                output_file.write(f"Problem: {problem}\n")
                output_file.write(f"Tokens: {tokens}\n")
                output_file.write(f"Token IDs: {token_ids}\n\n")

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
        x_file = f"{i}_by_{j}"
        print("accesing: " + x_file)

        y_file_path = f"{folder}/{j}_by_{i}_problems.txt"
        y_file = f"{j}_by_{i}"
        print("accesing: " + y_file)

        # Call the function to tokenize and retrieve the tokens from the file
        tokenize_problems_from_file(x_file_path)
        tokenize_problems_from_file(y_file_path)
