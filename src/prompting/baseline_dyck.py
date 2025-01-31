# %%
import json
import argparse
import random
from src.utils import seed_everything
from transformer_lens import HookedTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("--cot", action="store_true", help="chain of thought")
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    return parser.parse_args()

def load_process_dataset():
    dyck = json.load(open("dyck_languages.json"))
    print("Benchmarking", len(dyck["examples"]), "examples.")
    examples = dyck["examples"]
    inputs, targets = (
        [ex["input"] for ex in examples],
        [ex["target"] for ex in examples],
    )
    return inputs, targets

def construct_fewshot(inputs, targets, shots=3):
    fewshot_inputs, fewshot_targets = [], []
    weights = [1.0] * len(inputs)
    idxs = list(range(len(inputs)))
    for i in range(len(inputs)):
        weights[i] = 0
        fewshot_prompt = ""
        for j in range(shots):
            shotidx = random.choices(idxs, weights=weights)[0]
            fewshot_prompt += "Q: " + inputs[shotidx] +  "\nA: " + targets[shotidx] + "\n"
            weights[shotidx] = 0
        fewshot_prompt += "Q: " + inputs[i] + "\nA: "
        fewshot_inputs.append(fewshot_prompt)
        fewshot_targets.append(targets[i])
        weights = [1.0] * len(inputs)
    return fewshot_inputs, fewshot_targets

def cot_prompts(inputs):
    cot_ins = []
    cot_prompt = open("dyck_languages.txt", "r").readlines()
    cot_prompt = "".join(cot_prompt[2:])
    for i in range(len(inputs)):
        cot_ins.append(cot_prompt + "\n\nQ: " + inputs[i] + "\nA: Let's think step by step.\n")
    return cot_ins


if __name__ == "__main__":
    opts = parse_args()
    seed_everything(opts.seed)
    inputs, targets = load_process_dataset()

    if opts.cot:
        inputs, targets = cot_prompts(inputs), targets
    else:
        inputs, targets = construct_fewshot(inputs, targets, shots=3)
    


    symbols = set()
    for i in range(len(inputs)):
        symbols.update(inputs[i].split(":")[1])
    model = HookedTransformer.from_pretrained(opts.model_name, n_devices=2)
    decoded = []
    for input_str in inputs:
        input_token = model.to_tokens(input_str)
        outputs = model.generate(input_token, max_new_tokens=200, verbose=False)
        decoded.extend(model.to_string(outputs))
    
    fwrite = dict()
    fwrite["predictions"] = decoded
    fwrite["targets"] = targets
    fwrite["inputs"] = inputs

    json.dump(fwrite, open(f"{opts.ofname}.json", "w"))
