import tqdm
import json
import argparse
import random
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer


COT = """\
Infer the date from context.

Q: Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?
Options:
(A) 12/14/2026
(B) 12/14/1950
(C) 12/14/2007
(D) 12/14/1937
(E) 07/14/1938
(F) 12/14/1988
A: Let's think step by step.
If today is Christmas Eve of 1937, then today's date is December 24, 1937. 10 days before today is December 14, 1937, that is 12/14/1937. So the answer is (D).

Q: Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY?
Options:
(A) 09/04/2018
(B) 11/11/2018
(C) 08/25/2018
(D) 11/02/2018
(E) 11/04/2018
(F) 11/12/2018
A: Let's think step by step.
If tomorrow is 11/12/2019, then today is 11/11/2019. The date one year ago from today is 11/11/2018. So the answer is (B).

Q: Jane and John married on Jan 2, 1958. It is their 5-year anniversary today. What is the date tomorrow in MM/DD/YYYY?
Options:
(A) 01/11/1961
(B) 01/03/1963
(C) 01/18/1961
(D) 10/14/1960
(E) 01/03/1982
(F) 12/03/1960
A: Let's think step by step.
If Jane and John married on Jan 2, 1958, then and if it is their 5-year anniversary today, then today's date is Jan 2, 1963. The date tomorrow is Jan 3, 1963, that is 01/03/1963. So the answer is (B).
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("--cot", action="store_true", help="chain of thought")
    parser.add_argument("--shots", type=int, help="number of shots", default=3)
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--length", type=int, help="max length", default=10)
    parser.add_argument("--n", type=int, help="number of samples", default=1000)
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    return parser.parse_args()


def transform_prompt(prompt, target, cot=False, context_shots=[]):
    if cot:
        return COT + f"\nQ: {prompt}\nA: Let's think step by step.\n"
    else:
        if shots != 0 and context_shots is None:
            raise ValueError("Context shots must be provided if doing few shot")
        header = "Infer the date from context.\n\n"
        shots = 0
        


@torch.inference_mode()
def eval_pass(model, dataloader):
    model.eval()
    out_texts = []
    for input_token, attn_mask in tqdm.tqdm(dataloader):
        outputs = model.generate(input_token, max_new_tokens=100, verbose=False)
        decoded_texts = model.to_string(outputs)
        out_texts.extend(decoded_texts)
    return out_texts

if __name__ == "__main__":
    opts = parse_args()
    random.seed(opts.seed)

    date_understanding = json.load(open("prompting/date_understanding.json", "r"))
    examples = date_understanding["examples"]
    prompts = examples[v["input"] for v in examples]


    model = HookedTransformer.from_pretrained(opts.model_name, n_devices=2)

    dyck.tok_probs(model)
    dl = DataLoader(dyck, batch_size=opts.batch_size, shuffle=False)

    out_texts = eval_pass(model, dl)

    fwrite = dict()
    fwrite["predictions"] = out_texts
    fwrite["targets"] = [v[1] for v in dyck._probs]
    fwrite["inputs"] = [v[0] for v in dyck._probs]

    json.dump(fwrite, open(f"{opts.ofname}.json", "w"))
