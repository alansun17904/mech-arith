import tqdm
import json
import argparse
import random
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from prompting.dyck_dataset import DyckDataset


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

    dyck = DyckDataset(n=opts.n, max_length=opts.length)
    dyck.to_str(shots=opts.shots, cot=opts.cot)

    model = HookedTransformer.from_pretrained(opts.model_name, n_devices=2)

    dyck.tok_probs(model)
    dl = DataLoader(dyck, batch_size=opts.batch_size, shuffle=False)

    out_texts = eval_pass(model, dl)

    fwrite = dict()
    fwrite["predictions"] = out_texts
    fwrite["targets"] = [v[1] for v in dyck._probs]
    fwrite["inputs"] = [v[0] for v in dyck._probs]

    json.dump(fwrite, open(f"{opts.ofname}.json", "w"))
