import os
import tqdm
import pickle
import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from utils import seed_everything
from logic.arith_dataset import Op, ArithDataset


def seed_everything(seed: int = 42):
    random.seed(seed)
    transformers.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("op", choices=["ADD", "SUB", "MUL", "DIV"], help="operation")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--num", type=int, help="num problems", default=1000)
    parser.add_argument("--shots", type=int, help="few-shot prompting", default=3)
    return parser.parse_args()

@torch.inference_mode()
def eval_pass(model, dataloader):
    model.eval()
    out_texts = []
    for input_token, attn_mask in dataloader:
        outputs = model.generate(
            input_token, max_new_tokens=15, verbose=False
        )
        decoded_texts = model.to_string(outputs)
        out_texts.extend(decoded_texts)
    return out_texts


if __name__ == "__main__":
    opts = parse_args()
    seed_everything(opts.seed)

    op = Op[opts.op]
    d = dict()
    model = HookedTransformer.from_pretrained(opts.model_name, n_devices=1)

    for dig1 in tqdm.tqdm(range(1, 9)):
        for dig2 in tqdm.tqdm(range(1, 9)):
            dataset = ArithDataset(op)
            dataset.arith_probs(dig1, dig2, opts.num)
            dataset.to_str(shots=opts.shots, add_ans=False)
            dataset.tok_probs(model)
            loader = DataLoader(dataset, batch_size=opts.batch_size)
            out_texts = eval_pass(model, loader)
            d[(dig1, dig2)] = [dataset.parse_ans(v) for v in out_texts]
    pickle.dump(d, open(f"{opts.ofname}-benchmark.pkl", "wb+"))
