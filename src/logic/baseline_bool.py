import os
import tqdm
import pickle
import torch
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from utils import seed_everything
from logic.bool_dataset import BooleanDataset


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
    parser.add_argument("--no_not", action="store_true")
    parser.add_argument("--no_and", action="store_true")
    parser.add_argument("--no_or", action="store_true")
    parser.add_argument("--allow_parentheses", action="store_true")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument(
        "--depth", type=int, help="maximum parenthetical depth", default=3
    )
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--num", type=int, help="num problems", default=1000)
    parser.add_argument("--shots", type=int, help="few-shot prompting", default=3)
    return parser.parse_args()


@torch.inference_mode()
def eval_pass(model, dataloader):
    model.eval()

    total = 0
    correct = 0

    t_token, f_token = model.to_single_token("True"), model.to_single_token("False")
    for input_tokens, attn_mask, labels in dataloader:
        total += len(labels)
        logits = model(input_tokens, attention_mask=attn_mask)
        last_logits_tf = logits[:, -1, [f_token, t_token]]
        predict = torch.argmax(last_logits_tf, dim=1)
        correct += sum([labels[i] == predict[i] for i in range(len(labels))])
    return correct / total, total


if __name__ == "__main__":
    opts = parse_args()
    seed_everything(opts.seed)

    model = HookedTransformer.from_pretrained(
        opts.model_name, n_devices=1, trust_remote_code=True
    )

    d = dict()

    unary = tuple() if opts.no_not else ("not",)
    binary_ops = tuple() if opts.no_and else ("and",)
    binary_ops = binary_ops if opts.no_or else binary_ops + ("or",)

    print("unary ops", unary, "binary_ops", binary_ops)

    for i in range(3, 10):
        for dp in range(1, opts.depth + 1):
            bd = BooleanDataset(
                expression_lengths=i,
                unary_ops=unary,
                binary_ops=binary_ops,
                allow_parentheses=opts.allow_parentheses,
                parenthetical_depth=dp,
            )

            bd.bool_probs()
            bd.to_str(shots=3)
            bd.tok_probs(model)

            dl = DataLoader(bd, batch_size=opts.batch_size)

            d[(i, dp)] = eval_pass(model, dl)

            print(i, "expressions", "accuracy:", d[(i,dp)][0].item(), "depth", dp, "total", d[(i,dp)][1])

    pickle.dump(d, open(f"{opts.ofname}-benchmark.pkl", "wb+"))
