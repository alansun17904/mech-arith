import os
import tqdm
import pickle
import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from arith_dataset import Op, ArithDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("op", choices=["ADD", "SUB", "MUL", "DIV"], help="operation")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--seed", type=int, help="random seed", default=37)
    parser.add_argument("--num", type=int, help="num problems", default=100)
    parser.add_argument("--shots", type=int, help="few-shot prompting", default=3)
    return parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def init_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if "gpt" in model_name or "llama" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    return model, tokenizer

def eval_pass(model, tokenizer, dataloader):
    model.eval()
    out_texts = []
    for input_token, attn_mask in dataloader:
        input_token = input_token.to(model.device)
        attn_mask = attn_mask.to(model.device)
        outputs = model.generate(
            input_ids=input_token,
            attention_mask=attn_mask,
            max_new_tokens=15
        )
        decoded_texts = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        out_texts.extend(decoded_texts)
    return out_texts


if __name__ == "__main__":
    from transformers import logging

    logging.set_verbosity_error()
    args = parse_args()
    
    seed_everything(args.seed)
    model, tokenizer = init_model(args.model_name)
    op = Op[args.op]
    d = dict()

    for dig1 in tqdm.tqdm(range(1, 9)):
        lower = dig1 if op is Op.DIV else 1
        for dig2 in tqdm.tqdm(range(lower, 9), leave=False):
            dataset = ArithDataset(op)
            dataset.arith_probs(dig1, dig2, args.num)
            dataset.to_str(shots=args.shots)
            dataset.tok_probs(tokenizer)
            loader = DataLoader(dataset, batch_size=args.batch_size)

            out_texts = eval_pass(model, tokenizer, loader)
            d[(dig1, dig2)] = [dataset.parse_ans(v) for v in out_texts]
    pickle.dump(d, open(f"{args.ofname}-benchmark.json", "wb+"))
