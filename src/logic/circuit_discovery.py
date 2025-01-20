import os
import sys
import pickle
import random
import argparse
from functools import partial

import torch
import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer
import transformer_lens.patching as patching
from transformer_lens import HookedTransformer, ActivationCache
import torch.nn.functional as F

from .arith_dataset import Op, ArithDataset
from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute, tokenize_plus
import eap.utils as utils
from eap.dataset import EAPDataset


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
    parser.add_argument("dig1", type=int, help="number of digits in first operand")
    parser.add_argument("dig2", type=int, help="number of digits in second operand")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--seed", type=int, help="random seed", default=37)
    parser.add_argument("--num", type=int, help="num problems", default=1024)
    parser.add_argument("--shots", type=int, help="few-shot prompting", default=3)
    return parser.parse_args()

def get_equal_pos(str_tokens):
    for i in range(len(str_tokens) - 1, -1, -1):
        if "=" in str_tokens[i]:
            return i
    return -1


def get_logit_positions(model: HookedTransformer, logits, labels):
    str_tokens = model.to_str_tokens(labels)
    ## everything including the equal sign and not including the last token
    ## we count as a part of the answer token sequence
    equal_sign_pos = list(map(get_equal_pos, str_tokens))
    return logits[...,equal_sign_pos:, ...]

def metric(model: HookedTransformer, logits, clean_logits, input_length, labels, mean=True, loss=True):
    bs, npos, dvocab = logits.shape
    str_tokens = model.to_str_tokens(labels)
    equal_sign_pos = torch.LongTensor(list(map(get_equal_pos, str_tokens))).to(logits.device)
    pos_ids = torch.arange(npos, device=logits.device).unsqueeze(0).expand(bs, -1)
    mask = (pos_ids >= equal_sign_pos.unsqueeze(1)) & (pos_ids < npos - 1)
    mask = mask.unsqueeze(-1).expand(-1, -1, dvocab)

    clean_logits_selected = clean_logits[mask].view(-1, dvocab)
    logits_selected = logits[mask].view(-1, dvocab)

    probs = torch.softmax(logits_selected, dim=-1)
    clean_probs = torch.softmax(clean_logits_selected, dim=-1)
    results = F.kl_div(probs.log(), clean_probs.log(), log_target=True, reduction="none").mean(-1)
    return results.mean() if mean else results

def perplexity(model: HookedTransformer, logits, clean_logits, input_length, labels, mean=True, loss=True):
    log_probs = F.log_softmax(logits, dim=-1)
    label_toks = model.to_tokens(labels, prepend_bos=True, padding_side="left")
    correct_log_probs = log_probs.gather(dim=-1, index=label_toks.unsqueeze(-1))
    nll = -correct_log_probs.squeeze(-1)

    index_mask = torch.zeros_like(nll, dtype=torch.bool)
    str_tokens = model.to_str_tokens(labels)
    equal_sign_pos = torch.LongTensor(list(map(get_equal_pos, str_tokens))).to(logits.device)

    for i in range(logits.shape[0]):
        index_mask[i, equal_sign_pos[i]:logits.shape[1]-1] = 1

    nll = nll * index_mask.float()

    mean_nll = nll.sum(dim=-1) / index_mask.sum(dim=-1)
    perplexity = torch.exp(mean_nll).mean()
    return perplexity


if __name__ == "__main__":
    opts = parse_args()
    seed_everything(opts.seed)
    arith_dataset = ArithDataset(Op[opts.op])
    arith_dataset.arith_probs(opts.dig1, opts.dig2, opts.num)
    clean_strings = arith_dataset.to_str(shots=opts.shots)
    corrupted_strings = clean_strings[:]
    random.shuffle(corrupted_strings)

    data_dict = {
        "clean": clean_strings,
        "corrupted": corrupted_strings,
        "label": clean_strings
    }
    df = pd.DataFrame(data_dict)

    print("Number of patching data points:", len(df))
    print("Batch size:", opts.batch_size)


    eap_ds = EAPDataset(df)
    dataloader = eap_ds.to_dataloader(opts.batch_size)

    model = HookedTransformer.from_pretrained(opts.model_name)

    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    g = Graph.from_model(model)
    attribute(model, g, dataloader, partial(metric, model), method="EAP-IG", ig_steps=5)
    g.apply_topn(100, absolute=True)
    g.prune_dead_nodes()
    g.to_json(f"{opts.ofname}.json")

    baseline = evaluate_baseline(model, dataloader, partial(perplexity, model))
    results = evaluate_graph(model, g, dataloader, partial(perplexity, model))

    diff = (results - baseline).mean().item()

    print(f"The circuit incurred extra {diff} perplexity.")

    gz = g.to_graphviz()
    gz.draw(f"{opts.ofname}.png", prog="dot")

    print(g.count_included_nodes(), g.count_included_edges())
