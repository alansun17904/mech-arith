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
from transformer_lens.utils import get_attention_mask
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

def collate_fn_bool(model, xs):
    clean, corrupted, labels = zip(*xs)
    # the clean and corrupted strings together
    batch_size = len(clean)
    all_examples = clean + corrupted_strings
    tokens = model.to_tokens(all_examples, prepend_bos=True, padding_side="left")
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)
    input_lengths = attention_mask.sum(1)
    n_pos = attention_mask.size(1)
    return (
        (
            tokens[:batch_size],
            attention_mask[:batch_size],
            input_lengths[:batch_size],
            n_pos,
        ),
        (
            tokens[batch_size:],
            attention_mask[batch_size:],
            input_lengths[batch_size:],
            n_pos,
        ),
        list(labels),
    )

def get_logit_positions(model: HookedTransformer, logits, labels): ...

def metric(model, logits, clean_logits, input_length, labels, mean=True, loss=True): ...


if __name__ == "__main__":
    opts = parse_args()
    seed_everything(opts.seed)
    
