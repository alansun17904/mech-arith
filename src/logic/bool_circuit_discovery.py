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

from .bool_dataset import BooleanDataset
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
    parser.add_argument("--exp_length", type=int, help="expression length", default=5)
    parser.add_argument(
        "--depth", type=int, help="parenthetical depth", default=3
    )
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--num", type=int, help="num problems", default=1000)
    parser.add_argument("--shots", type=int, help="few-shot prompting", default=3)
    return parser.parse_args()

def collate_fn_bool(model, xs):
    clean, corrupted, labels = zip(*xs)
    # the clean and corrupted strings together
    batch_size = len(clean)
    all_examples = clean + corrupted
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

def metric(model, logits, clean_logits, input_length, labels, t_token, f_token, mean=True, loss=True):
    corrupted_logit_tokens = np.where(labels, f_token, t_token)
    clean_logit_tokens = np.where(labels, t_token, f_token)
    last_token_logits = logits[:,-1,:]
    last_token_clean_logits = clean_logits[:,-1,:]

    last_token_probs = torch.softmax(last_token_logits, dim=1)
    last_token_clean_probs = torch.softmax(last_token_clean_logits, dim=1)

    last_token_probs_clean = last_token_probs[:,clean_logit_tokens]
    last_token_probs_corr = last_token_probs[:,corrupted_logit_tokens]
    last_token_clean_probs_clean = last_token_clean_probs[:,clean_logit_tokens]
    last_token_clean_probs_corr = last_token_clean_probs[:,corrupted_logit_tokens]

    score = (
        (last_token_probs_corr - last_token_clean_probs_corr) /
            (last_token_clean_probs_corr + 1e-5)
        + (last_token_clean_probs_clean - last_token_probs_clean) /
            (last_token_probs_clean + 1e-5)
    )

    if loss:
        score = -score
    if mean:
        score = torch.mean(score)
    return score

def correct_probs(model, logits, clean_logits, input_length, labels, t_token, f_token):
    last_logit_tf = logits[:,-1,[f_token, t_token]]
    predict = torch.argmax(last_logit_tf, dim=1)
    correct = sum([labels[i] == predict[i] for i in range(len(labels))])
    return correct / len(labels)


if __name__ == "__main__":
    opts = parse_args()
    seed_everything(opts.seed)

    model = HookedTransformer.from_pretrained(
        opts.model_name, n_devices=1, trust_remote_code=True
    )

    unary = tuple() if opts.no_not else ("not",)
    binary_ops = tuple() if opts.no_and else ("and",)
    binary_ops = binary_ops if opts.no_or else binary_ops + ("or",)

    bd = BooleanDataset(
        expression_lengths=opts.exp_length,
        unary_ops=unary,
        binary_ops=binary_ops,
        allow_parentheses=opts.allow_parentheses,
        parenthetical_depth=opts.depth,
    )

    bd.bool_probs(n=opts.num)
    bd.to_str(shots=3)

    clean_prompts = bd.prompts
    clean_labels = bd.labels

    tf_labels = {
        True: [i for i in range(len(clean_labels)) if clean_labels[i]],
        False: [i for i in range(len(clean_labels)) if not clean_labels[i]]
    }

    corrupted_prompts = []
    for i in range(len(clean_prompts)):
        cf_label_idxs = tf_labels[not clean_labels[i]]
        cf_prompt_idx = random.choice(cf_label_idxs)
        corrupted_prompts.append(clean_prompts[cf_prompt_idx])
    data_dict = {
        "clean": clean_prompts,
        "corrupted": corrupted_prompts,
        "label": clean_labels
    }
    df = pd.DataFrame(data_dict)

    eap_ds = EAPDataset(df)
    collator = partial(collate_fn_bool, model)
    dataloader = eap_ds.to_dataloader(opts.batch_size, collate_fn=collator)

    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    t_token, f_token = model.to_single_token("True"), model.to_single_token("False")

    g = Graph.from_model(model)
    attribute(model, g, dataloader, partial(metric, model, t_token=t_token, f_token=f_token), method="EAP-IG", ig_steps=5)
    g.apply_topn(200, absolute=True)
    g.to_json(f"{opts.ofname}.json")
    g.prune_dead_nodes()

    baseline = evaluate_baseline(model, dataloader, partial(correct_probs, model, t_token=t_token, f_token=f_token))
    results = evaluate_graph(model, g, dataloader, partial(correct_probs, model, t_token=t_token, f_token=f_token))

    print("Baseline acc", baseline.mean().item(), "circuit acc", results.mean().item())

    gz = g.to_graphviz()
    gz.draw(f"{opts.ofname}.png", prog="dot")

    print(g.count_included_nodes(), g.count_included_edges())
