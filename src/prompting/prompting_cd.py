import os
import re
import sys
import json
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
    parser.add_argument("in_fname", type=str, help="input filename")
    parser.add_argument(
        "--psize", type=int, help="partition size as a percentage", default=0.1
    )
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--seed", type=int, help="random seed", default=37)
    return parser.parse_args()


def collate_fn(model, xs):
    clean, corrupted_strings, labels = zip(*xs)
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


def metric(
    model: HookedTransformer,
    logits,
    clean_logits,
    input_length,
    labels,
    mean=True,
    loss=True,
):
    bs, npos, dvocab = logits.shape
    starting_pos = (npos - input_length).to(logits.device)
    pos_ids = torch.arange(npos, device=logits.device).unsqueeze(0).expand(bs, -1)
    mask = (pos_ids >= starting_pos.unsqueeze(1)) & (pos_ids < npos - 1)
    mask = mask.unsqueeze(-1).expand(-1, -1, dvocab)

    clean_logits_selected = clean_logits[mask].view(-1, dvocab)
    logits_selected = logits[mask].view(-1, dvocab)

    probs = torch.softmax(logits_selected, dim=-1)
    clean_probs = torch.softmax(clean_logits_selected, dim=-1)
    results = F.kl_div(
        probs.log(), clean_probs.log(), log_target=True, reduction="none"
    ).mean(-1)
    return results.mean() if mean else results


def perplexity(
    model: HookedTransformer,
    logits,
    clean_logits,
    input_length,
    labels,
    mean=True,
    loss=True,
):
    bs, npos, dvocab = logits.shape
    log_probs = F.log_softmax(logits, dim=-1).to(logits.device)
    label_toks = model.to_tokens(labels, prepend_bos=True, padding_side="left").to(
        logits.device
    )
    correct_log_probs = log_probs.gather(dim=-1, index=label_toks.unsqueeze(-1))
    nll = -correct_log_probs.squeeze(-1)

    index_mask = torch.zeros_like(nll, dtype=torch.bool)
    str_tokens = model.to_str_tokens(labels)
    starting_pos = (npos - input_length).to(logits.device)

    for i in range(logits.shape[0]):
        index_mask[i, starting_pos[i] : logits.shape[1] - 1] = 1

    nll = nll * index_mask.float()

    mean_nll = nll.sum(dim=-1) / index_mask.sum(dim=-1)
    perplexity = mean_nll.mean()
    return perplexity


def get_partitions(clean_strings, psize):
    for i in range(0, len(clean_strings), int(len(clean_strings) * psize)):
        yield clean_strings[i : i + int(len(clean_strings) * psize)]


if __name__ == "__main__":
    opts = parse_args()
    seed_everything(opts.seed)

    in_prompts = json.load(open(opts.in_fname, "r"))
    clean_strings = in_prompts["predictions"]

    model = HookedTransformer.from_pretrained(opts.model_name, n_devices=2)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    g = Graph.from_model(model)

    for i, partition in enumerate(get_partitions(clean_strings, opts.psize)):
        corrupted_strings = partition[:]
        random.shuffle(corrupted_strings)
        data_dict = {
            "clean": partition,
            "corrupted": corrupted_strings,
            "label": partition,
        }
        df = pd.DataFrame(data_dict)
        eap_ds = EAPDataset(df)
        collator = partial(collate_fn, model)
        dataloader = eap_ds.to_dataloader(opts.batch_size, collate_fn=collator)

        attribute(
            model, g, dataloader, partial(metric, model), method="EAP-IG", ig_steps=5
        )
        g.apply_topn(200, absolute=True)
        g.to_json(f"{opts.ofname}-{i}.json")
        g.prune_dead_nodes()

        baseline = evaluate_baseline(model, dataloader, partial(perplexity, model))
        results = evaluate_graph(model, g, dataloader, partial(perplexity, model))

        diff = (results - baseline).mean().item()

        print(f"The circuit incurred extra {diff} perplexity.")

        gz = g.to_graphviz()
        gz.draw(f"{opts.ofname}-{i}.png", prog="dot")

        print(g.count_included_nodes(), g.count_included_edges())
