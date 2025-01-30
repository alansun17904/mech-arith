import os
import sys
import pickle
import random
import argparse
import json
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
from .bool_circuit_discovery import collate_fn_bool, metric, correct_probs


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
    parser.add_argument("graph_file", type=str, help="graph")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("--no_not", action="store_true")
    parser.add_argument("--no_and", action="store_true")
    parser.add_argument("--no_or", action="store_true")
    parser.add_argument("--allow_parentheses", action="store_true")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--exp_length", type=int, help="expression length", default=5)
    parser.add_argument("--depth", type=int, help="parenthetical depth", default=3)
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--num", type=int, help="num problems", default=1000)
    parser.add_argument("--shots", type=int, help="few-shot prompting", default=3)
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    seed_everything(opts.seed)

    model = HookedTransformer.from_pretrained(
        opts.model_name, n_devices=1, trust_remote_code=True
    )

    print("expression_length", opts.exp_length, "depth", opts.depth)

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
        False: [i for i in range(len(clean_labels)) if not clean_labels[i]],
    }

    corrupted_prompts = []
    for i in range(len(clean_prompts)):
        cf_label_idxs = tf_labels[not clean_labels[i]]
        cf_prompt_idx = random.choice(cf_label_idxs)
        corrupted_prompts.append(clean_prompts[cf_prompt_idx])
    data_dict = {
        "clean": clean_prompts,
        "corrupted": corrupted_prompts,
        "label": clean_labels,
    }
    df = pd.DataFrame(data_dict)

    print(df.head())

    eap_ds = EAPDataset(df)
    collator = partial(collate_fn_bool, model)
    dataloader = eap_ds.to_dataloader(opts.batch_size, collate_fn=collator)

    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    t_token, f_token = model.to_single_token("True"), model.to_single_token("False")

    g = Graph.from_json(opts.graph_file)

    ## get the total number of edges and the take a fraction of it, go 5% at a time
    n_components = len(g.edges)
    perf = []
    components = []
    for prct in np.logspace(0, np.log10(n_components), 20):
        remained_components = int(prct)
        components.append(remained_components)
        g.apply_topn(remained_components, absolute=True)
        g.prune_dead_nodes()

        empty = not g.nodes["logits"].in_graph
        if empty and prct != 1:
            continue

        result = evaluate_graph(
            model,
            g,
            dataloader,
            partial(correct_probs, model, t_token=t_token, f_token=f_token),
        )
        perf.append(result.mean().item())
        print("n_comps:", remained_components, "result", result.mean().item())

    pareto = {"perf": perf, "components": components}

    json.dump(pareto, open(f"{opts.ofname}-pareto.json", "w+"))
