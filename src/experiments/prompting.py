import os
import sys
import pickle
import random
import argparse
from functools import partial

from cdatasets import DatasetBuilder, PromptFormatter
from eap import Graph, attribute, evaluate_baseline, evaluate_graph
from .utils import (
    seed_everything,
    parse_key_value_pairs,
    make_dataset,
    get_metric,
    get_extraction,
    extraction_schema,
    kl_all_pos
)
from cdatasets import PromptDataset

import torch
import torch.nn.functional as F
from scalene import scalene_profiler
from transformer_lens import HookedTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--ndevices", type=int, help="number of devices", default=1)
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--n", type=int, help="number of prompts to patch", default=1000)
    parser.add_argument(
        "--response_name",
        type=str,
        help="model response",
        required=True,
    )
    parser.add_argument(
        "--patching_metric", type=str, default="kl", help="patching metric"
    )
    parser.add_argument(
        "--extraction",
        type=str,
        default="last_token",
        help="method for extracting comparison tokens",
    )
    parser.add_argument("--ig_steps", type=int, default=5, help="number of IG steps")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opts = parse_args()
    seed_everything(opts.seed)

    model = HookedTransformer.from_pretrained(opts.model_name, n_devices=opts.ndevices, dtype=torch.bfloat16)

    dataset = PromptDataset(opts.response_name, opts.n)
    dataset.get_questions()
    dataset.format_questions()
    dataloader = dataset.to_dataloader(model, opts.batch_size)

    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    metric = partial(kl_all_pos, model)

    g = Graph.from_model(model)
    attribute(model, g, dataloader, metric, method="EAP-IG", ig_steps=opts.ig_steps)
    g.apply_topn(200, absolute=False)
    g.to_json(f"{opts.ofname}.json")
    g.prune_dead_nodes()

    baseline = evaluate_baseline(model, dataloader, metric)
    results = evaluate_graph(model, g, dataloader, metric)

    diff = (results - baseline).mean().item()

    print(f"The circuit incurred extra {diff} loss.")

    gz = g.to_graphviz()
    gz.draw(f"{opts.ofname}.png", prog="dot")
