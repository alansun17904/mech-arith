import os
import tqdm
import pickle
import torch
import random
import argparse
from transformer_lens import HookedTransformer

from .utils import seed_everything
from cdatasets import DatasetBuilder, PromptFormatter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--ndevices", type=int, help="number of devices", default=1)
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--num", type=int, help="num problems", default=1000)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DatasetBuilder.ids.keys()),
        help="dataset name",
        required=True,
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=list(PromptFormatter.ids.keys()),
        help="format name",
        required=True,
    )
    parser.add_argument("--data_params", nargs="*", default=[], help="dataset params")
    parser.add_argument("--format_params", nargs="*", default=[], help="format params")
    args = parser.parse_args()
    args.data_params = parse_key_value_pairs(args.data_params)
    args.format_params = parse_key_value_pairs(args.format_params)
    return args

def parse_key_value_pairs(pairs):
    """Convert a list of key=value strings into a dictionary."""
    params = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"Invalid argument format: {pair}. Expected key=value.")
        key, value = pair.split("=", 1)
        # Attempt to convert to int or float if applicable\
        try:
            params[key] = eval(value)
        except:
            params[key] = value
    return params

def make_dataset(dataset_id, dataset_params, formatter_id, formatter_params):
    db = DatasetBuilder.get_strategy(dataset_id)
    formatter = PromptFormatter.get_strategy(formatter_id, **formatter_params)
    for k, v in dataset_params.items():
        db.set_param(k, v)
    dataset = db.build()
    dataset.get_questions()
    dataset.format_questions(formatter)
    return dataset
    
@torch.inference_mode()
def eval_pass(model, dataloader):
    model.eval()
    inputs, out_texts, labels = [], [], []
    for clean_prompt, _, label in tqdm.tqdm(dataloader):
        tokens, _, _, _ = clean_prompt
        outputs = model.generate(tokens, max_new_tokens=15, verbose=False)
        decoded_texts = model.to_string(outputs)
        out_texts.extend(decoded_texts)
        labels.extend(label)
        inputs.extend(model.to_string(tokens))
    return out_texts, labels


if __name__ == "__main__":
    opts = parse_args()
    seed_everything(opts.seed)
    dataset = make_dataset(opts.dataset, opts.data_params, opts.format, opts.format_params)
    
    model = HookedTransformer.from_pretrained(opts.model_name, n_devices=opts.ndevices)
    loader = dataset.to_dataloader(model, opts.batch_size)

    inputs, out_texts, labels = eval_pass(model, loader)
    d = {
        "input": inputs,
        "output": out_texts,
        "target": labels,
    }
    pickle.dump(d, open(f"{opts.ofname}-benchmark.pkl", "wb+"))
