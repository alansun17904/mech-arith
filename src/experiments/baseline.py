import pickle
import argparse
from transformer_lens import HookedTransformer

from .utils import seed_everything, parse_key_value_pairs, make_dataset, eval_pass
from cdatasets import DatasetBuilder, PromptFormatter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--ndevices", type=int, help="number of devices", default=1)
    parser.add_argument("--seed", type=int, help="random seed", default=42)
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
