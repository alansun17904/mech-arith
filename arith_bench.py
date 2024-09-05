import re
import os
import random
import pickle
import argparse
import itertools

import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="huggingface model id")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--seed", type=int, help="random seed", default=37)
    parser.add_argument(
        "--subtraction",
        action="store_true",
        help="if given, perform subtract instead of addition",
    )
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer.pad_token = tokenizer.bos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


def arith_probs(dig1: int, dig2: int, n=1000, sub=False):
    """Generate binary arithmetic problems (both addition and subtraction).

    Args:
        dig1 (int): the number of digits in the first operand.
        dig2 (int): the number of digits in the second operand.
        n (int): the total number of problems.
        sub (bool): if true, the operation of the problem becomes subtraction;
            otherwise, generates only addition problems.
    Returns:
        a list of tuple of three integers (op1, op2, ans) where op1, op2
        have dig1, dig2 number of digits, respectively;
        and op1 +/- op2 = ans. Note that op1 >= 0 and op2 >= 0.
    """
    probs = []
    for _ in range(n):
        a = random.randint(10 ** (dig1 - 1), 10**dig1 - 1)
        b = random.randint(10 ** (dig2 - 1), 10**dig2 - 1)
        ans = a + b if not sub else a - b
        probs.append((a, b, ans))
    return probs


def parse_answer(result: str):
    """Given the causal response from a language model parse the problem
    it was required to solve using regex.

    Args:
        result (str): Resulting output from a LM.
    Returns:
        a tuple of three integers (a, b, ans) where the given question to
        LM was `a + b = ` and the LM's response was `ans`.
    """
    ptrn = r"(-?\d+)\s+\(-|+)\s+(-?\d+)\s=\s(-?\d+)"
    srch = re.search(ptrn, result.split("\n")[2])
    if srch is not None:
        return (int(srch.group(1)), int(srch.group(2)), int(srch.group(3)))
    else:
        # get the problem
        ptrn = r"(-?\d+)\s+\+\s+(-?\d+)"
        srch = re.search(ptrn, result.split("\n")[2])
        return (int(srch.group(1)), int(srch.group(2)), -np.inf)


def tok_probs(tokenizer, probs, sub=False, k=2):
    """Tokenize a list of problems in the form of (a, b, ans).

    Args:
        tokenizer: model tokenizer
        probs (List[Tuple[int, int, int]]): a list of tuples of the form
            (a, b, ans).
        sub (bool): whether the problem is subtract
        k (int): number of examples given to the model for in-context learning
    Returns:
        list of problems have been tokenized.
    """
    convert_prob = lambda x: f"{x[0]} {'-' if sub else '+'} {x[1]} = "
    convert_few = lambda x: f"{x[0]} {'-' if sub else '+'} {x[1]} = {x[2]}"
    str_probs = []
    for i, p in enumerate(probs):
        # sample the few shot problems
        few_shot_examples = [convert_few(v) for v in random.sample(probs, k=k)]
        # str_probs.append("100 + 200 = 300\n520 + 890 = 1410" + "\n" + convert_prob(p))
        str_probs.append("\n".join(few_shot_examples) + "\n" + convert_prob(p))
    # return str_probs
    return tokenizer.batch_encode_plus(str_probs, return_tensors="pt", padding=True)


class Toks(Dataset):
    def __init__(self, toks):
        self.toks = toks

    def __len__(self):
        return len(self.toks["input_ids"])

    def __getitem__(self, idx):
        return self.toks["input_ids"][idx], self.toks["attention_mask"][idx]


if __name__ == "__main__":
    from transformers import logging

    logging.set_verbosity_warning()

    args = parse_args()
    seed_everything(args.seed)
    tokenizer, model = init_model(args.model_name)

    d = dict()

    for dig1 in tqdm.tqdm(range(1, 9)):
        for dig2 in tqdm.tqdm(range(1, 9), leave=False):
            probs = arith_probs(dig1, dig2, sub=args.subtraction)
            tokenized = tok_probs(tokenizer, probs, sub=args.subtraction)

            dl = DataLoader(Toks(tokenized), batch_size=128)

            texts = []
            for x, y in dl:
                x, y = x.to(model.device), y.to(model.device)
                outputs = model.generate(
                    input_ids=x,
                    #                    attention_mask=y,
                    max_new_tokens=15,
                )
                decoded_texts = tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                # print([v for v in decoded_texts])
                texts.append(decoded_texts)

            # parsed_texts = list(map(parse_answer, itertools.chain.from_iterable(texts)))
            d[(dig1, dig2)] = list(itertools.chain.from_iterable(texts))

    pickle.dump(d, open(f"{args.ofname}-benchmark.pkl", "wb+"))
