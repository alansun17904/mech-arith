# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Single Digit Add Memorization

# +
import os
import sys
import pickle
import random
import argparse
from functools import partial

import torch
import numpy as np
from transformers import AutoTokenizer
import transformer_lens.patching as patching
from transformer_lens import HookedTransformer, ActivationCache
import transformer_lens.utils as utils

from arith_bench import seed_everything, arith_probs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("op1_digs", type=int, help="digits in the first operand")
    parser.add_argument("op2_digs", type=int, help="digits in the second operand")
    parser.add_argument("--fewshot_k", type=int, help="fewshot problems", default=2)
    return parser.parse_args()


def fewshot_probs(probs, sub=False, k=2):
    """Generate list of problems with fewshot prompting."""
    convert_few = lambda x: f"{x[0]} {'-' if sub else '+'} {x[1]} = {x[2]}"
    str_probs = []
    for _, p in enumerate(probs):
        # sample the few shot problems
        if k != 0:
            few_shot_examples = [convert_few(v) for v in random.sample(probs, k=k)]
            str_probs.append("\n".join(few_shot_examples) + "\n" + convert_few(p))
        else:
            str_probs.append(convert_few(p))
    return str_probs


def answer_logit_indices(tokenized_input, problems):
    """Given a list of problems and their corresponding tokenization find the
    set of tokens that correspond to the answer and their token ids."""
    cindices = list(map(lambda x: x.rfind(" ") + 1, problems))
    indices = []
    answer_indices = []
    for i in range(len(problems)):
        om = tokenized_input["offset_mapping"][i]
        counter = len(om) - 2  # the last token used as input
        idxs = []  # the logit indices that we will be interested in
        ans_idxs = []  # the answers to those logit indices
        for j in range(len(om) - 1, -1, -1):
            if om[j][0] >= cindices[i]:
                idxs.append(counter)
                ans_idxs.append(
                    tokenized_input["input_ids"][i][counter + 1].item()
                )
                counter -= 1
            else:
                break
        indices.append(idxs)
        answer_indices.append(ans_idxs)
    return indices, answer_indices


def pad_ans_idx_id(ans_idxs, ans_ids):
    maxlen = lambda x: max(x, key=lambda y: len(y))
    max_idx_len, max_ids_len = maxlen(ans_idxs), maxlen(ans_ids)
    mil = len(max_idx_len)
    ail = len(max_ids_len)
    for i in range(len(idxs)):
        if len(ans_idxs[i]) < mil:
            ans_idxs[i] = [0] * (mil - len(ans_idxs[i])) + ans_idxs[i]
        if len(ans_ids[i]) < ail:
            ans_ids[i] = [0] * (ail - len(ans_ids[i])) + ans_ids[i]
    return ans_idxs, ans_ids


def logit_diff(
    patched_logits, clean_logits, corrupted_logits, answer_token_idxs, answer_token_ids
):
    mask = (answer_token_idxs != 0).to(patched_logits.device)
    n = (
        torch.arange(len(answer_token_idxs))
        .unsqueeze(1)
        .expand(len(answer_token_idxs), answer_token_idxs.shape[1])
    )
    patched_logits, clean_logits, corrupted_logits = (
        torch.softmax(patched_logits, dim=2),
        torch.softmax(clean_logits, dim=2),
        torch.softmax(corrupted_logits, dim=2)
    )
    corr_ans_logits = corrupted_logits[n, answer_token_idxs, answer_token_ids]
    pat_ans_logits = patched_logits[n, answer_token_idxs, answer_token_ids]
    pert_change = (pat_ans_logits - corr_ans_logits)
    return torch.sum(mask * pert_change) / torch.sum(mask)


def corr_hook_noise(clean, hook):
    che = torch.std(clean).to("cpu") * 3
    clean = clean + torch.normal(
        torch.zeros(clean.shape), che * torch.ones(clean.shape)
    ).to(clean.device)
    return clean


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    toks = AutoTokenizer.from_pretrained("google/gemma-2-2b", padding_side="left")
    toks.pad_token = toks.bos_token

    m = HookedTransformer.from_pretrained("gemma-2b")

    N = 20

    probs = arith_probs(args.op1_digs, args.op2_digs, n=N)
    str_probs = fewshot_probs(probs, k=args.fewshot_k)

    tokenized_input = toks(
        str_probs, return_offsets_mapping=True, padding=True, return_tensors="pt"
    )
    idxs, ans_ids = answer_logit_indices(tokenized_input, str_probs)
    idxs, ans_ids = pad_ans_idx_id(idxs, ans_ids)
    idxs, ans_ids = torch.LongTensor(idxs), torch.LongTensor(ans_ids)

    # print(str_probs)
    # print(tokenized_input["input_ids"].shape)
    # print(idxs, ans_ids)
    # print(m.to_string(ans_ids))

    # sys.exit(0)

    batch_size = 24 

    chunks = torch.chunk(torch.arange(N), N // batch_size)

    all_blocks = None 
    all_clean_logits = None 

    for i, batch in enumerate(chunks):
        batch_ids = tokenized_input["input_ids"][batch] 
        batch_idxs = idxs[batch]
        batch_ans_ids = ans_ids[batch]
        clean_logits = m(batch_ids)

        with m.hooks(fwd_hooks=[("hook_embed", corr_hook_noise)]):
            noise_logits, noise_cache = m.run_with_cache(batch_ids)

        metric = partial(
            logit_diff,
            clean_logits=clean_logits,
            corrupted_logits=noise_logits,
            answer_token_idxs=batch_idxs,
            answer_token_ids=batch_ans_ids,
        )

        attn_head_results = patching.get_act_patch_attn_head_pattern_all_pos(
            m, batch_ids, noise_cache, metric
        ).to("cpu")
        
        if all_blocks is None:
            all_blocks = attn_head_results * len(batch)
        else:
            all_blocks += attn_head_results * len(batch)

    pickle.dump(str_probs, open(f"data/patching_circuit/{args.op1_digs}-{args.op2_digs}-probs.pkl", "wb"))
    pickle.dump((1 / N) * all_blocks, open(f"data/patching_circuit/{args.op1_digs}-{args.op2_digs}-all_blocks.pkl", "wb"))
