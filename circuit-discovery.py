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
import pickle
import random
from functools import partial

import torch
import numpy as np
from transformers import AutoTokenizer
import transformer_lens.patching as patching
from transformer_lens import HookedTransformer, ActivationCache
import transformer_lens.utils as utils

from arith_bench import seed_everything, arith_probs


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
        counter = len(om) - 1
        idxs = []
        ans_idxs = []
        for j in range(len(om) - 1, -1, -1):
            if om[j][0] >= cindices[i]:
                idxs.append(counter)
                ans_idxs.append(tokenized_input["input_ids"][i][counter].item())
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
    clean_ans_logits = clean_logits[n, answer_token_idxs, answer_token_ids]
    corr_ans_logits = corrupted_logits[n, answer_token_idxs, answer_token_ids]
    pat_ans_logits = patched_logits[n, answer_token_idxs, answer_token_ids]
    pert_change = (clean_ans_logits - pat_ans_logits) / (
        clean_ans_logits - corr_ans_logits
    )
    return torch.sum(mask * pert_change) / torch.sum(mask)


def corr_hook_noise(clean, hook):
    che = torch.std(clean).to("cpu") * 3
    clean = clean + torch.normal(
        torch.zeros(clean.shape), che * torch.ones(clean.shape)
    ).to(clean.device)
    return clean


if __name__ == "__main__":
    seed_everything(42)

    toks = AutoTokenizer.from_pretrained("google/gemma-2-2b", padding_side="left")
    toks.pad_token = toks.bos_token

    m = HookedTransformer.from_pretrained("gemma-2b")

    probs = arith_probs(1, 1, n=5)
    str_probs = [fewshot_probs(probs, k=0)[0]]

    tokenized_input = toks(
        str_probs, return_offsets_mapping=True, padding=True, return_tensors="pt"
    )
    idxs, ans_ids = answer_logit_indices(tokenized_input, str_probs)
    idxs, ans_ids = pad_ans_idx_id(idxs, ans_ids)
    idxs, ans_ids = torch.LongTensor(idxs), torch.LongTensor(ans_ids)

    clean_logits = m(tokenized_input["input_ids"])

    with m.hooks(fwd_hooks=[("hook_embed", corr_hook_noise)]):
        noise_logits, noise_cache = m.run_with_cache(tokenized_input["input_ids"])

    metric = partial(
        logit_diff,
        clean_logits=clean_logits,
        corrupted_logits=noise_logits,
        answer_token_idxs=idxs,
        answer_token_ids=ans_ids,
    )

    resid_pre_act_patch_results = patching.get_act_patch_resid_pre(
        m, tokenized_input["input_ids"], noise_cache, metric
    )

    mlp_out_results = patching.get_act_patch_mlp_out(
        m, tokenized_input["input_ids"], noise_cache, metric
    )

    all_blocks_results = patching.get_act_patch_block_every(
        m, tokenized_input["input_ids"], noise_cache, metric
    )

    pickle.dump(str_probs, open("probs.pkl", "wb"))
    pickle.dump(mlp_out_results, open("mlp_out_results.pkl", "wb"))
    pickle.dump(all_blocks_results, open("all_blocks_results.pkl", "wb"))
    pickle.dump(resid_pre_act_patch_results, open("resid_pre_act_results.pkl", "wb"))
