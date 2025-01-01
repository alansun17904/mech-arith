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


class CircuitDiscovery:
    pass


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
                ans_idxs.append(tokenized_input["input_ids"][i][counter + 1].item())
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
    patched_logits, clean_logits, corrupted_logits, clean_ans_pos, clean_ans_ids
):
    clean_mask = (clean_ans_pos != 0).to(patched_logits.device)

    n = (
        torch.arange(len(clean_ans_pos))
        .unsqueeze(1)
        .expand(len(clean_ans_pos), clean_ans_pos.shape[1])
    )
    patched_logits, corrupted_logits = (
        torch.softmax(patched_logits, dim=2),
        # torch.softmax(clean_logits, dim=2),
        torch.softmax(corrupted_logits, dim=2),
    )
    corr_ans_logits = corrupted_logits[n, clean_ans_pos, clean_ans_ids]
    pat_ans_logits = patched_logits[n, clean_ans_pos, clean_ans_ids]
    # clean_ans_logits = clean_logits[n, clean_ans_pos, clean_ans_ids]
    pert_change = pat_ans_logits - corr_ans_logits
    pert_zoned = clean_mask * (pert_change)

    pert_change_row = torch.sum(pert_zoned, axis=1) / torch.sum(clean_mask, axis=1)
    return torch.sum(pert_change_row) / len(pert_change_row)


def corr_hook_noise(clean, hook):
    che = torch.std(clean).to("cpu")
    clean = clean + torch.normal(
        torch.zeros(clean.shape), che * torch.ones(clean.shape)
    ).to(clean.device)
    return clean


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    print(torch.cuda.device_count(), torch.cuda.current_device())

    toks = AutoTokenizer.from_pretrained("google/gemma-2-2b", padding_side="left")
    toks.pad_token = toks.bos_token

    m = HookedTransformer.from_pretrained("gemma-2b")

    probs = arith_probs(args.op1_digs, args.op2_digs, n=args.N)
    str_probs = fewshot_probs(probs, k=args.fewshot_k)

    tokenized_input = toks(
        str_probs, return_offsets_mapping=True, padding=True, return_tensors="pt"
    )
    idxs, ans_ids = answer_logit_indices(tokenized_input, str_probs)
    idxs, ans_ids = pad_ans_idx_id(idxs, ans_ids)
    idxs, ans_ids = torch.LongTensor(idxs), torch.LongTensor(ans_ids)

    chunks = torch.chunk(torch.arange(args.N), args.N // args.batch_size)

    all_blocks = None
    all_clean_logits = None

    for i, batch in enumerate(chunks):
        clean_batch_ids = tokenized_input["input_ids"][batch]
        corr_batch_ids = tokenized_input["input_ids"][batch - 1]
        clean_pos, corr_pos = idxs[batch], idxs[batch - 1]  # logit answer indices
        clean_ans_ids, corr_ans_ids = (
            ans_ids[batch],
            ans_ids[batch],
        )  # token answer indices

        corr_logits = m(corr_batch_ids)
        _, clean_cache = m.run_with_cache(clean_batch_ids)

        metric = partial(
            logit_diff,
            clean_logits=None,
            corrupted_logits=corr_logits,
            clean_ans_pos=clean_pos,
            clean_ans_ids=clean_ans_ids,
        )

        attn_head_results = patching.get_act_patch_attn_head_pattern_all_pos(
            m, corr_batch_ids, clean_cache, metric
        ).to("cpu")

        if all_blocks is None:
            all_blocks = attn_head_results * len(batch)
        else:
            all_blocks += attn_head_results * len(batch)

    pickle.dump(
        str_probs,
        open(f"data/patching_circuit/{args.op1_digs}-{args.op2_digs}-probs.pkl", "wb"),
    )
    pickle.dump(
        (1 / args.N) * all_blocks,
        open(
            f"data/patching_circuit/{args.op1_digs}-{args.op2_digs}-all_blocks.pkl",
            "wb",
        ),
    )
