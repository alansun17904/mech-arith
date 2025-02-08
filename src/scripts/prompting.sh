#!/bin/bash

export MODEL="meta-llama/Llama-3.1-8B"
export MNAME="llama-8b"
export PROMPT_ROOT="experiments/data/baselines-cot"
export ROOT="experiments/data/$MODEL-prompting"

mkdir -p $ROOT

python3 -m experiments.prompting $MODEL "$ROOT/$MNAME-sports-zero" --batch_size 4 \
    --response_name "$PROMPT_ROOT/$MNAME-sports-zero-benchmark.pkl"

python3 -m experiments.prompting $MODEL "$ROOT/$MNAME-sports-cot" --batch_size 4 \
    --response_name "$PROMPT_ROOT/$MNAME-sports-cot-benchmark.pkl"

python3 -m experiments.prompting $MODEL "$ROOT/$MNAME-sports-fs" --batch_size 4 \
    --response_name "$PROMPT_ROOT/$MNAME-sports-fs-benchmark.pkl"

python3 -m experiments.prompting $MODEL "$ROOT/$MNAME-sports-fsm" --batch_size 4 \
    --response_name "$PROMPT_ROOT/$MNAME-sports-fsm-benchmark.pkl"