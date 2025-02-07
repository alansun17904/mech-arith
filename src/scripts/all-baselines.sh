#!/bin/bash

export MODEL="meta-llama/Llama-3.2-1B-Instruct"
export MNAME="llama-1b"
export ROOT="experiments/data/baselines-cot"

mkdir -p $ROOT

# sports understanding
python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset sports --format "few-shot" \
    --data_params n=1000 --format_params shots=3

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset sports --format "chain-of-thought" \
    --data_params n=1000

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset sports --format "zero-shot" \
    --data_params n=1000

# date understanding
python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset date --format "few-shot" \
    --data_params n=1000 --format_params shots=3

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset date --format "chain-of-thought" \
    --data_params n=1000

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset date --format "zero-shot" \
    --data_params n=1000

# movie recommendation
python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset movie --format "few-shot" \
    --data_params n=1000 --format_params shots=3

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset movie --format "chain-of-thought" \
    --data_params n=1000

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset movie --format "zero-shot" \
    --data_params n=1000

# dyck language completion
python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset dyck --format "few-shot" \
    --data_params n=1000 max_length=7 --format_params shots=3

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset dyck --format "chain-of-thought" \
    --data_params n=1000 max_length=7

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset dyck --format "zero-shot" \
    --data_params n=1000 max_length=7

# common sense reasoning
python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset csense --format "few-shot" \
    --data_params n=1000 --format_params shots=3

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset csense --format "chain-of-thought" \
    --data_params n=1000

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME" --dataset csense --format "zero-shot" \
    --data_params n=1000
