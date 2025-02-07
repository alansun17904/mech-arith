#!/bin/bash

export MODEL="meta-llama/Llama-3.2-1B-Instruct"
export MNAME="llama-1b"
export ROOT="experiments/data/baselines-cot"

mkdir -p $ROOT

# sports understanding
python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-sports-fs" --dataset sports --format "few-shot" \
    --data_params n=1000 --format_params shots=3

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-sports-cot" --dataset sports --format "chain-of-thought" \
    --data_params n=1000

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-sports-zero" --dataset sports --format "zero-shot" \
    --data_params n=1000

# date understanding
python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-date-fs" --dataset date --format "few-shot" \
    --data_params n=1000 --format_params shots=3

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-date-cot" --dataset date --format "chain-of-thought" \
    --data_params n=1000

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-date-zero" --dataset date --format "zero-shot" \
    --data_params n=1000

# movie recommendation
python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-movie-fs" --dataset movie --format "few-shot" \
    --data_params n=1000 --format_params shots=3

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-movie-cot" --dataset movie --format "chain-of-thought" \
    --data_params n=1000

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-movie-zero" --dataset movie --format "zero-shot" \
    --data_params n=1000

# dyck language completion
python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-dyck-fs" --dataset dyck --format "few-shot" \
    --data_params n=1000 max_length=7 --format_params shots=3

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-dyck-cot" --dataset dyck --format "chain-of-thought" \
    --data_params n=1000 max_length=7

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-dyck-zero" --dataset dyck --format "zero-shot" \
    --data_params n=1000 max_length=7

# common sense reasoning
python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-csense-fs" --dataset csense --format "few-shot" \
    --data_params n=1000 --format_params shots=3

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-csense-cot" --dataset csense --format "chain-of-thought" \
    --data_params n=1000

python3 -m experiments.baseline $MODEL "$ROOT/$MNAME-csense-zero" --dataset csense --format "zero-shot" \
    --data_params n=1000
