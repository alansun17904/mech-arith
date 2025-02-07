#!/bin/bash

mkdir -p "experiments/data/bool-no-paren"
export ROOT="experiments/data/bool-no-paren"
export MODEL="phi-1_5"
export BATCH_SIZE=8
export N=1000

# No parentheses
for expl in {1..9}; do
    python3 -m experiments.baseline $MODEL "$ROOT/$MODEL-all-$expl" --dataset bool --format "few-shot" \
    --data_params expression_lengths=$expl allow_parentheses=False variable_length=False binary_ops='("and", "or")' \
    n=$N --format_params shots=3 --batch_size $BATCH_SIZE
done

# No parentheses
for expl in {1..9}; do
    python3 -m experiments.baseline $MODEL "$ROOT/$MODEL-only-not-$expl" --dataset bool --format "few-shot" \
    --data_params expression_lengths=$expl allow_parentheses=False variable_length=False binary_ops='tuple()' \
    n=$N --format_params shots=3 --batch_size $BATCH_SIZE
done

# No parentheses
for expl in {1..9}; do
    python3 -m experiments.baseline $MODEL "$ROOT/$MODEL-and-or-$expl" --dataset bool --format "few-shot" \
    --data_params expression_lengths=$expl allow_parentheses=False variable_length=False binary_ops='("or", "and")' \
    n=$N --format_params shots=3 --batch_size $BATCH_SIZE
done