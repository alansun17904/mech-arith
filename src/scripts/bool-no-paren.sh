#!/bin/bash

mkdir -p "experiments/data/bool-no-paren"
export ROOT="experiments/data/bool-no-paren"
export MODEL="phi-1_5"
export BATCH_SIZE=8
export N=1000

# No parentheses
for expl in {3..9}; do
    python3 -m experiments.baseline $MODEL "$ROOT/$MODEL-all-$expl" --dataset bool --format "few-shot" \
    --data_params expression_lengths=$expl allow_parentheses=False variable_length=True binary_ops='("and", "or")' \
    n=$N --format_params shots=3 --batch_size $BATCH_SIZE
done

# # No parentheses
# python3 -m experiments.baseline $MODEL "$ROOT/$MODEL-only-not" --dataset bool --format "few-shot" \
# --data_params expression_lengths=9 allow_parentheses=False variable_length=True binary_ops='tuple()' \
# n=$N --format_params shots=3 --batch_size $BATCH_SIZE

# # No parentheses
# python3 -m experiments.baseline $MODEL "$ROOT/$MODEL-and-not" --dataset bool --format "few-shot" \
# --data_params expression_lengths=9 allow_parentheses=False variable_length=True binary_ops='("and",)' \
# n=$N --format_params shots=3 --batch_size $BATCH_SIZE

# # No parentheses
# python3 -m experiments.baseline $MODEL "$ROOT/$MODEL-not-or" --dataset bool --format "few-shot" \
# --data_params expression_lengths=9 allow_parentheses=False variable_length=True binary_ops='("or",)' \
# n=$N --format_params shots=3 --batch_size $BATCH_SIZE
