#!/bin/bash

mkdir -p "experiments/data/precedence"
export ROOT="experiments/data/precedence"
export MODEL="gpt2"

# No parentheses
for seed in {1..9}; do
    python3 -m experiments.circuit_discovery $MODEL "$ROOT/$MODEL-only-not-$seed" --dataset bool --format "few-shot" \
    --data_params expression_lengths=7 allow_parentheses=False variable_length=True binary_ops='tuple()' \
    parenthetical_depth=3 n=100 --format_params shots=3 --batch_size 8
done

for seed in {1..9}; do
    python3 -m experiments.circuit_discovery $MODEL "$ROOT/$MODEL-and-not-$seed" --dataset bool --format "few-shot" \
    --data_params expression_lengths=7 allow_parentheses=False variable_length=True binary_ops='("and",)' \
    parenthetical_depth=3 n=100 --format_params shots=3 --batch_size 8

for seed in {1..9}; do
    python3 -m experiments.circuit_discovery $MODEL "$ROOT/$MODEL-all-$seed" --dataset bool --format "few-shot" \
    --data_params expression_lengths=7 allow_parentheses=False variable_length=True binary_ops='("and", "or")' \
    parenthetical_depth=3 n=100 --format_params shots=3 --batch_size 8
done

# With parentheses
for seed in {1..9}; do
    python3 -m experiments.circuit_discovery $MODEL "$ROOT/$MODEL-only-not-np-$seed" --dataset bool --format "few-shot" \
    --data_params expression_lengths=7 allow_parentheses=False variable_length=True binary_ops='tuple()' \
    allow_parenthese=False n=100 --format_params shots=3 --batch_size 8
done

for seed in {1..9}; do
    python3 -m experiments.circuit_discovery $MODEL "$ROOT/$MODEL-and-not-np-$seed" --dataset bool --format "few-shot" \
    --data_params expression_lengths=7 allow_parentheses=False variable_length=True binary_ops='("and",)' \
    --allow_parentheses=False n=100 --format_params shots=3 --batch_size 8
done

for seed in {1..9}; do
    python3 -m experiments.circuit_discovery $MODEL "$ROOT/$MODEL-all-np-$seed" --dataset bool --format "few-shot" \
    --data_params expression_lengths=7 allow_parentheses=False variable_length=True binary_ops='("and", "or")' \
    allow_parentheses=False n=100 --format_params shots=3 --batch_size 8
done