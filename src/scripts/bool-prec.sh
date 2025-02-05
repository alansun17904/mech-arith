#!/bin/bash

for seed in {1..9}; do
    python3 -m logic.bool_circuit_discovery phi-1_5 "logic/phi-only-not-$seed" --no_or --no_and --seed $seed --batch_size 8
done

for seed in {1..9}; do
    python3 -m logic.bool_circuit_discovery phi-1_5 "logic/data/phi-not-and-$seed" --no_or --seed $seed --batch_size 8
done

for seed in {1..9}; do
    python3 -m logic.bool_circuit_discovery phi-1_5 "logic/data/phi-all-no-paren-$seed" --seed $seed --batch_size 8
done
