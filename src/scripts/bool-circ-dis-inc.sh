#!/bin/bash

for expl in {1..9}; do
  for depth in {1..6}; do
    python3 -m logic.bool_circuit_discovery meta-llama/Llama-3.2-1B "logic/data/kl-llama-1b-bool-$expl$depth" --allow_parentheses \
      --batch_size 4 --exp_length $expl --depth $depth --num 1000
  done
done
