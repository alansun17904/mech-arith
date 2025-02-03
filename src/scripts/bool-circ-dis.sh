#!/bin/bash

for expl in $(seq 9 -1 3); do
  for depth in {1..6}; do
    python3 -m logic.bool_circuit_discovery meta-llama/Llama-3.2-1B "logic/data/kl-llama-1b-bool-$expl$depth" --allow_parentheses \
  done
done
