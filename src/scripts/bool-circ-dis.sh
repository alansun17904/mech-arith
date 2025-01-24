#!/bin/bash

# Loop through all values of opr1, opr2 in [1,8]
for expl in {3..9}; do
  for depth in {1..6}; do
    python3 -m logic.bool_circuit_discovery "phi-1_5 logic/data/phi-1_5-bool-$expl$depth" --allow_parentheses \
      --batch_size 8 --exp_length $expl --depth $depth --num 1000
  done
done
