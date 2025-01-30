#!/bin/bash

for expl in {3..9}; do
  for depth in {1..6}; do
    python3 -m logic.bool_pareto phi-1_5 "logic/data/phi-1_5-bool-$expl$depth.json" "logic/data/phi-1_5-bool-$expl$depth" --allow_parentheses \
      --batch_size 32 --exp_length $expl --depth $depth --num 1000
  done
done
