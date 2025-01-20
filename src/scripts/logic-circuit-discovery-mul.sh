#!/bin/bash

# Loop through all values of opr1, opr2 in [1,8]
for opr1 in {1..4}; do
  for opr2 in {1..4}; do
    python3 -m logic.circuit_discovery meta-llama/Llama-3.2-1B \
    	"meta-llama-add-$opr1$opr2" MUL $opr1 $opr2 --batch_size 16 --num 1000
  done
done
