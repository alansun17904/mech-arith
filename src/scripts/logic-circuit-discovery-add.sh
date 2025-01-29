#!/bin/bash

# Loop through all values of opr1, opr2 in [1,8]
for opr1 in {1..8}; do
  for opr2 in {4..8}; do
    python3 -m logic.circuit_discovery gemma-2-2b \
    	"logic/data/gemma-add-$opr1$opr2" ADD $opr1 $opr2 --batch_size 4 --num 1000
  done
done
