#!/bin/bash

# Loop through all values of opr1, opr2 in [1,8]
# for opr1 in {1..8}; do
#   for opr2 in {1..8}; do
#     python3 -m logic.circuit_discovery meta-llama/llama-3.2-1b \
#     	"logic/data/meta-llama-add-$opr1$opr2" add $opr1 $opr2 --batch_size 16 --num 1000
#   done
# done

python3 -m logic.circuit_discovery meta-llama/llama-3.2-1b \
    "logic/data/meta-llama-add-33" ADD 3 3 --batch_size 16 --num 1000
python3 -m logic.circuit_discovery meta-llama/llama-3.2-1b \
    "logic/data/meta-llama-add-31" ADD 3 1 --batch_size 16 --num 1000
python3 -m logic.circuit_discovery meta-llama/llama-3.2-1b \
    "logic/data/meta-llama-add-32" ADD 3 2 --batch_size 16 --num 1000
python3 -m logic.circuit_discovery meta-llama/llama-3.2-1b \
    "logic/data/meta-llama-add-36" ADD 3 6 --batch_size 16 --num 1000
python3 -m logic.circuit_discovery meta-llama/llama-3.2-1b \
    "logic/data/meta-llama-add-46" ADD 4 6 --batch_size 16 --num 1000
python3 -m logic.circuit_discovery meta-llama/llama-3.2-1b \
    "logic/data/meta-llama-add-47" ADD 4 7 --batch_size 16 --num 1000
python3 -m logic.circuit_discovery meta-llama/llama-3.2-1b \
    "logic/data/meta-llama-add-48" ADD 4 8 --batch_size 16 --num 1000
python3 -m logic.circuit_discovery meta-llama/llama-3.2-1b \
    "logic/data/meta-llama-add-56" ADD 5 6 --batch_size 16 --num 1000
python3 -m logic.circuit_discovery meta-llama/llama-3.2-1b \
    "logic/data/meta-llama-add-63" ADD 6 3 --batch_size 16 --num 1000
python3 -m logic.circuit_discovery meta-llama/llama-3.2-1b \
    "logic/data/meta-llama-add-64" ADD 6 4 --batch_size 16 --num 1000
python3 -m logic.circuit_discovery meta-llama/llama-3.2-1b \
    "logic/data/meta-llama-add-65" ADD 6 5 --batch_size 16 --num 1000
python3 -m logic.circuit_discovery meta-llama/llama-3.2-1b \
    "logic/data/meta-llama-add-66" ADD 6 6 --batch_size 16 --num 1000

