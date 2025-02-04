#!/bin/bash

for expl in {1..9}; do
    python3 -m logic.bool_circuit_discovery phi-1_5 "phi-only-not-$expl" --no_or --no_and --seed $expl
done

for expl in {1..9}; do
    python3 -m logic.bool_circuit_discovery phi-1_5 "phi-not-and-$expl" --no-or --seed $expl
done

for expl in {1..9}; do
    python3 -m logic.bool_circuit_discovery phi-1_5 "phi-all-no-paren-$expl" --seed $expl
done
