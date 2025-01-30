#!/bin/bash

# python3 -m logic.baseline_bool phi-1_5 phi-1_5-only-not --no_and --no_or \
# 	--allow_parentheses --num 1000 --depth 6

# python3 -m logic.baseline_bool phi-1_5 phi-1_5-only-and --no_not --no_or \
# 	--allow_parentheses --num 1000 --depth 6

# python3 -m logic.baseline_bool phi-1_5 phi-1_5-only-or --no_not --no_and \
# 	--allow_parentheses --num 1000 --depth 6

python3 -m logic.baseline_bool phi-1_5 no-paran-phi-1_5-only-not --no_and --no_or \
	--num 1000

python3 -m logic.baseline_bool phi-1_5 no-paran-phi-1_5-only-and --no_not --no_or \
	--num 1000

python3 -m logic.baseline_bool phi-1_5 no-paran-phi-1_5-only-or --no_not --no_and \
	--num 1000

# python3 logic.baseline_bool phi-1_5 phi-1_5-not-and --no_and --no_or \
# 	--allow_parentheses --num 1000 --depth 6

# python3 logic.baseline_bool phi-1_5 phi-1_5-only-not --no_not --no_or \
# 	--allow_parentheses --num 1000 --depth 6

# python3 logic.baseline_bool phi-1_5 phi-1_5-only-not --no_not --no_and \
# 	--allow_parentheses --num 1000 --depth 6
