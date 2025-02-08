#!/bin/bash

export MODEL="meta-llama/Llama-3.1-8B"
export MNAME="llama-8b"
export PROMPT_ROOT="experiments/data/baselines-cot"
export ROOT="experiments/data/$MODEL-prompting"
export TASK="csense"
export BATCH_SIZE=2
export N=2

mkdir -p $ROOT

scalene --- -m experiments.prompting $MODEL "$ROOT/$MNAME-$TASK-zero" --batch_size $BATCH_SIZE \
    --response_name "$PROMPT_ROOT/$MNAME-$TASK-zero-benchmark.pkl" --ndevices 2 --n $N

# python3 -m experiments.prompting $MODEL "$ROOT/$MNAME-$TASK-cot" --batch_size $BATCH_SIZE \
#     --response_name "$PROMPT_ROOT/$MNAME-$TASK-cot-benchmark.pkl" --ndevices 2 --n $N

# python3 -m experiments.prompting $MODEL "$ROOT/$MNAME-$TASK-fs" --batch_size $BATCH_SIZE \
#     --response_name "$PROMPT_ROOT/$MNAME-$TASK-fs-benchmark.pkl" --ndevices 2 --n $N

# python3 -m experiments.prompting $MODEL "$ROOT/$MNAME-$TASK-fsm" --batch_size $BATCH_SIZE \
#     --response_name "$PROMPT_ROOT/$MNAME-$TASK-fsm-benchmark.pkl" --ndevices 2 --n $N
