#!/bin/bash

# Oracle baseline agent for collecting expert training data
# Uses ground truth positions for optimal behavior

CHUNKS=30
NUM_PARALLEL=2
SEED=101
SAVE_PATH="sim_data/stt_train_oracle/seed_${SEED}"
IDX=0

echo "=== Oracle Agent Data Collection ==="
echo "Save path: $SAVE_PATH"
echo "Seed: $SEED"

while [ $IDX -lt $CHUNKS ]; do
    for ((i = 0; i < NUM_PARALLEL && IDX < CHUNKS; i++)); do
        echo "Launching job IDX=$IDX on GPU=$((IDX % NUM_PARALLEL))"
        CUDA_VISIBLE_DEVICES=$((i)) SAVE_VIDEO=1 PYTHONPATH="habitat-lab" python run_oracle.py \
            --split-num $CHUNKS \
            --split-id $IDX \
            --exp-config 'habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml' \
            --run-type 'eval' \
            --save-path $SAVE_PATH \
            habitat.simulator.seed=${SEED} &
        ((IDX++))
    done
    wait
done
