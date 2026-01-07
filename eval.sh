CHUNKS=15
NUM_PARALLEL=1
SAVE_PATH="sim_data/eval/stt"

# Set model paths
export HF_MODEL_DIR="/mnt/data/models/opentrackvla-qwen06b"
export DINOV3_MODEL_PATH="/mnt/data/models/vision_tower/dinov3-vits16"

echo "[eval] Using planner: ${HF_MODEL_DIR}"
echo "[eval] Using DINOv3: ${DINOV3_MODEL_PATH}"

IDX=0
while [ $IDX -lt $CHUNKS ]; do
    for ((i = 0; i < NUM_PARALLEL && IDX < CHUNKS; i++)); do
        echo "Launching job IDX=$IDX on GPU=$((IDX % NUM_PARALLEL))"
        #CUDA_VISIBLE_DEVICES=$((i)) SAVE_VIDEO=1 PYTHONPATH="habitat-lab" python run_eval.py \
        CUDA_VISIBLE_DEVICES=1 SAVE_VIDEO=1 PYTHONPATH="habitat-lab" python run_eval.py \
            --split-num $CHUNKS \
            --split-id $IDX \
            --exp-config 'habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml' \
            --run-type 'eval' \
            --save-path $SAVE_PATH &
        ((IDX++))
    done
    wait
done
