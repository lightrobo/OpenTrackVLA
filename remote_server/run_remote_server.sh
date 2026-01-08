#!/bin/bash

export HF_MODEL_DIR="/mnt/data/models/opentrackvla-qwen06b"
export DINOV3_MODEL_PATH="/mnt/data/models/vision_tower/dinov3-vits16"

echo "[Server] Using planner: ${HF_MODEL_DIR}"
echo "[Server] Using DINOv3: ${DINOV3_MODEL_PATH}"

CUDA_VISIBLE_DEVICES=0 python remote_server.py --port 50051

