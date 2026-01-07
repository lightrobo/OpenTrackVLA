#!/bin/bash
# 先杀启动脚本（防止它们重新拉起 python 进程）
pkill -9 -f "stt_data.*\.sh"
pkill -9 -f "data_collect.*\.sh"

# 再杀 python 进程
pkill -9 -f "python run\.py"
pkill -9 -f "python run_oracle\.py"
pkill -9 -f "python run_improved\.py"

echo "Done."

