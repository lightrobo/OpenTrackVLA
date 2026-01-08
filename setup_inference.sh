#!/bin/bash
# Minimal inference environment setup for OpenTrackVLA
# No habitat-sim required - perfect for ARM64 platforms!

set -e

ENV_NAME="${1:-opentrackvla_infer}"
PYTHON_VERSION="${2:-3.9}"

echo "=========================================="
echo "OpenTrackVLA 最小化推理环境安装"
echo "=========================================="
echo ""
echo "环境名称: $ENV_NAME"
echo "Python 版本: $PYTHON_VERSION"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda 命令"
    echo "请先安装 Miniconda 或 Anaconda"
    exit 1
fi

# Create conda environment
echo "[1/4] 创建 Conda 环境..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -c conda-forge -y

# Activate environment
echo "[2/4] 激活环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Detect CUDA version and install PyTorch
echo "[3/4] 安装 PyTorch..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "检测到 CUDA 版本: $CUDA_VERSION"
    
    # Map CUDA version to PyTorch index
    if [[ $(echo "$CUDA_VERSION >= 12.0" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        PT_INDEX="cu121"
    elif [[ $(echo "$CUDA_VERSION >= 11.8" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        PT_INDEX="cu118"
    else
        PT_INDEX="cu118"  # Default to 11.8
    fi
    
    echo "安装 PyTorch (CUDA $PT_INDEX)..."
    # 安装 PyTorch 2.1+ (transformers 要求)
    pip install torch>=2.1.0 torchvision torchaudio --index-url "https://download.pytorch.org/whl/$PT_INDEX"
else
    echo "未检测到 CUDA，安装 CPU 版本 PyTorch..."
    # 安装 PyTorch 2.1+ (transformers 要求)
    pip install "torch>=2.1.0" torchvision torchaudio
fi

# Install inference dependencies
echo "[4/4] 安装推理依赖..."
if [ -f "requirements_inference.txt" ]; then
    pip install -r requirements_inference.txt
else
    echo "警告: requirements_inference.txt 不存在，使用默认依赖..."
    pip install transformers>=4.30.0 opencv-python>=4.5.0 Pillow>=9.0.0 "numpy>=1.20.0,<1.24.0" huggingface-hub>=0.16.0
fi

# Verify installation
echo ""
echo "=========================================="
echo "验证安装..."
python -c "import torch; import transformers; import cv2; print('✅ 所有依赖安装成功!')" || {
    echo "❌ 验证失败，请检查错误信息"
    exit 1
}

# Note about project path
echo ""
echo "=========================================="
echo "✅ 安装完成!"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  conda activate $ENV_NAME"
echo "  cd $(pwd)  # 确保在项目根目录"
echo "  python inference/run_inference_agx.py --model_dir <模型路径>"
echo ""
echo "注意: 脚本会自动添加项目根目录到 Python 路径，"
echo "      但建议在项目根目录下运行脚本。"
echo ""

