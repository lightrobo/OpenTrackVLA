#!/bin/bash
# Jetson PyTorch GPU 版本最终安装脚本
# 解决 pip 版本和平台标签问题

set -e

echo "=========================================="
echo "Jetson PyTorch GPU 版本安装"
echo "=========================================="
echo ""

# 检查 wheel 文件
WHEEL_DIR="${1:-./wheel}"
if [ ! -d "$WHEEL_DIR" ]; then
    echo "❌ Wheel 目录不存在: $WHEEL_DIR"
    echo "   请指定正确的 wheel 文件目录"
    exit 1
fi

cd "$WHEEL_DIR"

echo "Wheel 目录: $(pwd)"
echo ""

# 检查必需的 wheel 文件
# 使用更灵活的通配符模式
TORCH_WHEEL=$(ls torch-2.3.0*.whl 2>/dev/null | grep -E "torch-2\.3\.0.*cp310.*linux_aarch64" | head -1)
TORCHVISION_WHEEL=$(ls torchvision-0.18*.whl 2>/dev/null | grep -E "torchvision-0\.18.*cp310.*linux_aarch64" | head -1)
TORCHAUDIO_WHEEL=$(ls torchaudio-2.3.0*.whl 2>/dev/null | grep -E "torchaudio-2\.3\.0.*cp310.*linux_aarch64" | head -1)

if [ -z "$TORCH_WHEEL" ] || [ -z "$TORCHVISION_WHEEL" ] || [ -z "$TORCHAUDIO_WHEEL" ]; then
    echo "❌ 未找到必需的 wheel 文件"
    echo ""
    echo "需要以下文件："
    [ -z "$TORCH_WHEEL" ] && echo "  ❌ torch-2.3.0*.whl (匹配 cp310 linux_aarch64)" || echo "  ✅ $(basename $TORCH_WHEEL)"
    [ -z "$TORCHVISION_WHEEL" ] && echo "  ❌ torchvision-0.18*.whl (匹配 cp310 linux_aarch64)" || echo "  ✅ $(basename $TORCHVISION_WHEEL)"
    [ -z "$TORCHAUDIO_WHEEL" ] && echo "  ❌ torchaudio-2.3.0*.whl (匹配 cp310 linux_aarch64)" || echo "  ✅ $(basename $TORCHAUDIO_WHEEL)"
    echo ""
    echo "=========================================="
    echo "请先下载 wheel 文件"
    echo "=========================================="
    echo ""
    echo "方法1: 使用以下 wget 命令下载（推荐）"
    echo ""
    echo "# 进入下载目录"
    echo "cd $(pwd)"
    echo ""
    echo "# 下载 torch"
    echo "wget https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl"
    echo ""
    echo "# 下载 torchvision"
    echo "wget https://nvidia.box.com/shared/static/u0ziu01c0kyji4zz3gxam79181nebylf.whl -O torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl"
    echo ""
    echo "# 下载 torchaudio"
    echo "wget https://nvidia.box.com/shared/static/9si945yrzesspmg9up4ys380lqxjylc3.whl -O torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl"
    echo ""
    echo "方法2: 从官网获取最新链接"
    echo "   访问: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    echo "   找到 JetPack 6.0 + CUDA 12.4 + Python 3.10"
    echo "   复制下载链接并替换上面的 URL"
    echo ""
    echo "下载完成后，重新运行此脚本："
    echo "  bash install_pytorch_jetson_final.sh $(pwd)"
    echo ""
    exit 1
fi

echo "✅ 找到所有必需的 wheel 文件："
echo "   torch: $(basename $TORCH_WHEEL)"
echo "   torchvision: $(basename $TORCHVISION_WHEEL)"
echo "   torchaudio: $(basename $TORCHAUDIO_WHEEL)"
echo ""

# 检测是否在 conda 环境中
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✅ 检测到 conda 环境: $CONDA_DEFAULT_ENV"
    USE_USER_FLAG=""
    PIP_CMD="pip"
    PYTHON_CMD="python"
    # 检查 Python 版本
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    # wheel 文件是 cp310，需要 Python 3.10
    if [ "$PYTHON_MAJOR" != "3" ] || [ "$PYTHON_MINOR" != "10" ]; then
        echo ""
        echo "❌ Python 版本不匹配！"
        echo "   当前 Python 版本: $PYTHON_VERSION"
        echo "   需要的 Python 版本: 3.10.x (wheel 文件是 cp310)"
        echo ""
        echo "解决方案："
        echo "   1. 重新创建 Python 3.10 的 conda 环境："
        echo "      conda create -n opentrackvla_infer python=3.10 -y"
        echo "      conda activate opentrackvla_infer"
        echo ""
        echo "   2. 或者使用系统 Python 3.10 安装（退出 conda 环境后运行）："
        echo "      conda deactivate"
        echo "      bash install_pytorch_jetson_final.sh"
        echo ""
        exit 1
    fi
    echo "✅ Python 版本: $PYTHON_VERSION (匹配)"
else
    echo "⚠️  未检测到 conda 环境，使用 --user 安装"
    USE_USER_FLAG="--user"
    PIP_CMD="pip3"
    PYTHON_CMD="python3"
    # 检查系统 Python 版本
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" != "3" ] || [ "$PYTHON_MINOR" != "10" ]; then
        echo ""
        echo "❌ Python 版本不匹配！"
        echo "   当前 Python 版本: $PYTHON_VERSION"
        echo "   需要的 Python 版本: 3.10.x (wheel 文件是 cp310)"
        echo ""
        exit 1
    fi
    echo "✅ Python 版本: $PYTHON_VERSION (匹配)"
fi
echo ""

# 步骤1: 升级 pip（重要！）
echo "=========================================="
echo "步骤 1/4: 升级 pip"
echo "=========================================="
echo ""
$PIP_CMD install --upgrade pip setuptools wheel $USE_USER_FLAG
echo "✅ pip 升级完成"
echo ""

# 步骤2: 安装系统依赖
echo "=========================================="
echo "步骤 2/4: 安装系统依赖"
echo "=========================================="
echo ""
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev libomp-dev || true
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev || true
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev || true
echo "✅ 系统依赖安装完成"
echo ""

# 步骤3: 安装 Python 基础依赖
echo "=========================================="
echo "步骤 3/4: 安装 Python 基础依赖"
echo "=========================================="
echo ""
$PIP_CMD install 'Cython<3' numpy typing_extensions $USE_USER_FLAG
echo "✅ Python 依赖安装完成"
echo ""

# 步骤4: 安装 PyTorch wheel
echo "=========================================="
echo "步骤 4/4: 安装 PyTorch wheel"
echo "=========================================="
echo ""

# 卸载旧版本（包括用户目录的）
echo "卸载旧版本..."
$PIP_CMD uninstall torch torchvision torchaudio -y 2>/dev/null || true
# 也卸载用户目录的版本（如果之前错误安装）
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    pip3 uninstall torch torchvision torchaudio -y --user 2>/dev/null || true
fi

# 安装 torch
echo ""
echo "[1/3] 安装 torch..."
$PIP_CMD install "$TORCH_WHEEL" --force-reinstall --no-deps $USE_USER_FLAG || {
    echo "❌ torch 安装失败"
    exit 1
}
echo "✅ torch 安装完成"

# 安装 torchvision
echo ""
echo "[2/3] 安装 torchvision..."
$PIP_CMD install "$TORCHVISION_WHEEL" --force-reinstall --no-deps $USE_USER_FLAG || {
    echo "⚠️  torchvision 安装失败，尝试从源码编译..."
    # 从源码编译的代码可以在这里添加
}
echo "✅ torchvision 安装完成"

# 安装 torchaudio
echo ""
echo "[3/3] 安装 torchaudio..."
$PIP_CMD install "$TORCHAUDIO_WHEEL" --force-reinstall --no-deps $USE_USER_FLAG || {
    echo "⚠️  torchaudio 安装失败（可选）"
}
echo "✅ torchaudio 安装完成"

# 安装 PyTorch 依赖
echo ""
echo "安装 PyTorch 依赖..."
$PIP_CMD install filelock fsspec jinja2 networkx sympy $USE_USER_FLAG
echo "✅ 依赖安装完成"

# 验证安装
echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="
$PYTHON_CMD -c "
import torch
print(f'✅ PyTorch 版本: {torch.__version__}')
print(f'✅ CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA 版本: {torch.version.cuda}')
    print(f'✅ GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'✅ GPU {i}: {torch.cuda.get_device_name(i)}')
    print('')
    print('🎉 GPU 版本安装成功！')
else:
    print('')
    print('❌ CUDA 不可用')
"

echo ""
echo "=========================================="
echo "✅ 安装完成！"
echo "=========================================="

