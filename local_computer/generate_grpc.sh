#!/bin/bash
# 从 proto 文件生成 Python gRPC 代码

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "生成 gRPC Python 代码..."

python -m grpc_tools.protoc \
    --proto_path=. \
    --python_out=. \
    --grpc_python_out=. \
    inference.proto

echo "生成完成!"
echo "  - inference_pb2.py"
echo "  - inference_pb2_grpc.py"

