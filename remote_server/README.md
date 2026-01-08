# 云端推理服务器 (Remote Server)

在云端开发机上运行的 gRPC 推理服务。

## 运行环境

- 云端开发机（有 GPU）
- 需要完整的 OpenTrackVLA 项目代码
- PyTorch + CUDA

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成 gRPC 代码

```bash
chmod +x generate_grpc.sh
./generate_grpc.sh
```

### 3. 启动服务

```bash
python remote_server.py --model-dir /path/to/model --port 50051
```

## 命令行参数

```
--model-dir     模型目录路径（必需）
--port          gRPC端口，默认 50051
--device        计算设备，默认 cuda
--history       历史帧数量，默认 31
--workers       工作线程数，默认 4
```

## 示例

```bash
# 使用 GPU
python remote_server.py --model-dir ./checkpoints/opentrackvla --port 50051

# 使用 CPU（调试用）
python remote_server.py --model-dir ./checkpoints/opentrackvla --device cpu
```

## 日志输出

```
[Server] 使用设备: cuda
[Server] 加载模型: ./checkpoints/opentrackvla
[Server] OpenTrackVLA模型加载完成
[Server] Vision encoder初始化完成
[Server] gRPC服务启动: 0.0.0.0:50051
[Server] 等待连接...
```

## 确保端口可访问

SSH 隧道会连接到这个端口，确保防火墙允许本地连接：

```bash
# 检查端口是否在监听
netstat -tlnp | grep 50051
```

