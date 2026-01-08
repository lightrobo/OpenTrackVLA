# 本地电脑 (Local Computer)

SSH 隧道中转 + 测试客户端。在你的本地电脑上运行。

## 运行环境

- 你的本地电脑（Mac/Linux/Windows WSL）
- 需要能 SSH 到云端开发机

## 快速开始

### 1. 配置 SSH 隧道

编辑 `run_ssh_tunneling.sh`，修改配置：

```bash
REMOTE_USER="your_username"          # 云端用户名
REMOTE_HOST="your_cloud_server.com"  # 云端服务器地址
```

### 2. 启动 SSH 隧道

```bash
chmod +x run_ssh_tunneling.sh
./run_ssh_tunneling.sh
```

保持这个终端开着，隧道会一直工作。

### 3.（可选）测试连接

另开一个终端：

```bash
# 安装依赖
pip install -r requirements.txt

# 生成 gRPC 代码
chmod +x generate_grpc.sh
./generate_grpc.sh

# 测试连接
python local_client.py --server localhost:50051
```

## 网络拓扑

```
AGX Orin (192.168.1.50)
    │
    │ TCP → 连接到本地电脑的 50051 端口
    ▼
本地电脑 (192.168.1.100:50051)
    │
    │ SSH 隧道 → 转发到云端的 50051 端口
    ▼
云端开发机 (localhost:50051)
```

## 测试命令

### 测试 本地电脑 → 云端（在本地电脑上执行）

验证 SSH 隧道是否正常工作：

```bash
# 健康检查
python local_client.py --server localhost:50051

# 延迟测试（10次往返）
python local_client.py --server localhost:50051 --latency-test

# 测试单张图片推理
python local_client.py --server localhost:50051 --image test.jpg

# 测试视频推理
python local_client.py --server localhost:50051 --video test.mp4 --max-frames 50
```

### 测试 AGX → 本地电脑 → 云端（在 AGX 上执行）

验证完整链路是否正常：

```bash
# 健康检查（192.168.1.100 替换为你本地电脑的IP）
python local_client.py --server 192.168.1.100:50051

# 延迟测试
python local_client.py --server 192.168.1.100:50051 --latency-test
```

### 参数说明

| 参数 | 作用 |
|-----|-----|
| `--server` | gRPC 服务器地址（默认 localhost:50051） |
| `--image` | 测试单张图片推理 |
| `--video` | 测试视频流推理 |
| `--latency-test` | 运行 10 次往返延迟测试 |
| `--max-frames` | 视频测试最大帧数（默认 100） |

## 确保 AGX 能连接到本地电脑

AGX Orin 需要能访问本地电脑的 50051 端口：

```bash
# 在本地电脑上，确保防火墙允许 50051 端口
# macOS:
sudo pfctl -d  # 关闭防火墙（测试用）

# Linux:
sudo ufw allow 50051/tcp
```

## 故障排查

1. **SSH 隧道连接失败**：检查云端服务器地址和 SSH 密钥
2. **端口被占用**：`lsof -i :50051` 查看占用进程
3. **AGX 连不上**：检查本地电脑和 AGX 是否在同一网络

