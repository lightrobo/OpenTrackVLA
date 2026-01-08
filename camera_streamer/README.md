# AGX Orin 摄像头采集端 (Camera Streamer)

在 AGX Orin 上运行的摄像头采集 + gRPC 客户端。

## 运行环境

- NVIDIA Jetson AGX Orin
- USB 摄像头
- 能连接到本地电脑的网络

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

### 3. 启动摄像头采集

```bash
# 连接到本地电脑（本地电脑运行 SSH 隧道）
python camera_streamer.py --server 192.168.1.100:50051 --camera 0

# 或者直接连接云端（如果有公网IP）
python camera_streamer.py --server cloud.example.com:50051
```

## 命令行参数

```
--server        gRPC服务器地址，默认 localhost:50051
--camera        摄像头索引，默认 0
--width         图像宽度，默认 640
--height        图像高度，默认 480
--fps           目标帧率，默认 10
--quality       JPEG质量 (1-100)，默认 80
--instruction   文本指令，默认 "Follow the person"
--no-display    不显示画面（headless模式）
```

## 示例

```bash
# 基础运行
python camera_streamer.py --server 192.168.1.100:50051

# 调整参数
python camera_streamer.py \
    --server 192.168.1.100:50051 \
    --camera 0 \
    --width 640 \
    --height 480 \
    --fps 5 \
    --quality 70 \
    --instruction "Follow the person"

# 无显示模式（SSH 运行时）
python camera_streamer.py --server 192.168.1.100:50051 --no-display
```

## 键盘控制

运行时可以使用键盘：

- `q` - 退出
- `r` - 重置历史

## 网络拓扑

```
[AGX Orin]                           [本地电脑]                     [云端]
camera_streamer.py  ──────────────►  :50051 (SSH隧道)  ──────────►  推理服务
                    TCP连接                             SSH转发
```

## 故障排查

1. **摄像头无法打开**：检查 `ls /dev/video*`
2. **连接超时**：确认本地电脑 IP 和端口正确
3. **帧率低**：降低 --quality 或 --width/--height
4. **显示报错**：使用 --no-display 模式
