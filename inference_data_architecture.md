# OpenTrackVLA 推理架构

## 系统概览

本项目采用**客户端-中转-服务器**三层架构，将摄像头采集（AGX Orin）与模型推理（云端GPU）分离。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   AGX Orin                  本地电脑                     云端开发机     │
│   (机器人端)                (中转站)                     (推理端)       │
│                                                                         │
│  ┌──────────────┐       ┌──────────────┐           ┌──────────────┐    │
│  │camera_streamer│       │  SSH 隧道    │           │remote_server │    │
│  │  (gRPC客户端) │──────►│  端口转发    │──────────►│ (gRPC服务端) │    │
│  └──────────────┘       └──────────────┘           └──────────────┘    │
│         │                      │                          │            │
│         │                      │                          ▼            │
│    USB摄像头               run_ssh_tunneling.sh      GTBBoxAgent       │
│    采集画面                                          模型推理          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 模块职责

| 模块 | 位置 | 角色 | 核心功能 |
|------|------|------|----------|
| `camera_streamer/` | AGX Orin | gRPC 客户端 | 采集摄像头画面，发送到服务器，接收 waypoints |
| `local_computer/` | 本地电脑 | 中转隧道 | SSH 隧道转发，测试工具 |
| `remote_server/` | 云端开发机 | gRPC 服务端 | 加载模型，执行推理，返回导航路径 |

## 数据流

### 请求流（图像 → 云端）

```
USB摄像头
    │
    ▼ cv2.VideoCapture
┌─────────────────┐
│  BGR 图像帧     │
│  (numpy array)  │
└────────┬────────┘
         │ cv2.imencode('.jpg')
         ▼
┌─────────────────┐
│  JPEG 二进制    │
│  (bytes)        │
└────────┬────────┘
         │ gRPC InferRequest
         ▼
┌─────────────────┐
│  InferRequest   │
│  - image_data   │  ──────► SSH隧道 ──────► 云端服务器
│  - instruction  │
│  - frame_id     │
│  - timestamp_ms │
└─────────────────┘
```

### 响应流（云端 → AGX）

```
云端 GTBBoxAgent
    │
    ▼ _planner_action()
┌─────────────────┐
│  轨迹预测       │
│  (N, 3) array   │  ◄────── 模型推理
│  [x, y, theta]  │
└────────┬────────┘
         │ flatten().tolist()
         ▼
┌─────────────────┐
│  InferResponse  │
│  - waypoints    │  ──────► SSH隧道 ──────► AGX Orin
│  - n_waypoints  │
│  - inference_ms │
│  - success      │
└─────────────────┘
         │
         ▼ reshape(n_waypoints, 3)
┌─────────────────┐
│  导航路径点     │
│  用于机器人控制 │
└─────────────────┘
```

## gRPC 协议定义

```protobuf
service InferenceService {
    rpc Infer(InferRequest) returns (InferResponse);        // 单帧推理
    rpc StreamInfer(stream InferRequest) returns (stream InferResponse);  // 双向流
    rpc HealthCheck(Empty) returns (HealthResponse);        // 健康检查
}

message InferRequest {
    bytes image_data = 1;      // JPEG 编码的图像
    string instruction = 2;    // 文本指令 (如 "follow the person")
    int64 frame_id = 3;        // 帧ID
    int64 timestamp_ms = 4;    // 时间戳
}

message InferResponse {
    int64 frame_id = 1;        // 帧ID
    repeated float waypoints = 2;  // 展平的路径点 [x,y,θ, x,y,θ, ...]
    int32 n_waypoints = 3;     // 路径点数量
    float inference_time_ms = 4;   // 推理耗时
    string error = 5;          // 错误信息
    bool success = 6;          // 是否成功
}
```

## 网络配置

### 典型部署场景

```
AGX Orin: 192.168.1.50
    │
    │ TCP → 连接 192.168.1.100:50051
    ▼
本地电脑: 192.168.1.100
    │
    │ SSH隧道 → 转发到 cloud-server:50051
    ▼
云端开发机: cloud-server.com:50051
```

### 为什么需要中转？

1. AGX Orin 通常在局域网内（机器人/嵌入式设备）
2. 云端开发机在公网或 VPN 后面
3. AGX 可能无法直接访问云端
4. 本地电脑可以 SSH 到云端，充当跳板

### 直连模式（可选）

如果 AGX 能直接访问云端，可跳过中转：

```
AGX Orin ────────► 云端开发机:50051
         直接 gRPC 连接
```

## 启动顺序

```bash
# 1. 云端：启动推理服务
cd remote_server
python remote_server.py --model-dir /path/to/model --port 50051

# 2. 本地电脑：启动 SSH 隧道
cd local_computer
./run_ssh_tunneling.sh

# 3. AGX Orin：启动摄像头采集
cd camera_streamer
python camera_streamer.py --server 192.168.1.100:50051 --instruction "follow the person"
```

## 关键代码路径

| 文件 | 关键函数 | 作用 |
|------|----------|------|
| `camera_streamer.py` | `CameraStreamer.infer()` | 编码图像，发送 gRPC 请求 |
| `camera_streamer.py` | `CameraStreamer.draw_waypoints()` | 可视化路径点 |
| `remote_server.py` | `InferenceServicer.Infer()` | 接收请求，调用模型推理 |
| `trained_agent.py` | `GTBBoxAgent._planner_action()` | 实际的模型推理逻辑 |

