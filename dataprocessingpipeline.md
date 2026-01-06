# OpenTrackVLA 数据处理管线详解

本文档补充 README 中缺失的数据管线细节，特别是 **数据采集步骤**。

---

## 数据流全景图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OpenTrackVLA 数据管线                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌─────────────┐
│  场景数据    │ -> │  仿真采集     │ -> │  make_tracking  │ -> │  precache   │
│  (HM3D/MP3D)│    │  (run.py)    │    │  _data.py       │    │  _frames.py │
└─────────────┘    └──────────────┘    └─────────────────┘    └─────────────┘
     静态3D场景      采集机器人视角       滑窗切片+轨迹积分       提取视觉token
                     + 控制信号                                  (可选加速)
```

---

## Step 0: 场景数据（静态资产）

你下载的 HM3D/MP3D 场景数据是 **静态3D场景**，不是训练数据！

```
data/
├── scene_datasets/
│   ├── hm3d/           # 房间、家具、墙壁等3D mesh
│   │   ├── train/
│   │   ├── val/
│   │   └── minival/
│   └── mp3d/
└── humanoids/          # 可动人形模型
    └── humanoid_data/
        ├── female_2/
        └── male_0/
```

这些是 Habitat 仿真器的"舞台背景"和"演员"。

---

## Step 1: 仿真采集 (README未提及！)

> ⚠️ **README 跳过了这一步**，直接假设你有 `sim_data/` 数据。

### 采集原理

用 Habitat 仿真器在场景里跑机器人跟踪人，采集第一视角视频和控制信号：

```python
# baseline_agent.py 核心逻辑
while not env.episode_over:
    obs = sim.get_sensor_observations()  # 获取机器人视角RGB
    action = robot_config.act(...)       # PID控制器决定动作 [vx, vy, wz]
    env.step(action_dict)                # 执行动作
    
    # 记录每一步的状态
    record_info = {
        "step": iter_step,
        "dis_to_human": float(...),      # 到目标人的距离
        "facing": info['human_following'], # 是否面向目标
        "base_velocity": action           # [前进速度, 侧移速度, 转向角速度]
    }
    record_infos.append(record_info)
```

### 采集命令（推测）

```bash
# README没写，但代码支持
python run.py \
    --run-type eval \
    --exp-config evt_bench/configs/track_stt.yaml \
    --split-id 0 \
    --split-num 1 \
    --save-path sim_data/my_rollouts
```

### 采集输出

每个成功的 episode 产生3个文件：

```
sim_data/my_rollouts/seed_xxx/scene_id/
├── episode_id.mp4        # 机器人第一视角视频 (RGB序列)
├── episode_id_info.json  # 每步状态: base_velocity + dis_to_human + facing
└── episode_id.json       # episode元信息: success/collision/instruction
```

**episode_id_info.json 结构**：
```json
[
  {"step": 1, "dis_to_human": 1.40, "facing": 1.0, "base_velocity": [0.41, -0.08, -0.22]},
  {"step": 2, "dis_to_human": 1.46, "facing": 0.0, "base_velocity": [-0.47, -0.13, -0.29]},
  ...
]
```

**episode_id.json 结构**：
```json
{
  "finish": true,
  "status": "Normal",
  "success": 1.0,
  "following_rate": 0.56,
  "following_step": 34,
  "total_step": 61,
  "collision": 0.0,
  "instruction": "Follow the person wearing a brown armored outfit and green sleeves."
}
```

---

## Step 2: 数据处理 (make_tracking_data.py)

这一步把采集的原始数据转换成训练格式。

### 命令

```bash
python make_tracking_data.py \
    --input_root sim_data/sample \
    --output_root data/sample \
    --history 31 \
    --horizon 8 \
    --dt 0.1 \
    --only_success \
    --instruction "Follow the target person without collision."
```

### 参数说明

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--input_root` | 采集数据目录 (sim_data/xxx) | 必填 |
| `--output_root` | 输出目录 | 必填 |
| `--history` | 历史帧数量 | 31 |
| `--horizon` | 未来预测步数 | 8 |
| `--dt` | 每步时间间隔(秒) | 0.1 |
| `--only_success` | 只保留成功episode | False |
| `--instruction` | 覆盖所有样本的instruction | None |

### 处理流程

#### 2.1 视频抽帧

```python
# 用 ffmpeg 把 .mp4 拆成 .jpg 序列
frame_paths = extract_frames_ffmpeg(ffmpeg_path, episode.mp4, output_dir)
# frame_00001.jpg, frame_00002.jpg, ...
```

#### 2.2 滑窗切片

对每个时间点 j，生成一个训练样本：

```python
for j in range(len(frames)):
    # 历史帧: j-31 到 j-1
    images_window = frames[max(0, j-history):j]
    
    # 当前帧
    current_frame = frames[j]
    
    # 未来轨迹: 从 j 积分 horizon 步
    trajectory = integrate_future_trajectory(actions, j, horizon, dt)
```

#### 2.3 轨迹积分

把速度序列 `[vx, vy, wz]` 积分成位置序列 `[x, y, θ]`：

```python
def integrate_future_trajectory(actions, start_index, horizon, dt):
    """
    运动学积分: 速度 → 位移
    
    输入: actions = [[vx, vy, wz], ...]  # 机器人本体坐标系速度
    输出: trajectory = [[x, y, θ], ...]  # 相对于起点的位置
    """
    x, y, theta = 0.0, 0.0, 0.0
    trajectory = []
    
    for k in range(start_index, start_index + horizon):
        vx, vy, wz = actions[k]
        
        # 本体速度转全局位移 (旋转到当前朝向)
        dx = vx * cos(theta) - vy * sin(theta)
        dy = vx * sin(theta) + vy * cos(theta)
        
        # 积分
        x += dx * dt
        y += dy * dt
        theta += wz * dt
        
        trajectory.append([x, y, theta])
    
    return trajectory  # [[x1,y1,θ1], [x2,y2,θ2], ...]
```

### 输出结构

```
data/sample/
├── frames/                              # 抽帧图像
│   └── seed_100/17DRP5sb8fy/4/
│       ├── frame_00001.jpg
│       ├── frame_00002.jpg
│       └── ...
├── jsonl/                               # 训练数据 (每episode一个文件)
│   └── seed_100/17DRP5sb8fy/
│       └── 4.jsonl
└── dataset.json                         # 汇总文件 (可选)
```

### 训练样本格式

`jsonl/` 里每行是一个训练样本：

```json
{
  "images": [
    "frames/seed_100/17DRP5sb8fy/4/frame_00001.jpg",
    "frames/seed_100/17DRP5sb8fy/4/frame_00002.jpg",
    ...
    "frames/seed_100/17DRP5sb8fy/4/frame_00031.jpg"
  ],
  "current": "frames/seed_100/17DRP5sb8fy/4/frame_00032.jpg",
  "instruction": "Follow the person wearing a brown armored outfit and green sleeves.",
  "trajectory": [
    [0.041, 0.008, -0.022],
    [0.082, 0.015, -0.044],
    [0.123, 0.021, -0.066],
    ...
  ],
  "actions": [
    [0.41, -0.08, -0.22],
    [0.40, -0.07, -0.21],
    ...
  ],
  "collision": false,
  "target_distance": 1.4
}
```

| 字段 | 含义 |
|------|------|
| `images` | 历史帧路径列表 (31帧) |
| `current` | 当前帧路径 |
| `instruction` | 自然语言指令 |
| `trajectory` | 未来轨迹 waypoints `[[x,y,θ], ...]` (8个点) |
| `actions` | 原始速度命令 `[[vx,vy,wz], ...]` |
| `collision` | 当前步是否碰撞 |
| `target_distance` | 到目标人的距离 |

---

## Step 3: 视觉Token预计算 (可选加速)

训练时每帧都要过 DINO + SiGLIP 编码器，非常慢。预计算可以大幅加速。

### 命令

```bash
python precache_frames.py \
    --data_root data/sample \
    --cache_root data/sample/vision_cache \
    --batch_size 8 \
    --image_size 384
```

### 处理流程

```python
def _encode_single(pil_image, encoder):
    # DINOv2: 提取局部视觉特征
    tok_dino, H, W = encoder._encode_dino([pil_image])  # (1, H*W, 384)
    
    # SiGLIP: 提取语义特征
    tok_siglip = encoder._encode_siglip([pil_image])    # (1, H*W, 1152)
    
    # 拼接
    Vt_cat = torch.cat([tok_dino, tok_siglip], dim=-1)  # (1, H*W, 1536)
    
    # Grid pooling 压缩 token 数量
    Vfine = grid_pool_tokens(Vt_cat, out_tokens=64)     # (1, 64, 1536) 细粒度
    Vcoarse = grid_pool_tokens(Vt_cat, out_tokens=4)    # (1, 4, 1536)  粗粒度
    
    return Vcoarse, Vfine
```

### 输出结构

```
data/sample/vision_cache/
└── frames/seed_100/17DRP5sb8fy/4/
    ├── frame_00001_vfine.pt     # (64, 1536) tensor, fp16
    ├── frame_00001_vcoarse.pt   # (4, 1536) tensor, fp16
    ├── frame_00002_vfine.pt
    ├── frame_00002_vcoarse.pt
    └── ...
```

---

## Step 4: 训练

```bash
python train.py \
    --train_json data/sample/jsonl \
    --cache_root data/sample/vision_cache \
    --out_dir ckpt_sample \
    --epochs 2 \
    --batch_size 8 \
    --n_waypoints 8 \
    --history 31 \
    --lr 2e-5 \
    --mixed_precision
```

---

## 完整流程总结

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  data/scene_datasets/hm3d/  (静态3D场景)                                    │
│  data/humanoids/            (人形模型)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         │  Habitat 仿真器
                         │  baseline_agent.py + run.py  [README未提及!]
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  sim_data/sample/seed_xxx/scene_id/                                         │
│    ├── episode.mp4       (机器人视角视频)                                   │
│    ├── episode_info.json (每步: velocity, distance, facing)                 │
│    └── episode.json      (元信息: success, instruction)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         │  python make_tracking_data.py
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  data/sample/                                                               │
│    ├── frames/.../*.jpg   (视频抽帧)                                        │
│    ├── jsonl/.../*.jsonl  (滑窗样本: images + current + trajectory)         │
│    └── dataset.json       (汇总, 可选)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         │  python precache_frames.py  [可选]
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  data/sample/vision_cache/                                                  │
│    └── frames/.../*_vfine.pt, *_vcoarse.pt  (DINO+SiGLIP embedding)         │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         │  python train.py
                         ▼
                    模型训练 ✅
```

---

## 关键洞察

| 步骤 | 输入 | 输出 | 本质 |
|------|------|------|------|
| 仿真采集 | 3D场景 + humanoid | mp4 + info.json | 行为克隆数据采集 |
| make_tracking_data | mp4 + info.json | frames/ + jsonl/ | 滑窗切片 + 速度→轨迹积分 |
| precache_frames | frames/*.jpg | vision_cache/*.pt | 预计算视觉编码 (加速训练) |
| train | jsonl + vision_cache | model.pt | 训练VLA模型 |

**核心**：模型学习的监督信号是 **轨迹 waypoints** `[x, y, θ]`，不是原始速度。模型需要学会"看到这些画面 + 这个指令，预测未来8步的位置"。

---

## 数据量说明

| 数据 | 仓库提供 | 实际训练需要 |
|------|---------|-------------|
| sim_data/sample | 2个episode | 数万~数十万episode |
| 训练样本 | ~100条 | 1.7M条 (TrackVLA论文) |

仓库自带的 sample 只够跑通流程验证代码，**不够训练有效模型**。

