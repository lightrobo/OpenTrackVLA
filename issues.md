# OpenTrackVLA 问题记录

## Issue #1: Habitat 仿真采集失败 - EGL 渲染设备缺失

### 问题描述

运行 `stt_data.sh` 数据采集脚本时，Habitat 仿真器无法启动，报错：

```
Platform::WindowlessEglApplication::tryCreateContext(): unable to find CUDA device 0 among 1 EGL devices in total
WindowlessContext: Unable to create windowless context
```

### 环境信息

- GPU: 2x NVIDIA H20 (98GB 显存)
- Driver: 535.216.03
- CUDA: 12.2
- 运行环境: 容器 (火山云)

### 诊断结果

```bash
nvidia-smi          # ✅ GPU 正常识别
ls /dev/dri/        # ❌ No such file or directory
eglinfo             # ❌ command not found
```

**根本原因**: `/dev/dri/` 设备目录不存在，说明容器没有挂载 GPU 渲染设备。

Habitat 仿真器需要 EGL 来做 headless GPU 渲染，但当前容器配置缺少必要的渲染设备挂载。

### 解决方案

#### 方案1: 重新创建容器时加参数 (Docker)

```bash
docker run --gpus all \
    --device /dev/dri \
    -v /dev/dri:/dev/dri \
    ...
```

#### 方案2: 云平台/K8s 配置

联系平台管理员，请求启用 GPU 渲染设备挂载：
- 火山云: 可能需要开启 "GPU 渲染" 或 "OpenGL 支持"
- K8s: 需要配置渲染设备挂载

#### 方案3: CPU 渲染模式 (慢，不推荐)

```bash
export HABITAT_SIM_BACKEND=osmesa
# 或
export HABITAT_SIM_NO_CUDA=1
```

### 状态

- [ ] 待解决 - 需要联系平台管理员配置容器渲染设备

### 尝试过的解决方案

#### ❌ 方案A: 从源码编译 habitat-sim (headless 模式)

1. **修改源码**: 修改 `habitat-sim/src/esp/gfx/WindowlessContext.cpp`，当 `gpu_device_id=-1` 时不设置 CUDA 设备
2. **安装依赖**: OpenGL 开发库、Mesa 软件渲染驱动
3. **处理编译问题**: 
   - CMake 版本兼容性 (添加 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`)
   - GCC include-fixed 目录缺失 (添加 `-idirafter /usr/include`)
   - Assimp 版本检测失败 (修改为 warning 并假设版本)
4. **编译成功**: `cuda_enabled: False`, `built_with_bullet: True`

**结果**: 即使 `gpu_device_id=-1` 也无法运行。EGL 初始化本身需要 `/dev/dri` 渲染设备。

#### ❌ 方案B: xvfb + 软件渲染

```bash
xvfb-run -a -s "-screen 0 1024x768x24" python run.py ...
```

**结果**: 同样失败。Magnum 的 EGL 实现使用 `EGL_PLATFORM_DEVICE_EXT`，强制请求硬件设备。

#### ❌ 方案C: 环境变量

```bash
LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless python run.py ...
```

**结果**: 失败。警告 "Not allowed to force software rendering when API explicitly selects a hardware device"。

### 根本原因深入分析

Magnum 的 `WindowlessEglApplication.cpp` 在第 255 行使用 `eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, ...)` 获取 EGL display。这需要：

1. 系统有可用的 EGL 设备 (`eglQueryDevices` 返回 > 0)
2. 设备必须是有效的硬件渲染设备 (需要 `/dev/dri`)

**彻底解决需要**:
- 修改 Magnum 源码添加 `EGL_PLATFORM_SURFACELESS_MESA` 后备路径
- 或者在容器中挂载 `/dev/dri` 设备

### 相关命令

```bash
# 测试单个采集任务
CUDA_VISIBLE_DEVICES=0 SAVE_VIDEO=1 PYTHONPATH="habitat-lab" python run.py \
    --split-num 30 \
    --split-id 0 \
    --exp-config 'habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml' \
    --run-type 'eval' \
    --save-path sim_data/test_single \
    habitat.simulator.seed=101
```

---

## Issue #2: README 文档缺失 - 数据采集步骤未说明

### 问题描述

README.md 跳过了数据采集步骤，直接假设用户已有 `sim_data/` 格式的数据。

### 影响

- 新用户无法理解完整的数据处理流程
- 仓库自带的 `sim_data/sample/` 只有 2 个 episode，完全不够训练

### 实际情况

- `stt_data.sh` 采集脚本存在，但 README 未提及
- `baseline_agent.py` + `run.py` 可以用于采集，但没有文档说明

### 已创建补充文档

- `dataprocessingpipeline.md` - 完整数据处理管线说明
- `trackVla_or_Opentrackvla.md` - OpenTrackVLA vs TrackVLA 对比分析

### 状态

- [x] 已通过创建补充文档解决

---

## Issue #3: 评估 (Evaluation) 同样受 EGL 问题影响

### 问题描述

由于 Issue #1 的 `/dev/dri` 缺失问题，不仅数据采集无法运行，模型评估 (eval.sh) 也无法运行。

### 原因分析

评估流程同样依赖 Habitat 仿真器：

```
eval.sh → run_eval.py → trained_agent.py → habitat.TrackEnv → 需要 EGL 渲染
```

评估需要在仿真环境中让模型控制机器人跟踪目标人，计算 SR/TR/CR 指标。

### 影响范围

| 功能 | 需要 Habitat? | 当前可行? |
|------|--------------|----------|
| 下载/加载模型 | ❌ | ✅ |
| 处理 sample 数据 (make_tracking_data.py) | ❌ | ✅ |
| 预缓存视觉 token (precache_frames.py) | ❌ | ✅ |
| 用 sample 跑训练流程验证 | ❌ | ✅ |
| 单图推理测试 | ❌ | ✅ |
| **完整评估 (EVT-Bench)** | ✅ | ❌ 被阻塞 |
| **大规模数据采集** | ✅ | ❌ 被阻塞 |

### 临时替代方案

在等待 `/dev/dri` 问题解决期间，可以做以下工作：

#### 1. 下载并加载模型

```bash
# 下载模型到共享目录
huggingface-cli download omlab/opentrackvla-qwen06b \
    --local-dir /mnt/data/shared_models/opentrackvla
```

#### 2. 用 sample 数据验证训练流程

```bash
# 处理 sample 数据
python make_tracking_data.py \
    --input_root sim_data/sample \
    --output_root data/sample

# 预缓存视觉 token
python precache_frames.py \
    --data_root data/sample \
    --cache_root data/sample/vision_cache

# 用 2 个 episode 跑训练流程验证
python train.py \
    --train_json data/sample/jsonl \
    --cache_root data/sample/vision_cache \
    --out_dir ckpt_test \
    --epochs 1 \
    --batch_size 1
```

#### 3. 单图推理测试 (不需要 Habitat)

可以编写脚本对单张图片做推理，验证模型能正常加载和输出。

### 状态

- [ ] 被 Issue #1 阻塞 - 需要先解决 `/dev/dri` 问题

### 依赖关系

```
Issue #1 (EGL/dev/dri) 
    ├── 阻塞: 数据采集 (stt_data.sh)
    └── 阻塞: 模型评估 (eval.sh)  ← Issue #3
```

