# TrackVLA vs OpenTrackVLA：到底开源了什么？

## 核心结论

**TrackVLA 是"假开源"，OpenTrackVLA 才是"真开源"**

TrackVLA 发布了一个评估基准(benchmark)，但模型本身闭源。OpenTrackVLA 提供了完整的训练pipeline。

---

## 详细对比

### TrackVLA 实际开源内容

| 组件 | 状态 | 说明 |
|------|------|------|
| 模型权重 (Vicuna-7B) | ❌ 没有 | 论文里的7B模型权重**根本没发布** |
| model.py | ❌ 没有 | 没有模型架构定义代码 |
| train.py | ❌ 没有 | 没有训练代码 |
| 训练数据 (1.7M样本) | ❌ 没有 | 训练数据集没有发布 |
| EVT-Bench | ✅ 有 | 评估基准代码 |
| baseline_agent.py | ✅ 有 | **只是个PID控制器，不是TrackVLA模型** |
| 场景数据下载说明 | ✅ 有 | HM3D/MP3D场景、Humanoid数据 |

#### baseline_agent.py 的真相

TrackVLA仓库里的 `baseline_agent.py` 根本不是神经网络模型，而是一个简单的PID控制器：

```python
class GTBBoxAgent(AgentConfig):
    def __init__(self, result_path):
        # PID控制参数
        self.kp_t = 2   # 比例系数
        self.kd_t = 0   # 微分系数
        self.kp_f = 1
        self.kd_f = 0

    def act(self, observations, detector, episode_id):
        # 使用GT bounding box做简单的PID跟踪
        error_t = 0.5 - center_x
        yaw_speed = self.kp_t * error_t + self.kd_t * derivative_t
        move_speed = self.kp_f * error_f + self.kd_f * derivative_f
        action = [move_speed, 0, yaw_speed]
        return action
```

这就是一个**Ground Truth BBox + PID控制器**，连神经网络都没有。

---

### OpenTrackVLA 实际开源内容

| 组件 | 状态 | 说明 |
|------|------|------|
| 模型权重 (Qwen-0.6B) | ✅ 有 | `omlab/opentrackvla-qwen06b` |
| model.py | ✅ 有 | 764行，完整模型架构定义 |
| train.py | ✅ 有 | 1524行，完整训练代码 |
| make_tracking_data.py | ✅ 有 | 401行，数据处理pipeline |
| precache_frames.py | ✅ 有 | 215行，视觉token预计算 |
| trained_agent.py | ✅ 有 | 489行，推理代码 |
| convert_ckpt_to_hf.py | ✅ 有 | 模型转换脚本 |
| open_trackvla_hf/ | ✅ 有 | HuggingFace格式模型定义 |

---

## 本质区别

### TrackVLA 的"开源"策略

```
┌─────────────────────────────────────────────────────────┐
│  TrackVLA 发布了什么？                                   │
│                                                         │
│  ✅ EVT-Bench (评估基准)                                 │
│     - 仿真环境配置                                       │
│     - 评估指标 (SR/TR/CR)                                │
│     - 测试episode数据                                    │
│                                                         │
│  ❌ TrackVLA 模型本身                                    │
│     - 没有权重                                           │
│     - 没有架构                                           │
│     - 没有训练代码                                       │
│     - 没有训练数据                                       │
└─────────────────────────────────────────────────────────┘
```

**你能做的**：在他们的benchmark上测试自己的方法，看跟论文里的数字差多少。

**你不能做的**：复现TrackVLA模型、fine-tune、理解它是怎么工作的。

### OpenTrackVLA 的开源策略

```
┌─────────────────────────────────────────────────────────┐
│  OpenTrackVLA 发布了什么？                               │
│                                                         │
│  ✅ 完整训练Pipeline                                     │
│     - 模型权重 (0.6B)                                    │
│     - 模型架构 (model.py)                                │
│     - 训练代码 (train.py)                                │
│     - 数据处理 (make_tracking_data.py)                   │
│     - 推理代码 (trained_agent.py)                        │
│                                                         │
│  ✅ 评估能力                                             │
│     - 兼容EVT-Bench                                      │
│     - eval.sh                                            │
└─────────────────────────────────────────────────────────┘
```

**你能做的**：从头训练、fine-tune、修改架构、理解原理、改进方法。

---

## 性能对比 (EVT-Bench)

| Methods | STT (SR↑ / TR↑ / CR↓) | DT (SR↑ / TR↑ / CR↓) | AT (SR↑ / TR↑ / CR↓) |
|---------|------------------------|----------------------|----------------------|
| TrackVLA (Vicuna-7B) | 85.1 / 78.6 / 1.65 | 57.6 / 63.2 / 5.80 | 50.2 / 63.7 / 17.1 |
| OpenTrackVLA (Qwen-0.6B) | 64.8 / 84.4 / 5.00 | 33.6 / 66.3 / 8.84 | 39.6 / 76.7 / 6.38 |

- **SR (Success Rate)**：成功率，越高越好
- **TR (Tracking Rate)**：跟踪率，越高越好
- **CR (Collision Rate)**：碰撞率，越低越好

**观察**：
- OpenTrackVLA 0.6B 在 TR (跟踪率) 上全面超过 TrackVLA 7B
- OpenTrackVLA 在 SR (成功率) 上落后，这是模型容量差距导致的（0.6B vs 7B）
- OpenTrackVLA 在 CR (碰撞率) 上表现良好

---

## 总结

这是学术界的经典操作：

> "We open-sourced our benchmark!"  
> "But the model that beats everything on that benchmark? That's proprietary, sorry."

TrackVLA 发布了一个"公平竞技场"，但自己手里攥着最强的武器不放。这不叫开源，这叫**设卡收费**——你可以进场比赛，但赢家已经内定了。

OpenTrackVLA 虽然 0.6B 打不过 7B（物理定律决定的），但至少你能：
- 看到它是怎么做的
- 自己训练
- 自己改进
- 在上面做研究

**这才是开源精神。**

---

## 文件结构对比

### TrackVLA 仓库
```
TrackVLA/
├── baseline_agent.py      # PID控制器baseline（不是TrackVLA模型）
├── agent_uninavid.py      # Uni-NaVid评估支持
├── eval_baseline.sh       # 评估脚本
├── eval_uninavid.sh
├── run.py
├── analyze_results.py
├── evt_bench/             # 评估基准代码
├── track_episode_step/    # 测试episode数据
├── habitat-lab/           # Habitat环境
└── data/                  # 场景数据（需自行下载）
```

### OpenTrackVLA 仓库
```
OpenTrackVLA/
├── model.py               # ✅ 模型架构定义
├── train.py               # ✅ 训练代码
├── trained_agent.py       # ✅ 推理代码
├── make_tracking_data.py  # ✅ 数据处理
├── precache_frames.py     # ✅ 视觉token预计算
├── convert_ckpt_to_hf.py  # ✅ 模型转换
├── open_trackvla_hf/      # ✅ HuggingFace模型
├── eval.sh                # 评估脚本
├── run_eval.py
├── evt_bench/             # 评估基准代码
├── habitat-lab/           # Habitat环境
├── sim_data/              # 示例数据
└── data/                  # 场景数据
```

