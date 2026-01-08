# Episode 终止条件与成功判定

## 终止条件

Episode 有 **3 种终止方式**：

### 1. 正常完成 (Normal)
目标人物走完所有预设路径点 (waypoints) 后停下。

```python
# evt_bench/additional_action.py L996-1000
if self.wait_step_for_robot > self.max_stop_step:
    kwargs['task'].should_end = True
    kwargs['task'].is_stop_called = True
```

流程：目标走完 waypoints → 等待 `max_stop_step` 步 → `env.episode_over = True`

### 2. 跟丢 (Lost)
机器人与目标距离 > 4 米，**连续 20 步**。

```python
# oracle_baseline_agent.py L99-104
if dist > 4.0:
    too_far_count += 1
    if too_far_count > 20:
        status = 'Lost'
        break
```

### 3. 碰撞 (Collision)
机器人与目标距离 < 0.5 米，触发碰撞判定。

```python
# evt_bench/additional_metric.py L155-157
if collid < 0.5 or self._ever_collide:
    self._metric = 1.0
    self._ever_collide = True
```

---

## 成功判定

### human_following
当前帧是否正在跟踪（目标在视野内且距离 < 3 米）。

```python
# evt_bench/additional_metric.py L628
success_distance: float = 3.0
```

### human_following_success
Episode 结束时的成功判定，需要**同时满足**：

1. `is_stop_called = True`（目标人物已走完路线并停下）
2. 距离在 **1.0 ~ 3.0 米** 之间
3. 正在跟踪（目标在视野内）

```python
# evt_bench/additional_metric.py L631-635
class HumanFollowingSuccessConfig(MeasurementConfig):
    success_following_distance_lower: float = 1.0
    success_following_distance_upper: float = 3.0
    max_episode_steps: int = 300
```

### 最终成功判定逻辑

```python
# oracle_baseline_agent.py L123-126
if iter_step < 300:
    result['success'] = info['human_following_success'] and info['human_following']
else:
    result['success'] = info['human_following']
```

- 短 episode (< 300 步)：需要 `human_following_success` **且** `human_following`
- 长 episode (≥ 300 步)：只需要 `human_following`

---

## 关键距离阈值

| 阈值 | 值 | 含义 |
|------|-----|------|
| 碰撞距离 | < 0.5m | 触发 Collision |
| 成功下限 | 1.0m | human_following_success 最小距离 |
| 成功上限 | 3.0m | human_following_success 最大距离 |
| 跟丢距离 | > 4.0m | 连续 20 步触发 Lost |

---

## Agent 距离控制参数

```python
# oracle_baseline_agent.py
DANGER_DISTANCE = 0.8   # 太近，后退
MIN_DISTANCE = 1.2      # 停止前进
FOLLOW_DISTANCE = 2.0   # 理想跟踪距离（在 1.0~3.0 成功区间内）
```

---

## 流程图

```
Episode 开始
     ↓
目标人物开始行走
     ↓
┌────────────────────────────────────────┐
│            主循环 (每步检查)             │
├────────────────────────────────────────┤
│ 1. 距离 > 4m 连续 20 步? → Lost, 终止   │
│ 2. 距离 < 0.5m?         → Collision, 终止│
│ 3. 目标走完路线?        → Normal, 终止  │
└────────────────────────────────────────┘
     ↓
Episode 结束，计算成功判定
     ↓
┌────────────────────────────────────────┐
│ 成功条件：                              │
│ - 距离在 1.0 ~ 3.0m                    │
│ - 目标在视野内                          │
│ - (短episode) 目标已停下                │
└────────────────────────────────────────┘
```

