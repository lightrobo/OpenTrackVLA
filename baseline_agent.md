# Baseline Agent 分析报告

## 概述

`baseline_agent.py` 实现了一个基于视觉的人体跟踪agent，使用PD控制器根据目标人物在画面中的位置和大小来控制机器人移动。

---

## 文件结构

```
baseline_agent.py
├── evaluate_agent()      # 主评估循环
└── class GTBBoxAgent     # 基于BBox的跟踪控制器
    ├── __init__()        # 初始化PD参数
    ├── reset()           # 重置状态，保存视频
    └── act()             # 核心控制逻辑
```

---

## 核心代码分析

### 1. PD控制器参数初始化

```python
class GTBBoxAgent(AgentConfig):
    def __init__(self, result_path, target_id=None):
        super().__init__()
        print("Initialize gtbbox agent")

        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        self.target_id = target_id
        
        self.rgb_list = []
        self.rgb_box_list = []

        # PD控制器参数
        self.kp_t = 2      # 转向比例增益（控制yaw）
        self.kd_t = 0      # 转向微分增益
        self.kp_f = 1      # 前进比例增益（控制前进速度）
        self.kd_f = 0      # 前进微分增益
        self.kp_y = 0.5    # 侧移比例增益
        self.kd_y = 0      # 侧移微分增益

        self.prev_error_t = 0  # 上一帧转向误差
        self.prev_error_f = 0  # 上一帧前进误差

        self.first_inside = True
        self.reset()
```

**问题**：`kd_t = 0` 和 `kd_f = 0`，微分项完全没用，这是个纯P控制器，没有阻尼，容易震荡和超调。

---

### 2. 核心控制逻辑 `act()`

```python
def act(self, observations, detector, episode_id):
    self.episode_id = episode_id
    
    rgb = observations["agent_1_articulated_agent_jaw_rgb"]
    rgb_ = rgb[:, :, :3]
    image = np.asarray(rgb_[:, :, ::-1])
    height, width = image.shape[:2]
    
    action = [0, 0, 0]  # 默认动作：不动

    target_tracked = False
    
    # 方式1：如果指定了target_id，使用panoptic分割
    if self.target_id is not None and "agent_1_articulated_agent_jaw_panoptic" in observations:
        panoptic = observations["agent_1_articulated_agent_jaw_panoptic"]
        target_mask = (panoptic == self.target_id)
        if hasattr(target_mask, "ndim") and target_mask.ndim == 3:
            target_mask = target_mask[:, :, 0]
        if np.any(target_mask):
            # 从mask计算bounding box
            rows = np.any(target_mask, axis=1)
            cols = np.any(target_mask, axis=0)
            r_idx = np.where(rows)[0]
            c_idx = np.where(cols)[0]
            rmin, rmax = int(r_idx[0]), int(r_idx[-1])
            cmin, cmax = int(c_idx[0]), int(c_idx[-1])
            box = np.array([cmin, rmin, cmax, rmax], dtype=np.float32)
            
            # 归一化box: [center_x, center_y, width, height]
            best_box = np.array([
                (box[0] + box[2]) / (2 * width),
                (box[1] + box[3]) / (2 * height),
                (box[2] - box[0]) / width,
                (box[3] - box[1]) / height,
            ], dtype=np.float32)

            # 计算控制误差
            center_x = best_box[0]
            error_t = 0.5 - center_x  # 转向误差：让人保持在画面中心
            
            bbox_area = (box[2] - box[0]) * (box[3] - box[1])
            error_f = (30000 - bbox_area) / 10000  # 前进误差：用bbox面积估计距离
            if abs(error_f) < 0.5:
                error_f = 0  # 死区

            # PD控制
            derivative_t = error_t - self.prev_error_t
            derivative_f = error_f - self.prev_error_f

            yaw_speed = self.kp_t * error_t + self.kd_t * derivative_t   # = 2 * error_t
            move_speed = self.kp_f * error_f + self.kd_f * derivative_f  # = 1 * error_f
            y_speed = self.kp_y * error_t + self.kd_y * derivative_t     # = 0.5 * error_t

            self.prev_error_t = error_t
            self.prev_error_f = error_f

            action = [move_speed, y_speed, yaw_speed]
            target_tracked = True

    # 方式2：使用detector提供的主要人物bbox
    if not target_tracked:
        if detector['agent_1_main_humanoid_detector_sensor']['facing']:
            box = detector['agent_1_main_humanoid_detector_sensor']['box']
            best_box = np.array([
                (box[0]+box[2])/(2*width), 
                (box[1]+box[3])/(2*height), 
                (box[2]-box[0])/width, 
                (box[3]-box[1])/height
            ], dtype=np.float32)
            
            center_x = best_box[0]
            error_t = 0.5 - center_x
            error_f = (30000 - (box[2]-box[0])*(box[3]-box[1])) / 10000
            if abs(error_f) < 0.5:
                error_f = 0

            derivative_t = error_t - self.prev_error_t
            derivative_f = error_f - self.prev_error_f

            yaw_speed = self.kp_t * error_t + self.kd_t * derivative_t
            move_speed = self.kp_f * error_f + self.kd_f * derivative_f
            y_speed = self.kp_y * error_t + self.kd_y * derivative_t

            self.prev_error_t = error_t
            self.prev_error_f = error_f

            action = [move_speed, y_speed, yaw_speed]
        else:
            action = [0, 0, 0]  # 看不到人就停下
    
    self.last_action = action
    self.rgb_list.append(rgb_)

    return action
```

---

### 3. 评估循环与失败检测

```python
def evaluate_agent(config, dataset_split, save_path, target_id=None) -> None:
    robot_config = GTBBoxAgent(save_path, target_id)
    with habitat.TrackEnv(config=config, dataset=dataset_split) as env:
        sim = env.sim
        robot_config.reset()
        
        num_episodes = len(env.episodes)
        for _ in trange(num_episodes):
            obs = env.reset()
            # ... 光照设置省略 ...

            result = {}
            record_infos = []
            
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent

            iter_step = 0
            followed_step = 0
            too_far_count = 0
            status = 'Normal'

            while not env.episode_over:
                record_info = {}
                obs = sim.get_sensor_observations()
                detector = env.task._get_observations(env.current_episode)
                action = robot_config.act(obs, detector, env.current_episode.episode_id)

                action_dict = {
                    "action": ("agent_0_humanoid_navigate_action", "agent_1_base_velocity", ...),
                    "action_args": {"agent_1_base_vel": action}
                }
                
                iter_step += 1
                env.step(action_dict)

                info = env.get_metrics()
                if info['human_following'] == 1.0:
                    print("Followed")
                    followed_step += 1
                    too_far_count = 0
                else:
                    print("Lost")

                # 失败条件1：距离超过4米持续20步
                if np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos) > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        print("Too far from human!")
                        status = 'Lost'
                        finished = False
                        break

                # 记录信息
                record_info["step"] = iter_step
                record_info["dis_to_human"] = float(np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos))
                record_info["facing"] = info['human_following']
                record_info["base_velocity"] = action
                record_infos.append(record_info)

                # 失败条件2：碰撞
                if info['human_collision'] == 1.0:
                    print("Collision detected!")
                    status = 'Collision'
                    finished = False
                    break
```

---

### 4. 成功判定与数据保存

```python
            # 成功判定逻辑
            if env.episode_over:
                finished = True

            result['finish'] = finished
            result['status'] = status
            
            # 关键：成功判定
            if iter_step < 300:
                result['success'] = info['human_following_success'] and info['human_following']
            else:
                result['success'] = info['human_following']
            
            result['following_rate'] = followed_step / iter_step
            result['following_step'] = followed_step
            result['total_step'] = iter_step
            result['collision'] = info['human_collision']

            # 只有成功才保存数据！
            if result['success']:
                scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split('.')[0]
                save_dir = os.path.join(save_path, scene_key)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存轨迹信息
                with open(os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w") as f:
                    json.dump(record_infos, f, indent=2)
                
                # 保存结果摘要
                with open(os.path.join(save_dir, "{}.json".format(env.current_episode.episode_id)), "w") as f:
                    json.dump(result, f, indent=2)

            # 重置agent，成功时保存视频
            robot_config.reset(env.current_episode, success=result['success'])
```

---

### 5. 视频保存逻辑

```python
def reset(self, episode: NavigationEpisode = None, success: bool = False):
    if len(self.rgb_list) != 0 and episode is not None:
        if success:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
            imageio.mimsave(output_video_path, self.rgb_list)
            print(f"Successfully saved the episode video with episode id {episode.episode_id}")
        self.rgb_list = []  # 清空，不管成功与否
    
    self.first_inside = True
```

---

## 致命问题分析

### 问题1：用bbox面积估计距离 — 根本性错误

```python
bbox_area = (box[2] - box[0]) * (box[3] - box[1])
error_f = (30000 - bbox_area) / 10000
```

**假设**：`bbox面积小 = 人远，面积大 = 人近`

**现实**：bbox面积受多种因素影响，与实际距离不是简单的反比关系：

| 情况 | bbox面积 | 实际距离 | 控制结果 |
|------|----------|----------|----------|
| 人在画面中央正对 | 大 | 近 | ✅ 正确减速 |
| 人在画面边缘 | 小 | 近 | ❌ 错误加速 |
| 人被部分遮挡 | 小 | 近 | ❌ 错误加速 |
| 人侧身 | 小 | 中 | ❌ 错误加速 |
| 人弯腰 | 变化 | 不变 | ❌ 速度震荡 |

**实测后果**：
```
Step 12: action=[3.0, 0.25, 1.0] dis=1.27m → Followed
Step 13: action=[3.0, 0.25, 1.0] dis=0.91m → Collision!
```

距离仅0.91米时，bbox可能只有几千像素，`error_f ≈ 3.0`，agent以最大速度前冲 → **碰撞**

---

### 问题2：没有速度限制

```python
move_speed = self.kp_f * error_f  # kp_f = 1, 无上限！
```

当 `bbox_area = 0`（完全看不到或面积极小）时：
```python
error_f = (30000 - 0) / 10000 = 3.0
move_speed = 1 * 3.0 = 3.0  # 全速前进！
```

没有任何 `clamp` 或速度上限。

---

### 问题3：没有安全距离阈值

代码中完全没有基于**实际距离**的安全机制：

```python
# 应该有但没有的逻辑：
if actual_distance < 1.0:
    move_speed = 0  # 太近了，停下！
```

仅依赖不可靠的bbox面积估计，没有使用环境中可用的：
- 深度传感器
- 真实距离（`np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos)`，仅用于日志）

---

### 问题4：Lost状态处理过于简陋

```python
if not detector['agent_1_main_humanoid_detector_sensor']['facing']:
    action = [0, 0, 0]  # 完全停止，什么都不做
```

看不到人就完全停止，没有：
- 惯性/动量补偿
- 搜索旋转行为
- 历史轨迹预测
- 最后已知位置追踪

---

### 问题5：只保存成功数据

```python
if result['success']:
    # 保存数据
```

失败的episode（碰撞、跟丢）不保存任何数据。如果agent表现太差（全失败），`save_path` 目录会是空的。

---

## 实测表现

从运行日志看，大量episode因碰撞失败：

| Episode | 中间状态 | 结局 | 碰撞前距离 |
|---------|---------|------|-----------|
| ID: 0 | 有Followed | Collision | 0.91m |
| ID: 2 | 有Followed | Collision | 0.52m |
| ID: 15 | 有Followed | Collision | 0.53m |
| ID: 27 | 有Followed | Collision | 0.50m |
| ID: 3 | 全Lost | Too far | N/A |

**共同特征**：
1. 能检测到人时疯狂前进
2. 近距离时bbox面积估计失效
3. 没有减速 → 直接撞上

---

## 改进建议

### 最小改动（让它能跑出数据）

```python
def act(self, observations, detector, episode_id):
    # ... 现有代码 ...
    
    # 在计算完 move_speed 后，添加：
    
    # 1. 速度限制
    move_speed = max(-1.0, min(1.0, move_speed))
    yaw_speed = max(-2.0, min(2.0, yaw_speed))
    
    # 2. 近距离安全阈值（基于bbox面积）
    if bbox_area > 40000:  # bbox很大说明很近
        move_speed = min(move_speed, 0)  # 不再前进，只能停或后退
    
    # 3. 或者更保守：bbox面积越大，最大速度越低
    max_speed = max(0.1, (50000 - bbox_area) / 50000 * 2.0)
    move_speed = max(-max_speed, min(max_speed, move_speed))
    
    action = [move_speed, y_speed, yaw_speed]
```

### 参数调整

```python
# 降低前进增益，减少激进性
self.kp_f = 0.3  # 从1.0降到0.3

# 提高目标bbox面积阈值（保持更远距离）
TARGET_BBOX_AREA = 50000  # 从30000提高到50000
error_f = (TARGET_BBOX_AREA - bbox_area) / 10000
```

### 根本改进

1. **使用深度传感器**
```python
depth = observations["agent_1_articulated_agent_jaw_depth"]
# 获取bbox区域的平均深度作为距离估计
target_depth = depth[rmin:rmax, cmin:cmax].mean()
error_f = (TARGET_DISTANCE - target_depth) / SCALE
```

2. **添加真实距离安全检查**
```python
# 在evaluate_agent中已经计算了真实距离
actual_dist = np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos)
if actual_dist < 1.0:
    action[0] = min(action[0], 0)  # 不能再前进
```

3. **实现减速曲线**
```python
# 距离越近，最大允许速度越低
if actual_dist < 2.0:
    max_forward_speed = actual_dist * 0.5  # 线性减速
    move_speed = min(move_speed, max_forward_speed)
```

4. **Lost状态搜索行为**
```python
if not facing:
    # 不是停下，而是旋转搜索
    if self.last_known_direction == 'left':
        action = [0, 0, 1.0]  # 向左转
    else:
        action = [0, 0, -1.0]  # 向右转
```

---

## 数据输出格式

成功时保存到 `{save_path}/{scene_key}/`：

1. **`{episode_id}.json`** - 结果摘要
```json
{
  "finish": true,
  "status": "Normal",
  "success": true,
  "following_rate": 0.85,
  "following_step": 170,
  "total_step": 200,
  "collision": 0.0,
  "instruction": "Follow the person in red shirt"
}
```

2. **`{episode_id}_info.json`** - 逐帧轨迹
```json
[
  {"step": 1, "dis_to_human": 2.5, "facing": 1.0, "base_velocity": [0.5, 0.1, 0.2]},
  {"step": 2, "dis_to_human": 2.3, "facing": 1.0, "base_velocity": [0.4, 0.1, 0.1]},
  ...
]
```

3. **`{episode_id}.mp4`** - RGB视频

---

## 评价

**品味评分**：🔴 垃圾

这是一个玩具级别的控制器。核心问题是**数据结构错了**——用2D bbox面积来估计3D距离，这个假设在任何实际场景都不成立。

正确的做法：
1. 直接用深度传感器获取距离
2. 或者用3D点云
3. 或者至少做个bbox面积到距离的标定曲线

而不是拍脑袋写个 `(30000 - bbox_area) / 10000`。
