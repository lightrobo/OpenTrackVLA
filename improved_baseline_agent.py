"""
Improved Baseline Agent for collecting better training data.
改进版基线智能体，用于收集更高质量的训练数据。

================================================================================
代码结构概览 (Code Structure Overview)
================================================================================

1. 模块导入与依赖
   1.1. Habitat仿真环境相关
   1.2. 数值计算与数据结构
   1.3. 文件操作与序列化

2. evaluate_agent() - 主评估函数
   2.1. 环境初始化与灯光设置
   2.2. Episode主循环
   2.3. 结果记录与保存

3. ImprovedBaselineAgent类 - 改进的基线智能体
   3.1. __init__() - 初始化
        3.1.1. PD控制器增益参数
        3.1.2. 目标追踪历史状态
        3.1.3. 目标丢失状态管理
        3.1.4. 动作平滑与噪声注入
   3.2. reset() - 状态重置
   3.3. _update_motion_prediction() - 运动预测更新
   3.4. _predict_target_position() - 目标位置预测
   3.5. _search_action() - 目标丢失时的搜索策略
   3.6. _smooth_action() - 动作平滑处理
   3.7. _add_training_noise() - 训练数据多样性噪声
   3.8. act() - 主行为决策函数
   3.9. _compute_tracking_action() - 跟踪动作计算

================================================================================
Key improvements over baseline_agent.py:
相比基础版的关键改进：
================================================================================
1. Target motion prediction using exponential moving average
   使用指数移动平均(EMA)进行目标运动预测
   
2. Search strategy when target is lost (turn toward last seen direction)
   目标丢失时的搜索策略（朝最后看到的方向转向）
   
3. Smooth action output to avoid jerky movements
   平滑动作输出，避免机器人动作抖动
   
4. Better distance control with adaptive gains
   使用自适应增益的更好距离控制
   
5. Diversity injection for richer training data
   注入多样性噪声以获取更丰富的训练数据
"""

# =============================================================================
# 1. 模块导入与依赖 (Module Imports and Dependencies)
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1. Habitat仿真环境相关 (Habitat Simulation Environment)
# -----------------------------------------------------------------------------
import habitat                                    # Habitat-Lab主框架
import warnings
warnings.filterwarnings('ignore')                 # 抑制警告信息
from habitat.config.default_structured_configs import AgentConfig  # 智能体配置基类
from habitat.tasks.nav.nav import NavigationEpisode               # 导航episode类型
from habitat_sim.gfx import LightInfo, LightPositionModel         # 灯光配置

# -----------------------------------------------------------------------------
# 1.2. 数值计算与数据结构 (Numerical Computing and Data Structures)
# -----------------------------------------------------------------------------
import numpy as np                    # 数值计算
from collections import deque         # 双端队列，用于历史位置缓存
from tqdm import trange               # 进度条显示

# -----------------------------------------------------------------------------
# 1.3. 文件操作与序列化 (File Operations and Serialization)
# -----------------------------------------------------------------------------
import os
import os.path as osp
import imageio                        # 视频保存
import json                           # JSON序列化


# =============================================================================
# 2. evaluate_agent() - 主评估函数 (Main Evaluation Function)
# =============================================================================
def evaluate_agent(config, dataset_split, save_path, target_id=None) -> None:
    """
    主评估循环函数。
    
    功能说明:
    - 创建Habitat环境并运行所有episodes
    - 收集机器人跟踪人类目标的轨迹数据
    - 记录每步的状态信息并保存为训练数据
    
    参数:
        config: Habitat环境配置
        dataset_split: 数据集分割（训练/验证/测试）
        save_path: 结果保存路径
        target_id: 目标对象的panoptic ID（可选）
    """
    
    # -------------------------------------------------------------------------
    # 2.1. 环境初始化 (Environment Initialization)
    # -------------------------------------------------------------------------
    robot_config = ImprovedBaselineAgent(save_path, target_id)
    
    with habitat.TrackEnv(
        config=config,
        dataset=dataset_split
    ) as env:
        sim = env.sim
        robot_config.reset()
        
        num_episodes = len(env.episodes)
        
        # ---------------------------------------------------------------------
        # 2.2. Episode主循环 (Main Episode Loop)
        # ---------------------------------------------------------------------
        for _ in trange(num_episodes):
            obs = env.reset()
            
            # 2.2.1. 灯光设置 - 四向全局光源，确保场景均匀照明
            # (Light Setup - Four directional global lights for uniform illumination)
            light_setup = [
                LightInfo(vector=[10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),   # 东向光
                LightInfo(vector=[-10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),  # 西向光
                LightInfo(vector=[0.0, -2.0, 10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),   # 南向光
                LightInfo(vector=[0.0, -2.0, -10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),  # 北向光
            ]
            sim.set_light_setup(light_setup)

            # 2.2.2. 结果记录变量初始化
            result = {}                # episode最终结果
            record_infos = []          # 每步详细信息列表

            # 2.2.3. 获取任务指令（如果存在）
            try:
                instruction = env.current_episode.info.get('instruction', None)
            except Exception:
                instruction = None

            # 2.2.4. 获取智能体引用
            # agent_0: 主要人形目标（被跟踪者）
            # agent_1: 跟踪机器人（我们控制的）
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent

            # 2.2.5. 状态计数器初始化
            iter_step = 0              # 当前步数
            followed_step = 0          # 成功跟踪的步数
            too_far_count = 0          # 距离过远的连续帧数
            status = 'Normal'          # episode状态: Normal/Lost/Collision
            info = env.get_metrics()

            # -----------------------------------------------------------------
            # 2.2.6. Step循环 - 核心交互过程 (Step Loop - Core Interaction)
            # -----------------------------------------------------------------
            while not env.episode_over:
                record_info = {}
                
                # 获取传感器观测
                obs = sim.get_sensor_observations()
                # 获取目标检测结果
                detector = env.task._get_observations(env.current_episode)
                # 智能体决策
                action = robot_config.act(obs, detector, env.current_episode.episode_id)

                # 构建多智能体动作字典
                # agent_0: 人形导航动作
                # agent_1: 机器人基座速度控制（我们的输出）
                # agent_2-5: 其他oracle导航障碍物智能体
                action_dict = {
                    "action": ("agent_0_humanoid_navigate_action", "agent_1_base_velocity", 
                               "agent_2_oracle_nav_randcoord_action_obstacle", 
                               "agent_3_oracle_nav_randcoord_action_obstacle", 
                               "agent_4_oracle_nav_randcoord_action_obstacle", 
                               "agent_5_oracle_nav_randcoord_action_obstacle"),
                    "action_args": {"agent_1_base_vel": action}
                }
                
                iter_step += 1
                env.step(action_dict)

                # 2.2.7. 跟踪成功判定
                info = env.get_metrics()
                if info['human_following'] == 1.0:
                    followed_step += 1
                    too_far_count = 0

                # 2.2.8. 目标丢失检测 - 距离超过4米且持续20帧则判定丢失
                if np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos) > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        status = 'Lost'
                        break

                # 2.2.9. 记录当前步信息
                record_info["step"] = iter_step
                record_info["dis_to_human"] = float(np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos))
                record_info["facing"] = info['human_following']
                record_info["base_velocity"] = action
                record_infos.append(record_info)

                # 2.2.10. 碰撞检测
                if info['human_collision'] == 1.0:
                    status = 'Collision'
                    break

            # -----------------------------------------------------------------
            # 2.3. 结果记录与保存 (Result Recording and Saving)
            # -----------------------------------------------------------------
            info = env.get_metrics()
            finished = env.episode_over

            # 2.3.1. 汇总episode结果
            result['finish'] = finished
            result['status'] = status
            # 成功判定逻辑：短episode需要最终跟踪成功，长episode只需当前帧跟踪
            if iter_step < 300:
                result['success'] = info['human_following_success'] and info['human_following']
            else:
                result['success'] = info['human_following']
            result['following_rate'] = followed_step / max(iter_step, 1)
            result['following_step'] = followed_step
            result['total_step'] = iter_step
            result['collision'] = info['human_collision']
            if instruction is not None:
                result['instruction'] = instruction

            # 2.3.2. 保存所有episodes的数据（不仅是成功的，用于更多样化的训练数据）
            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存每步详细信息
            with open(os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(record_infos, f, indent=2)
            # 保存episode总结
            with open(os.path.join(save_dir, "{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result, f, indent=2)

            # 重置智能体状态并保存视频
            robot_config.reset(env.current_episode, success=result['success'])


# =============================================================================
# 3. ImprovedBaselineAgent类 - 改进的基线智能体
#    (Improved Baseline Agent Class)
# =============================================================================
class ImprovedBaselineAgent(AgentConfig):
    """
    改进版基线智能体，相比原始版本增加以下功能:
    
    核心改进:
    - 目标运动预测: 使用位置变化的指数移动平均(EMA)预测目标轨迹
    - 搜索策略: 目标丢失时朝最后看到的方向搜索
    - 动作平滑: 指数平滑输出，避免机器人抖动
    - 自适应距离控制: 使用PD控制器而非纯P控制
    - 多样性注入: 添加小幅随机扰动，增加训练数据多样性
    """
    
    # -------------------------------------------------------------------------
    # 3.1. __init__() - 初始化 (Initialization)
    # -------------------------------------------------------------------------
    def __init__(self, result_path, target_id=None):
        """
        初始化改进版基线智能体。
        
        参数:
            result_path: 结果保存路径
            target_id: panoptic分割中目标对象的ID（可选）
        """
        super().__init__()
        print("Initialize improved baseline agent")

        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        self.target_id = target_id
        
        # RGB帧缓存，用于保存视频
        self.rgb_list = []

        # ---------------------------------------------------------------------
        # 3.1.1. PD控制器增益参数 (PD Controller Gains)
        # ---------------------------------------------------------------------
        # 相比基线版本增加了微分项(D)，使控制更平滑稳定
        # 
        # 角度/偏航控制 (Yaw Control):
        self.kp_t = 2.0   # 偏航比例增益 - 控制转向响应速度
        self.kd_t = 0.3   # 偏航微分增益 - 抑制转向振荡（新增！）
        
        # 前进控制 (Forward Control):
        self.kp_f = 1.0   # 前进比例增益 - 控制接近速度
        self.kd_f = 0.2   # 前进微分增益 - 抑制距离振荡（新增！）
        
        # 横向控制 (Lateral Control):
        self.kp_y = 0.5   # 横向比例增益 - 控制横移响应
        self.kd_y = 0.1   # 横向微分增益 - 抑制横移振荡（新增！）

        # ---------------------------------------------------------------------
        # 3.1.2. 目标追踪历史状态 (Target Tracking History State)
        # ---------------------------------------------------------------------
        # 存储最近10帧的目标位置信息: (center_x, center_y, bbox_area)
        self.position_history = deque(maxlen=10)
        # 目标速度的指数移动平均，用于预测下一帧位置
        self.velocity_ema = np.array([0.0, 0.0])
        # EMA平滑因子: 值越大，对新数据响应越快
        self.ema_alpha = 0.3
        
        # ---------------------------------------------------------------------
        # 3.1.3. 目标丢失状态管理 (Lost Target State Management)
        # ---------------------------------------------------------------------
        # 自上次看到目标以来的帧数
        self.frames_since_seen = 0
        # 目标最后出现的方向: -1=左侧, +1=右侧, 0=中间
        self.last_seen_direction = 0.0
        # 目标最后的估计距离（归一化）
        self.last_seen_distance = 1.0
        
        # ---------------------------------------------------------------------
        # 3.1.4. 动作平滑与噪声注入 (Action Smoothing and Noise Injection)
        # ---------------------------------------------------------------------
        # 上一帧的动作输出，用于平滑
        self.last_action = np.array([0.0, 0.0, 0.0])
        # 动作平滑系数: 值越大，动作变化越平滑但响应越慢
        self.action_smoothing = 0.7
        
        # 误差追踪，用于PD控制器的微分项
        self.prev_error_t = 0   # 上一帧角度误差
        self.prev_error_f = 0   # 上一帧距离误差
        
        # 多样性噪声参数（用于生成更丰富的训练数据）
        self.noise_scale = 0.05  # 噪声标准差
        self.add_noise = True    # 是否添加噪声

        self.reset()

    # -------------------------------------------------------------------------
    # 3.2. reset() - 状态重置 (State Reset)
    # -------------------------------------------------------------------------
    def reset(self, episode: NavigationEpisode = None, success: bool = False):
        """
        重置智能体状态。在每个新episode开始时调用。
        
        功能:
        - 保存上一个episode的视频（如果有）
        - 清空所有追踪状态
        
        参数:
            episode: 刚结束的episode对象（用于确定保存路径）
            success: 该episode是否成功（当前未使用，保留用于未来筛选）
        """
        # 3.2.1. 保存上一个episode的视频
        if len(self.rgb_list) != 0 and episode is not None:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
            imageio.mimsave(output_video_path, self.rgb_list)
            self.rgb_list = []
        
        # 3.2.2. 重置所有追踪状态
        self.position_history.clear()              # 清空位置历史
        self.velocity_ema = np.array([0.0, 0.0])   # 重置速度估计
        self.frames_since_seen = 0                 # 重置丢失计数
        self.last_seen_direction = 0.0             # 重置最后方向
        self.last_seen_distance = 1.0              # 重置最后距离
        self.last_action = np.array([0.0, 0.0, 0.0])  # 重置动作缓存
        self.prev_error_t = 0                      # 重置角度误差
        self.prev_error_f = 0                      # 重置距离误差

    # -------------------------------------------------------------------------
    # 3.3. _update_motion_prediction() - 运动预测更新
    #      (Motion Prediction Update)
    # -------------------------------------------------------------------------
    def _update_motion_prediction(self, center_x, center_y, bbox_area):
        """
        更新目标运动预测模型。
        
        原理:
        使用指数移动平均(EMA)估计目标在图像空间中的速度。
        EMA公式: v_new = α * v_current + (1-α) * v_old
        其中α=0.3，较小的α使估计更平滑但响应较慢。
        
        参数:
            center_x: 目标中心的归一化x坐标 [0, 1]
            center_y: 目标中心的归一化y坐标 [0, 1]
            bbox_area: 边界框面积（像素²），用于估计距离
        """
        current_pos = np.array([center_x, center_y])
        
        # 3.3.1. 计算速度并更新EMA
        if len(self.position_history) > 0:
            # 获取上一帧位置
            prev_pos = np.array([self.position_history[-1][0], self.position_history[-1][1]])
            # 计算当前帧速度（位置差分）
            velocity = current_pos - prev_pos
            # 使用EMA更新速度估计
            self.velocity_ema = self.ema_alpha * velocity + (1 - self.ema_alpha) * self.velocity_ema
        
        # 3.3.2. 添加当前帧到历史
        self.position_history.append((center_x, center_y, bbox_area))
        
        # 3.3.3. 更新"最后看到"的状态信息
        self.last_seen_direction = np.sign(center_x - 0.5)  # 目标在图像哪一侧
        self.last_seen_distance = np.sqrt(bbox_area) / 200  # 粗略距离估计（bbox越大越近）
        self.frames_since_seen = 0  # 重置丢失计数

    # -------------------------------------------------------------------------
    # 3.4. _predict_target_position() - 目标位置预测
    #      (Target Position Prediction)
    # -------------------------------------------------------------------------
    def _predict_target_position(self):
        """
        基于运动历史预测目标下一帧的位置。
        
        原理:
        使用恒速模型(Constant Velocity Model)，假设目标以EMA估计的速度继续运动。
        预测位置 = 当前位置 + 速度EMA
        
        返回:
            预测的下一帧位置 [x, y]，如果历史数据不足则返回None
        """
        # 需要至少2帧历史才能估计速度
        if len(self.position_history) < 2:
            return None
        
        # 获取最后已知位置
        last_pos = np.array([self.position_history[-1][0], self.position_history[-1][1]])
        # 使用速度EMA进行线性外推
        predicted = last_pos + self.velocity_ema
        return predicted

    # -------------------------------------------------------------------------
    # 3.5. _search_action() - 目标丢失时的搜索策略
    #      (Search Strategy When Target is Lost)
    # -------------------------------------------------------------------------
    def _search_action(self):
        """
        当目标不可见时生成搜索动作。
        
        策略分三个阶段:
        
        阶段1 (0-5帧): 惯性阶段
            - 继续执行上一个动作，但逐渐衰减
            - 假设目标只是暂时被遮挡，很快会重新出现
        
        阶段2 (5-30帧): 定向搜索
            - 朝目标最后出现的方向转向
            - 同时缓慢前进以探索环境
        
        阶段3 (30+帧): 旋转搜索
            - 原地旋转360度搜索目标
            - 作为最后的恢复手段
        
        返回:
            搜索动作 [forward_speed, lateral_speed, yaw_speed]
        """
        self.frames_since_seen += 1
        
        # 阶段1: 惯性延续 (Inertia Phase)
        # 目标可能只是暂时被遮挡，保持当前轨迹
        if self.frames_since_seen < 5:
            # 动作指数衰减: 0.8^frames
            decay = 0.8 ** self.frames_since_seen
            return self.last_action * decay
        
        # 阶段2: 定向搜索 (Directed Search)
        # 朝最后看到目标的方向转向
        if self.frames_since_seen < 30:
            # 朝最后看到目标的方向旋转
            yaw_speed = self.last_seen_direction * 0.5
            # 同时缓慢前进，增加搜索范围
            forward_speed = 0.2
            return np.array([forward_speed, 0.0, yaw_speed])
        
        # 阶段3: 旋转搜索 (Spinning Search)
        # 原地快速旋转，尝试重新发现目标
        yaw_speed = 0.8  # 较快的旋转速度
        return np.array([0.0, 0.0, yaw_speed])

    # -------------------------------------------------------------------------
    # 3.6. _smooth_action() - 动作平滑处理
    #      (Action Smoothing)
    # -------------------------------------------------------------------------
    def _smooth_action(self, action):
        """
        对动作输出应用指数平滑，避免机器人动作抖动。
        
        原理:
        平滑动作 = α * 上一动作 + (1-α) * 当前动作
        其中α=0.7（action_smoothing），较大的值使动作更平滑。
        
        参数:
            action: 当前计算的原始动作
            
        返回:
            平滑后的动作
        """
        action = np.array(action)
        # 指数平滑: 70%权重给历史，30%给当前
        smoothed = self.action_smoothing * self.last_action + (1 - self.action_smoothing) * action
        # 更新历史
        self.last_action = smoothed
        return smoothed

    # -------------------------------------------------------------------------
    # 3.7. _add_training_noise() - 训练数据多样性噪声
    #      (Training Data Diversity Noise)
    # -------------------------------------------------------------------------
    def _add_training_noise(self, action):
        """
        添加小幅随机噪声以增加训练数据多样性。
        
        动机:
        纯确定性策略会产生高度相关的训练数据。
        小幅噪声可以:
        - 增加状态空间覆盖
        - 模拟真实世界的传感器噪声和执行误差
        - 避免模型过拟合到特定轨迹
        
        参数:
            action: 原始动作
            
        返回:
            添加噪声后的动作
        """
        if not self.add_noise:
            return action
        # 高斯噪声，标准差=0.05
        noise = np.random.normal(0, self.noise_scale, size=3)
        return action + noise

    # -------------------------------------------------------------------------
    # 3.8. act() - 主行为决策函数
    #      (Main Action Decision Function)
    # -------------------------------------------------------------------------
    def act(self, observations, detector, episode_id):
        """
        根据观测生成动作的主函数。
        
        决策流程:
        1. 尝试通过panoptic分割定位目标（如果指定了target_id）
        2. 如果panoptic失败，回退到目标检测器
        3. 如果目标不可见，使用搜索策略
        4. 对动作进行平滑和噪声注入
        5. 裁剪到合理范围
        
        参数:
            observations: 传感器观测字典
            detector: 目标检测结果
            episode_id: 当前episode ID
            
        返回:
            动作列表 [forward_speed, lateral_speed, yaw_speed]
        """
        self.episode_id = episode_id
        
        # 3.8.1. 获取RGB图像
        rgb = observations["agent_1_articulated_agent_jaw_rgb"]
        rgb_ = rgb[:, :, :3]  # 去除可能的alpha通道
        height, width = rgb_.shape[:2]
        
        action = np.array([0.0, 0.0, 0.0])
        target_visible = False

        # ---------------------------------------------------------------------
        # 3.8.2. 方法1: Panoptic分割定位（优先级更高，更精确）
        # ---------------------------------------------------------------------
        if self.target_id is not None and "agent_1_articulated_agent_jaw_panoptic" in observations:
            panoptic = observations["agent_1_articulated_agent_jaw_panoptic"]
            # 提取目标掩码
            target_mask = (panoptic == self.target_id)
            # 处理可能的多通道情况
            if hasattr(target_mask, "ndim") and target_mask.ndim == 3:
                target_mask = target_mask[:, :, 0]
            
            if np.any(target_mask):
                # 3.8.2.1. 从掩码计算边界框
                rows = np.any(target_mask, axis=1)
                cols = np.any(target_mask, axis=0)
                r_idx = np.where(rows)[0]
                c_idx = np.where(cols)[0]
                rmin, rmax = int(r_idx[0]), int(r_idx[-1])
                cmin, cmax = int(c_idx[0]), int(c_idx[-1])
                
                # 3.8.2.2. 计算归一化中心和面积
                center_x = (cmin + cmax) / (2 * width)   # 归一化到[0,1]
                center_y = (rmin + rmax) / (2 * height)  # 归一化到[0,1]
                bbox_area = (cmax - cmin) * (rmax - rmin)  # 像素面积
                
                # 3.8.2.3. 计算跟踪动作并更新运动预测
                action = self._compute_tracking_action(center_x, bbox_area, width, height)
                self._update_motion_prediction(center_x, center_y, bbox_area)
                target_visible = True

        # ---------------------------------------------------------------------
        # 3.8.3. 方法2: 回退到目标检测器
        # ---------------------------------------------------------------------
        if not target_visible and detector['agent_1_main_humanoid_detector_sensor']['facing']:
            box = detector['agent_1_main_humanoid_detector_sensor']['box']
            # 从检测框计算中心和面积
            center_x = (box[0] + box[2]) / (2 * width)
            center_y = (box[1] + box[3]) / (2 * height)
            bbox_area = (box[2] - box[0]) * (box[3] - box[1])
            
            action = self._compute_tracking_action(center_x, bbox_area, width, height)
            self._update_motion_prediction(center_x, center_y, bbox_area)
            target_visible = True

        # ---------------------------------------------------------------------
        # 3.8.4. 目标不可见 - 使用搜索策略
        # ---------------------------------------------------------------------
        if not target_visible:
            action = self._search_action()
        
        # ---------------------------------------------------------------------
        # 3.8.5. 后处理: 平滑、噪声、裁剪
        # ---------------------------------------------------------------------
        action = self._smooth_action(action)      # 平滑动作
        action = self._add_training_noise(action)  # 添加噪声
        action = np.clip(action, -2.0, 2.0)        # 裁剪到合理范围
        
        # 保存RGB帧用于视频
        self.rgb_list.append(rgb_)
        return action.tolist()

    # -------------------------------------------------------------------------
    # 3.9. _compute_tracking_action() - 跟踪动作计算
    #      (Tracking Action Computation)
    # -------------------------------------------------------------------------
    def _compute_tracking_action(self, center_x, bbox_area, width, height):
        """
        计算跟踪目标的控制动作。
        
        控制目标:
        - 角度控制: 将目标保持在图像中心 (center_x = 0.5)
        - 距离控制: 将bbox面积保持在目标值 (表示合适的跟踪距离)
        
        使用PD控制器而非纯P控制，微分项可以:
        - 预测误差变化趋势
        - 抑制振荡
        - 提高响应速度
        
        参数:
            center_x: 目标中心的归一化x坐标
            bbox_area: 边界框面积
            width, height: 图像尺寸
            
        返回:
            控制动作 [forward_speed, lateral_speed, yaw_speed]
        """
        
        # 3.9.1. 角度误差计算
        # 目标：将目标保持在图像中心(0.5)
        # error_t > 0: 目标在左侧，需要左转
        # error_t < 0: 目标在右侧，需要右转
        error_t = 0.5 - center_x
        
        # 3.9.2. 距离误差计算
        # 目标bbox面积 = 25000像素² (比baseline的30000更近一些)
        # 面积越大表示越近，面积越小表示越远
        target_area = 25000
        error_f = (target_area - bbox_area) / 10000  # 归一化误差
        
        # 3.9.3. 距离死区 - 避免在接近目标距离时来回振荡
        # 只有误差大于0.3时才进行距离调整
        if abs(error_f) < 0.3:
            error_f = 0
        
        # 3.9.4. 预测补偿 - 根据目标运动趋势调整转向
        # 这使机器人可以"提前转向"，跟踪更顺畅
        predicted = self._predict_target_position()
        if predicted is not None:
            # 计算预测位置的误差
            predicted_error_t = 0.5 - predicted[0]
            # 混合当前误差和预测误差: 70%当前 + 30%预测
            error_t = 0.7 * error_t + 0.3 * predicted_error_t
        
        # 3.9.5. PD控制器计算
        # P项: 与误差成正比
        # D项: 与误差变化率成正比（抑制振荡）
        derivative_t = error_t - self.prev_error_t  # 角度误差变化
        derivative_f = error_f - self.prev_error_f  # 距离误差变化
        
        # 偏航速度 = Kp_t * error + Kd_t * d(error)/dt
        yaw_speed = self.kp_t * error_t + self.kd_t * derivative_t
        # 前进速度 = Kp_f * error + Kd_f * d(error)/dt
        move_speed = self.kp_f * error_f + self.kd_f * derivative_f
        # 横向速度（辅助转向）
        y_speed = self.kp_y * error_t + self.kd_y * derivative_t
        
        # 3.9.6. 保存当前误差用于下一帧的微分计算
        self.prev_error_t = error_t
        self.prev_error_f = error_f
        
        return np.array([move_speed, y_speed, yaw_speed])
