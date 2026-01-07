"""
================================================================================
baseline_agent.py - 基于边界框的人类跟踪机器人代理 (Baseline Agent)
================================================================================

【整体架构概述】
本文件实现了一个在 Habitat 仿真环境中跟踪人类目标的机器人代理。
核心思想：使用 PD 控制器（比例-微分控制）根据目标人类在图像中的位置和大小
来控制机器人的速度（前进、横移、转向），使机器人始终保持跟踪状态。

【数据流】
1. 传感器观测 (RGB/Panoptic) → 2. 目标检测/定位 → 3. PD控制计算 → 4. 速度动作输出

【主要组件】
1. evaluate_agent(): 评估主循环，运行所有 episode 并记录结果
2. GTBBoxAgent: 基于 Ground-Truth 边界框的跟踪代理类
"""

# =============================================================================
# 1. 依赖导入
# =============================================================================

# 1.1 Habitat 仿真环境核心模块
import habitat                                          # Habitat 主框架
import warnings
warnings.filterwarnings('ignore')                       # 忽略警告信息
import numpy as np                                      # 数值计算
from habitat.config.default_structured_configs import AgentConfig  # 代理配置基类
from habitat.tasks.nav.nav import NavigationEpisode     # 导航任务 Episode 定义
from habitat_sim.gfx import LightInfo, LightPositionModel  # 光照配置
from habitat.sims.habitat_simulator.actions import HabitatSimActions  # 动作定义
from tqdm import trange                                 # 进度条显示

# 1.2 系统与 I/O 模块
import os
import os.path as osp
import imageio                                          # 视频保存
import json                                             # JSON 序列化


# =============================================================================
# 2. 评估主函数 evaluate_agent()
# =============================================================================
def evaluate_agent(config, dataset_split, save_path, target_id=None) -> None:
    """
    评估代理在所有 episode 上的跟踪性能。
    
    【参数说明】
    - config: Habitat 环境配置对象
    - dataset_split: 数据集划分（包含多个 episode）
    - save_path: 结果保存路径
    - target_id: 目标人类的 ID（用于 panoptic 分割跟踪）
    
    【输出】
    - 成功的 episode 会保存：
      - {episode_id}.mp4: 跟踪视频
      - {episode_id}.json: 评估结果
      - {episode_id}_info.json: 每步详细信息
    """
    
    # -------------------------------------------------------------------------
    # 2.1 初始化：创建代理和环境
    # -------------------------------------------------------------------------
    robot_config = GTBBoxAgent(save_path, target_id)  # 创建跟踪代理实例
    
    with habitat.TrackEnv(
        config=config,
        dataset=dataset_split
    ) as env:
        sim = env.sim                                   # 获取仿真器引用
        robot_config.reset()                            # 重置代理状态
        
        # ---------------------------------------------------------------------
        # 2.2 Episode 循环：遍历所有评估场景
        # ---------------------------------------------------------------------
        num_episodes = len(env.episodes)
        for _ in trange(num_episodes):
            
            # -----------------------------------------------------------------
            # 2.2.1 Episode 初始化
            # -----------------------------------------------------------------
            obs = env.reset()                           # 重置环境，获取初始观测
            
            # 2.2.1.1 配置场景光照（四向全局光源，确保可见性）
            light_setup = [
                LightInfo(
                    vector=[10.0, -2.0, 0.0, 0.0],      # +X 方向光源
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[-10.0, -2.0, 0.0, 0.0],     # -X 方向光源
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[0.0, -2.0, 10.0, 0.0],      # +Z 方向光源
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=[0.0, -2.0, -10.0, 0.0],     # -Z 方向光源
                    color=[1.0, 1.0, 1.0],
                    model=LightPositionModel.Global,
                ),
            ]
            sim.set_light_setup(light_setup)

            # 2.2.1.2 初始化记录变量
            result = {}                                 # 最终评估结果
            record_infos = []                           # 每步详细记录

            # 2.2.1.3 获取任务指令（如果有）
            try:
                instruction = env.current_episode.info.get('instruction', None)
            except Exception:
                instruction = None

            # 2.2.1.4 初始化状态变量
            action_dict = dict()
            finished = False
            
            # 获取仿真中的代理引用
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent  # 主人类目标 (agent_0)
            robot_agent = sim.agents_mgr[1].articulated_agent          # 跟踪机器人 (agent_1)

            # 跟踪统计变量
            iter_step = 0                               # 当前步数
            followed_step = 0                           # 成功跟踪的步数
            human_no_move = 0                           # 人类未移动计数（未使用）
            too_far_count = 0                           # 距离过远的连续步数
            status = 'Normal'                           # 当前状态：Normal/Lost/Collision
            info = env.get_metrics()

            # -----------------------------------------------------------------
            # 2.2.2 Episode 主循环：逐步执行跟踪
            # -----------------------------------------------------------------
            while not env.episode_over:
                record_info = {}
                
                # 2.2.2.1 获取传感器观测
                obs = sim.get_sensor_observations()

                # 2.2.2.2 获取检测结果并计算动作
                detector = env.task._get_observations(env.current_episode)
                action = robot_config.act(obs, detector, env.current_episode.episode_id)

                # 2.2.2.3 构建动作字典
                # 动作格式：多代理动作，agent_1 使用 base_velocity 控制
                action_dict = {
                    "action": ("agent_0_humanoid_navigate_action",                    # 主人类导航
                               "agent_1_base_velocity",                                # 跟踪机器人速度控制
                               "agent_2_oracle_nav_randcoord_action_obstacle",         # 其他行人
                               "agent_3_oracle_nav_randcoord_action_obstacle",
                               "agent_4_oracle_nav_randcoord_action_obstacle",
                               "agent_5_oracle_nav_randcoord_action_obstacle"),
                    "action_args": {
                        "agent_1_base_vel" : action     # [前进速度, 横移速度, 转向速度]
                    }
                }
                
                # 2.2.2.4 执行动作
                iter_step += 1
                env.step(action_dict)

                # 2.2.2.5 评估跟踪状态
                info = env.get_metrics()
                if info['human_following'] == 1.0:      # 成功跟踪（目标在视野内）
                    print("Followed")
                    followed_step += 1
                    too_far_count = 0                   # 重置距离过远计数
                else:
                    print("Lost")

                # 2.2.2.6 检查距离约束（防止丢失目标）
                if np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos) > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:              # 连续 20 步距离过远，判定丢失
                        print("Too far from human!")
                        status = 'Lost'
                        finished = False
                        break

                # 2.2.2.7 记录当前步信息
                record_info["step"] = iter_step
                record_info["dis_to_human"] = float(np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos))
                record_info["facing"] = info['human_following']
                record_info["base_velocity"] = action
                record_infos.append(record_info)

                # 2.2.2.8 检查碰撞约束
                if info['human_collision'] == 1.0:
                    print("Collision detected!")
                    status = 'Collision'
                    finished = False
                    break
                
                print(f"========== ID: {env.current_episode.episode_id} Step now is: {iter_step} action is: {action} dis_to_main_human: {np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos)} ============")

            # -----------------------------------------------------------------
            # 2.2.3 Episode 结束：汇总结果
            # -----------------------------------------------------------------
            print("finished episode id: ", env.current_episode.episode_id)
            info = env.get_metrics()

            if env.episode_over:
                finished = True

            # 2.2.3.1 构建评估结果
            result['finish'] = finished
            result['status'] = status
            # 成功判定：短 episode 需要同时满足 human_following_success 和 human_following
            if iter_step < 300:
                result['success'] = info['human_following_success'] and info['human_following']
            else:
                result['success'] = info['human_following']
            result['following_rate'] = followed_step / iter_step   # 跟踪成功率
            result['following_step'] = followed_step
            result['total_step'] = iter_step
            result['collision'] = info['human_collision']
            if instruction is not None:
                result['instruction'] = instruction

            # 2.2.3.2 保存成功 episode 的结果
            if result['success']:
                scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split('.')[0]
                save_dir = os.path.join(save_path, scene_key)
                os.makedirs(save_dir, exist_ok=True)
                # 保存详细信息 JSON
                with open(os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w") as f:
                    json.dump(record_infos, f, indent=2)
                # 保存评估结果 JSON
                with open(os.path.join(save_dir, "{}.json".format(env.current_episode.episode_id)), "w") as f:
                    json.dump(result, f, indent=2)

            # 2.2.3.3 重置代理状态，成功时保存视频
            robot_config.reset(env.current_episode, success=result['success'])


# =============================================================================
# 3. GTBBoxAgent 类：基于边界框的跟踪代理
# =============================================================================
class GTBBoxAgent(AgentConfig):
    """
    Ground-Truth Bounding Box Agent - 基于真实边界框的人类跟踪代理。
    
    【核心控制原理】
    使用 PD 控制器（Proportional-Derivative Controller）实现视觉伺服跟踪：
    
    1. 转向控制（yaw）: 使目标人类保持在图像中心 (center_x = 0.5)
       - 误差 error_t = 0.5 - center_x
       - 目标偏左 → error_t > 0 → 左转
       - 目标偏右 → error_t < 0 → 右转
    
    2. 前进控制（forward）: 使目标边界框面积保持在理想值 (~30000 像素)
       - 误差 error_f = (30000 - bbox_area) / 10000
       - 目标太小（远）→ error_f > 0 → 前进
       - 目标太大（近）→ error_f < 0 → 后退
    
    3. 横移控制（lateral）: 辅助转向，使跟踪更平滑
       - 与转向误差相关，但使用独立的增益系数
    
    【跟踪策略优先级】
    1. 首选：使用 panoptic 分割的 target_id 精确跟踪（如果提供）
    2. 备选：使用 detector 传感器提供的边界框（Ground-Truth 检测器）
    """
    
    # -------------------------------------------------------------------------
    # 3.1 初始化方法
    # -------------------------------------------------------------------------
    def __init__(self, result_path, target_id=None):
        """
        初始化跟踪代理。
        
        【参数】
        - result_path: 结果保存路径（视频和 JSON）
        - target_id: 目标人类在 panoptic 分割中的 ID（可选）
        """
        super().__init__()
        print("Initialize gtbbox agent")

        # 3.1.1 路径配置
        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        self.target_id = target_id                      # Panoptic 分割目标 ID
        
        # 3.1.2 视频帧缓存
        self.rgb_list = []                              # RGB 帧列表，用于保存视频
        self.rgb_box_list = []                          # 带边界框的帧（未使用）

        # 3.1.3 PD 控制器参数
        # 转向控制（Turn/Yaw）
        self.kp_t = 2                                   # 转向比例增益
        self.kd_t = 0                                   # 转向微分增益（当前未使用）
        
        # 前进控制（Forward）
        self.kp_f = 1                                   # 前进比例增益
        self.kd_f = 0                                   # 前进微分增益（当前未使用）
        
        # 横移控制（Lateral/Y-axis）- 独立于转向，提供更细腻的跟踪
        self.kp_y = 0.5                                 # 横移比例增益
        self.kd_y = 0                                   # 横移微分增益（当前未使用）

        # 3.1.4 PD 控制器状态（用于计算微分项）
        self.prev_error_t = 0                           # 上一步转向误差
        self.prev_error_f = 0                           # 上一步前进误差

        # 3.1.5 状态标志
        self.first_inside = True                        # 是否首次进入跟踪状态

        self.reset()

    # -------------------------------------------------------------------------
    # 3.2 重置方法
    # -------------------------------------------------------------------------
    def reset(self, episode: NavigationEpisode = None, success: bool = False):
        """
        重置代理状态。在 episode 结束时调用。
        
        【功能】
        1. 如果成功且有帧数据，保存视频到文件
        2. 清空帧缓存
        3. 重置状态标志
        
        【参数】
        - episode: 当前 episode 对象（用于获取 scene_id 和 episode_id）
        - success: 是否成功完成跟踪任务
        """
        # 3.2.1 保存成功 episode 的视频
        if len(self.rgb_list) != 0 and episode is not None:
            if success:
                scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
                save_dir = os.path.join(self.result_path, scene_key)
                os.makedirs(save_dir, exist_ok=True)
                output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
                imageio.mimsave(output_video_path, self.rgb_list)
                print(f"Successfully saved the episode video with episode id {episode.episode_id}")
            self.rgb_list = []                          # 清空帧缓存
        
        # 3.2.2 重置状态标志
        self.first_inside = True

    # -------------------------------------------------------------------------
    # 3.3 核心动作方法
    # -------------------------------------------------------------------------
    def act(self, observations, detector, episode_id):
        """
        根据观测计算控制动作。这是代理的核心决策函数。
        
        【输入】
        - observations: 传感器观测字典，包含：
          - agent_1_articulated_agent_jaw_rgb: RGB 图像 [H, W, 4]
          - agent_1_articulated_agent_jaw_panoptic: Panoptic 分割图（可选）
        - detector: 任务检测器输出，包含目标边界框
        - episode_id: 当前 episode ID
        
        【输出】
        - action: [前进速度, 横移速度, 转向速度] 三元组
        
        【控制逻辑流程】
        1. 尝试使用 panoptic 分割跟踪 target_id（精确）
        2. 若失败，回退到 detector 边界框跟踪
        3. 若目标不在视野，停止运动
        """
        self.episode_id = episode_id
        
        # ---------------------------------------------------------------------
        # 3.3.1 图像预处理
        # ---------------------------------------------------------------------
        rgb = observations["agent_1_articulated_agent_jaw_rgb"]  # [H, W, 4] RGBA
        rgb_ = rgb[:, :, :3]                            # 去除 alpha 通道
        image = np.asarray(rgb_[:, :, ::-1])            # RGB → BGR（OpenCV 格式）
        height, width = image.shape[:2]
        
        # 默认动作：停止
        action = [0, 0, 0]

        # ---------------------------------------------------------------------
        # 3.3.2 策略一：使用 Panoptic 分割进行精确跟踪
        # ---------------------------------------------------------------------
        target_tracked = False
        if self.target_id is not None and "agent_1_articulated_agent_jaw_panoptic" in observations:
            panoptic = observations["agent_1_articulated_agent_jaw_panoptic"]
            
            # 3.3.2.1 提取目标掩码
            target_mask = (panoptic == self.target_id)
            if hasattr(target_mask, "ndim") and target_mask.ndim == 3:
                target_mask = target_mask[:, :, 0]      # 处理 [H, W, 1] 的情况
            
            # 3.3.2.2 从掩码计算边界框
            if np.any(target_mask):
                rows = np.any(target_mask, axis=1)      # 每行是否有目标像素
                cols = np.any(target_mask, axis=0)      # 每列是否有目标像素
                r_idx = np.where(rows)[0]
                c_idx = np.where(cols)[0]
                rmin, rmax = int(r_idx[0]), int(r_idx[-1])
                cmin, cmax = int(c_idx[0]), int(c_idx[-1])
                box = np.array([cmin, rmin, cmax, rmax], dtype=np.float32)
                
                # 归一化边界框 [center_x, center_y, width, height]
                best_box = np.array([
                    (box[0] + box[2]) / (2 * width),    # 中心 x（归一化）
                    (box[1] + box[3]) / (2 * height),   # 中心 y（归一化）
                    (box[2] - box[0]) / width,          # 宽度（归一化）
                    (box[3] - box[1]) / height,         # 高度（归一化）
                ], dtype=np.float32)

                # 3.3.2.3 计算控制误差
                center_x = best_box[0]
                error_t = 0.5 - center_x                # 转向误差：目标偏离中心的程度
                
                bbox_area = (box[2] - box[0]) * (box[3] - box[1])  # 边界框面积（像素）
                error_f = (30000 - bbox_area) / 10000   # 前进误差：与理想面积的差距
                if abs(error_f) < 0.5:                  # 死区：面积接近理想值时不调整
                    error_f = 0

                # 3.3.2.4 PD 控制计算
                derivative_t = error_t - self.prev_error_t  # 转向误差变化率
                derivative_f = error_f - self.prev_error_f  # 前进误差变化率

                yaw_speed = self.kp_t * error_t + self.kd_t * derivative_t    # 转向速度
                move_speed = self.kp_f * error_f + self.kd_f * derivative_f   # 前进速度
                y_speed = self.kp_y * error_t + self.kd_y * derivative_t      # 横移速度

                # 更新误差历史
                self.prev_error_t = error_t
                self.prev_error_f = error_f

                action = [move_speed, y_speed, yaw_speed]
                target_tracked = True

        # ---------------------------------------------------------------------
        # 3.3.3 策略二：使用 Detector 边界框进行跟踪（回退方案）
        # ---------------------------------------------------------------------
        if not target_tracked:
            # 检查 detector 是否检测到目标
            if detector['agent_1_main_humanoid_detector_sensor']['facing']:
                box = detector['agent_1_main_humanoid_detector_sensor']['box']
                
                # 3.3.3.1 归一化边界框
                best_box = np.array([
                    (box[0]+box[2])/(2*width),          # 中心 x
                    (box[1]+box[3])/(2*height),         # 中心 y
                    (box[2]-box[0])/width,              # 宽度
                    (box[3]-box[1])/height              # 高度
                ], dtype=np.float32)
                
                center_x = best_box[0]

                # 3.3.3.2 计算控制误差
                error_t = 0.5 - center_x
                error_f = (30000 - (box[2]-box[0])*(box[3]-box[1])) / 10000
                if abs(error_f) < 0.5:
                    error_f = 0

                # 3.3.3.3 PD 控制计算
                derivative_t = error_t - self.prev_error_t
                derivative_f = error_f - self.prev_error_f

                yaw_speed = self.kp_t * error_t + self.kd_t * derivative_t
                move_speed = self.kp_f * error_f + self.kd_f * derivative_f
                y_speed = self.kp_y * error_t + self.kd_y * derivative_t

                self.prev_error_t = error_t
                self.prev_error_f = error_f

                action = [move_speed, y_speed, yaw_speed]
            else:
                # 3.3.3.4 目标不在视野：停止运动
                action = [0, 0, 0]
        
        # ---------------------------------------------------------------------
        # 3.3.4 记录与返回
        # ---------------------------------------------------------------------
        self.last_action = action                       # 保存上一次动作
        self.rgb_list.append(rgb_)                      # 添加帧到视频缓存

        return action
