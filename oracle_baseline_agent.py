"""
Oracle Baseline Agent v2 - 基于 Baseline 改进的专家轨迹收集器

================================================================================
核心设计思想
================================================================================

1. **保持 Baseline 的图像空间控制**
   - 转向控制：error_t = 0.5 - center_x (目标在图像中的水平位置)
   - 前进控制：error_f = (target_area - bbox_area) / 10000 (目标大小)
   - 这是经过验证有效的视觉伺服方法

2. **用 GT 信息增强**
   - 2.1 防撞：当 GT 距离 < 阈值时，强制后退
   - 2.2 追踪：当 GT 距离 > 阈值时，加速前进
   - 2.3 目标接近时主动避让：检测目标速度朝向

3. **与 Baseline 的区别**
   - Baseline：只保存成功 episode
   - Oracle：保存所有 episode（更多训练数据）
   - Oracle：有防撞逻辑（产生更安全的示范）
"""

import habitat
import habitat_sim
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from habitat.config.default_structured_configs import AgentConfig
from habitat.tasks.nav.nav import NavigationEpisode
from habitat_sim.gfx import LightInfo, LightPositionModel
from tqdm import trange
from collections import deque

import os
import os.path as osp
import imageio
import json


def evaluate_agent(config, dataset_split, save_path, target_id=None) -> None:
    """评估主循环 - 与 baseline 相同结构"""
    robot_config = OracleBaselineAgent(save_path, target_id)
    with habitat.TrackEnv(
        config=config,
        dataset=dataset_split
    ) as env:
        sim = env.sim
        robot_config.reset()
        
        num_episodes = len(env.episodes)
        for _ in trange(num_episodes):
            obs = env.reset()
            light_setup = [
                LightInfo(vector=[10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[-10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, 10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, -10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
            ]
            sim.set_light_setup(light_setup)

            result = {}
            record_infos = []

            try:
                instruction = env.current_episode.info.get('instruction', None)
            except Exception:
                instruction = None

            # 获取 agents
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent
            
            # 其他人类 (干扰者)
            other_humans = []
            for i in range(2, len(sim.agents_mgr)):
                try:
                    other_humans.append(sim.agents_mgr[i].articulated_agent)
                except:
                    pass

            iter_step = 0
            followed_step = 0
            too_far_count = 0
            status = 'Normal'
            info = env.get_metrics()

            # 传递 sim agents 给 oracle agent (包含 sim 用于 navmesh)
            robot_config.set_sim_agents(humanoid_agent_main, robot_agent, other_humans, sim)

            while not env.episode_over:
                record_info = {}
                obs = sim.get_sensor_observations()
                detector = env.task._get_observations(env.current_episode)
                
                action = robot_config.act(obs, detector, env.current_episode.episode_id)

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

                info = env.get_metrics()
                if info['human_following'] == 1.0:
                    followed_step += 1
                    too_far_count = 0

                dist = np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos)
                if dist > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        status = 'Lost'
                        break

                record_info["step"] = iter_step
                record_info["dis_to_human"] = float(dist)
                record_info["facing"] = info['human_following']
                record_info["base_velocity"] = action
                record_info["robot_pos"] = [float(x) for x in robot_agent.base_pos]
                record_info["target_pos"] = [float(x) for x in humanoid_agent_main.base_pos]
                record_infos.append(record_info)

                if info['human_collision'] == 1.0:
                    status = 'Collision'
                    break

            info = env.get_metrics()
            finished = env.episode_over

            result['finish'] = finished
            result['status'] = status
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

            # 保存所有 episode (不只是成功的)
            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(record_infos, f, indent=2)
            with open(os.path.join(save_dir, "{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result, f, indent=2)

            robot_config.reset(env.current_episode, success=True)  # 总是保存视频


class OracleBaselineAgent(AgentConfig):
    """
    Oracle Baseline Agent
    
    ================================================================================
    控制逻辑（与 Baseline 相同的图像空间控制 + GT 增强）
    ================================================================================
    
    1. 转向控制 (yaw_speed):
       - error_t = 0.5 - center_x
       - 目标偏左 (center_x < 0.5) → error_t > 0 → yaw_speed > 0 → 左转
       - 目标偏右 (center_x > 0.5) → error_t < 0 → yaw_speed < 0 → 右转
    
    2. 前进控制 (move_speed):
       - error_f = (TARGET_AREA - bbox_area) / 10000
       - 目标太小/远 → error_f > 0 → move_speed > 0 → 前进
       - 目标太大/近 → error_f < 0 → move_speed < 0 → 后退
    
    3. GT 增强（Oracle 独有）:
       - 3.1 当 GT 距离 < MIN_DISTANCE: 强制后退
       - 3.2 当 GT 距离 > MAX_DISTANCE: 增加前进速度
       - 3.3 当目标朝我们移动: 预先后退
    """
    
    # ============================================================
    # 参数配置
    # ============================================================
    
    # GT 距离阈值 (米)
    DANGER_DISTANCE = 0.8       # 紧急后退
    MIN_DISTANCE = 1.2          # 开始后退
    IDEAL_DISTANCE = 2.0        # 理想距离
    MAX_DISTANCE = 3.0          # 开始加速
    
    # 图像空间目标面积 (像素²)
    TARGET_AREA = 25000         # 理想 bbox 面积 (比 baseline 的 30000 小一点，保持更远距离)
    
    # PD 控制增益 (与 Baseline 相同)
    KP_T = 2.0                  # 转向比例
    KP_F = 1.0                  # 前进比例
    KP_Y = 0.5                  # 横移比例
    
    # 速度限制
    MAX_FORWARD = 2.0
    MAX_BACKWARD = 1.0
    MAX_YAW = 2.0
    
    def __init__(self, result_path, target_id=None):
        super().__init__()
        print("Initialize Oracle Baseline Agent v2")

        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        self.target_id = target_id
        
        self.rgb_list = []
        
        # GT agents (由 set_sim_agents 设置)
        self.target_human = None
        self.robot = None
        self.other_humans = []
        self.sim = None  # 用于 navmesh
        
        # 速度估计
        self.target_pos_history = deque(maxlen=5)
        
        # 误差历史 (PD 控制)
        self.prev_error_t = 0
        self.prev_error_f = 0
        
        self.reset()

    def set_sim_agents(self, target_human, robot, other_humans, sim=None):
        """设置 sim agents 引用"""
        self.target_human = target_human
        self.robot = robot
        self.other_humans = other_humans
        self.sim = sim  # 用于 navmesh 路径规划
        self.target_pos_history.clear()

    def reset(self, episode: NavigationEpisode = None, success: bool = False):
        """重置 + 保存视频"""
        if len(self.rgb_list) != 0 and episode is not None:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
            imageio.mimsave(output_video_path, self.rgb_list)
            self.rgb_list = []
        
        self.target_pos_history.clear()
        self.prev_error_t = 0
        self.prev_error_f = 0

    def _get_gt_distance(self):
        """获取 GT 距离 (米)"""
        if self.target_human is None or self.robot is None:
            return 2.0  # 默认值
        
        robot_pos = np.array([float(x) for x in self.robot.base_pos])
        target_pos = np.array([float(x) for x in self.target_human.base_pos])
        
        # 存储历史
        self.target_pos_history.append(target_pos.copy())
        
        # 只用 XZ 平面距离 (Y 是高度)
        dist_2d = np.sqrt((robot_pos[0] - target_pos[0])**2 + (robot_pos[2] - target_pos[2])**2)
        return dist_2d

    def _path_to_point(self, point):
        """
        使用 navmesh 获取到目标点的路径
        
        返回: 路径点列表，或 None（如果无法规划）
        """
        if self.sim is None or self.robot is None:
            return None
        
        agent_pos = np.array(self.robot.base_pos)
        
        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = np.array(point)
        found_path = self.sim.pathfinder.find_path(path)
        
        if not found_path:
            return None
        return path.points

    def _is_target_approaching(self):
        """检测目标是否在朝我们移动"""
        if len(self.target_pos_history) < 3:
            return False
        
        robot_pos = np.array([float(x) for x in self.robot.base_pos])
        
        # 计算目标位移
        prev_pos = self.target_pos_history[-3]
        curr_pos = self.target_pos_history[-1]
        target_vel = curr_pos - prev_pos
        
        # 目标到机器人的方向
        to_robot = robot_pos - curr_pos
        to_robot_2d = np.array([to_robot[0], to_robot[2]])
        to_robot_norm = np.linalg.norm(to_robot_2d)
        if to_robot_norm < 0.01:
            return False
        to_robot_2d = to_robot_2d / to_robot_norm
        
        # 目标速度在朝向机器人方向的投影
        target_vel_2d = np.array([target_vel[0], target_vel[2]])
        approach_speed = np.dot(target_vel_2d, to_robot_2d)
        
        return approach_speed > 0.05  # 阈值

    def act(self, observations, detector, episode_id):
        """
        主控制函数
        
        逻辑流程:
        1. 获取图像和 GT 信息
        2. 用图像空间 PD 控制 (和 Baseline 一样)
        3. 用 GT 距离修正 (防撞)
        """
        self.episode_id = episode_id
        
        # ============================================================
        # 1. 图像预处理
        # ============================================================
        rgb = observations["agent_1_articulated_agent_jaw_rgb"]
        rgb_ = rgb[:, :, :3]
        height, width = rgb_.shape[:2]
        self.rgb_list.append(rgb_)
        
        # ============================================================
        # 2. 获取 GT 距离
        # ============================================================
        gt_distance = self._get_gt_distance()
        target_approaching = self._is_target_approaching()
        
        # ============================================================
        # 3. 图像空间控制 (和 Baseline 完全一样)
        # ============================================================
        action = [0.0, 0.0, 0.0]
        target_visible = False
        
        # 3.1 尝试 panoptic 分割
        if self.target_id is not None and "agent_1_articulated_agent_jaw_panoptic" in observations:
            panoptic = observations["agent_1_articulated_agent_jaw_panoptic"]
            target_mask = (panoptic == self.target_id)
            if hasattr(target_mask, "ndim") and target_mask.ndim == 3:
                target_mask = target_mask[:, :, 0]
            
            if np.any(target_mask):
                rows = np.any(target_mask, axis=1)
                cols = np.any(target_mask, axis=0)
                r_idx = np.where(rows)[0]
                c_idx = np.where(cols)[0]
                rmin, rmax = int(r_idx[0]), int(r_idx[-1])
                cmin, cmax = int(c_idx[0]), int(c_idx[-1])
                
                center_x = (cmin + cmax) / (2 * width)
                bbox_area = (cmax - cmin) * (rmax - rmin)
                
                action = self._compute_pd_action(center_x, bbox_area)
                target_visible = True
        
        # 3.2 回退到 detector
        if not target_visible and detector['agent_1_main_humanoid_detector_sensor']['facing']:
            box = detector['agent_1_main_humanoid_detector_sensor']['box']
            center_x = (box[0] + box[2]) / (2 * width)
            bbox_area = (box[2] - box[0]) * (box[3] - box[1])
            
            action = self._compute_pd_action(center_x, bbox_area)
            target_visible = True
        
        # 3.3 目标不可见 - 使用 GT 位置搜索!
        if not target_visible:
            action = self._search_action()
        
        # ============================================================
        # 4. GT 距离修正 (Oracle 独有)
        # ============================================================
        move_speed, y_speed, yaw_speed = action
        
        # 4.1 防撞: 太近则后退
        if gt_distance < self.DANGER_DISTANCE:
            move_speed = -self.MAX_BACKWARD  # 紧急后退
            print(f"[Oracle] DANGER! dist={gt_distance:.2f}m, backing off!")
        elif gt_distance < self.MIN_DISTANCE:
            # 线性插值后退
            back_factor = (self.MIN_DISTANCE - gt_distance) / (self.MIN_DISTANCE - self.DANGER_DISTANCE)
            move_speed = min(move_speed, -self.MAX_BACKWARD * back_factor * 0.5)
            print(f"[Oracle] Too close! dist={gt_distance:.2f}m")
        
        # 4.2 目标朝我们来 - 预先后退
        if target_approaching and gt_distance < self.IDEAL_DISTANCE:
            move_speed = min(move_speed, -0.3)
            print(f"[Oracle] Target approaching!")
        
        # 4.3 太远 - 加速追
        if gt_distance > self.MAX_DISTANCE:
            move_speed = max(move_speed, self.MAX_FORWARD * 0.8)
            print(f"[Oracle] Too far! dist={gt_distance:.2f}m, speeding up!")
        
        # ============================================================
        # 5. Navmesh 避障修正 (防止卡墙)
        # ============================================================
        # 如果要前进，检查 navmesh 路径是否与当前方向一致
        if move_speed > 0.1 and self.sim is not None:
            target_pos = np.array([float(x) for x in self.target_human.base_pos])
            path_points = self._path_to_point(target_pos)
            
            if path_points is not None and len(path_points) > 1:
                robot_pos = np.array([float(x) for x in self.robot.base_pos])
                next_waypoint = np.array(path_points[1])
                to_waypoint = next_waypoint - robot_pos
                to_waypoint_2d = np.array([to_waypoint[0], to_waypoint[2]])
                
                # 获取机器人朝向
                try:
                    import magnum as mn
                    rot = self.robot.base_rot
                    forward_mn = rot.transform_vector(mn.Vector3(0, 0, -1))
                    robot_forward = np.array([float(forward_mn.x), float(forward_mn.z)])
                except:
                    robot_forward = np.array([0.0, -1.0])
                
                # 检查 waypoint 方向与当前朝向的夹角
                waypoint_dir = to_waypoint_2d / (np.linalg.norm(to_waypoint_2d) + 1e-6)
                robot_dir = robot_forward / (np.linalg.norm(robot_forward) + 1e-6)
                
                # 用 navmesh waypoint 修正转向
                cross = robot_dir[0] * waypoint_dir[1] - robot_dir[1] * waypoint_dir[0]
                navmesh_yaw = np.clip(cross * 2.0, -self.MAX_YAW, self.MAX_YAW)
                
                # 融合视觉转向和 navmesh 转向 (navmesh 权重更高)
                yaw_speed = 0.3 * yaw_speed + 0.7 * navmesh_yaw
        
        # ============================================================
        # 6. 输出
        # ============================================================
        action = [
            np.clip(move_speed, -self.MAX_BACKWARD, self.MAX_FORWARD),
            np.clip(y_speed, -0.5, 0.5),
            np.clip(yaw_speed, -self.MAX_YAW, self.MAX_YAW)
        ]
        
        return action

    def _search_action(self):
        """
        搜索动作 - 当目标不在视野内时使用 navmesh 路径规划
        
        核心优势：
        1. 即使看不到目标，也知道该往哪走
        2. 使用 navmesh 绕过障碍物，不会卡住!
        """
        if self.target_human is None or self.robot is None:
            return [0.0, 0.0, 0.5]  # 默认: 原地旋转搜索
        
        robot_pos = np.array([float(x) for x in self.robot.base_pos])
        target_pos = np.array([float(x) for x in self.target_human.base_pos])
        
        # ============================================================
        # 使用 navmesh 路径规划 (核心改进!)
        # ============================================================
        path_points = self._path_to_point(target_pos)
        
        if path_points is not None and len(path_points) > 1:
            # 跟随路径的下一个 waypoint，而不是直接往目标走
            next_waypoint = np.array(path_points[1])
            to_waypoint = next_waypoint - robot_pos
            to_waypoint_2d = np.array([to_waypoint[0], to_waypoint[2]])
        else:
            # 无法规划路径，直接往目标走
            to_waypoint = target_pos - robot_pos
            to_waypoint_2d = np.array([to_waypoint[0], to_waypoint[2]])
        
        # 获取机器人朝向
        try:
            import magnum as mn
            rot = self.robot.base_rot
            forward_mn = rot.transform_vector(mn.Vector3(0, 0, -1))
            robot_forward = np.array([float(forward_mn.x), float(forward_mn.z)])
        except:
            robot_forward = np.array([0.0, -1.0])
        
        # 归一化
        waypoint_dir = to_waypoint_2d / (np.linalg.norm(to_waypoint_2d) + 1e-6)
        robot_dir = robot_forward / (np.linalg.norm(robot_forward) + 1e-6)
        
        # cross > 0: waypoint 在左边，左转 (yaw > 0)
        # cross < 0: waypoint 在右边，右转 (yaw < 0)
        cross = robot_dir[0] * waypoint_dir[1] - robot_dir[1] * waypoint_dir[0]
        
        yaw_speed = np.clip(cross * 3.0, -self.MAX_YAW, self.MAX_YAW)
        move_speed = 0.5  # 稍快一点，因为有路径规划保证不撞
        
        print(f"[Oracle] SEARCH with navmesh: cross={cross:.2f}, yaw={yaw_speed:.2f}")
        
        return [move_speed, 0.0, yaw_speed]

    def _compute_pd_action(self, center_x, bbox_area):
        """
        PD 控制计算 - 与 Baseline 完全相同
        
        输入:
            center_x: 目标中心 x 坐标 (归一化, 0-1)
            bbox_area: 边界框面积 (像素²)
        
        输出:
            [move_speed, y_speed, yaw_speed]
        """
        # 转向误差: 目标应该在图像中心 (center_x = 0.5)
        error_t = 0.5 - center_x
        
        # 前进误差: 目标应该有理想大小 (bbox_area = TARGET_AREA)
        error_f = (self.TARGET_AREA - bbox_area) / 10000
        if abs(error_f) < 0.5:  # 死区
            error_f = 0
        
        # PD 控制
        derivative_t = error_t - self.prev_error_t
        derivative_f = error_f - self.prev_error_f
        
        yaw_speed = self.KP_T * error_t  # + KD_T * derivative_t (未使用)
        move_speed = self.KP_F * error_f  # + KD_F * derivative_f (未使用)
        y_speed = self.KP_Y * error_t
        
        # 更新历史
        self.prev_error_t = error_t
        self.prev_error_f = error_f
        
        return [move_speed, y_speed, yaw_speed]
