"""
Oracle Baseline Agent v3 - 简化版，基于 Baseline 改进

核心原则：
1. 图像空间 PD 控制 (和 Baseline 完全一样)
2. 只在目标不可见时用 GT 位置搜索
3. GT 距离防撞 (太近后退)
4. 不做过度工程！
"""

import habitat
import habitat_sim
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
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
    """评估主循环"""
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

            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent
            
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

            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(record_infos, f, indent=2)
            with open(os.path.join(save_dir, "{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result, f, indent=2)

            robot_config.reset(env.current_episode, success=True)


class OracleBaselineAgent(AgentConfig):
    """
    Oracle Baseline Agent v3 - 简化版
    
    和 Baseline 完全相同的 PD 控制，只增加：
    1. GT 距离防撞
    2. GT 位置搜索 (目标不可见时)
    """
    
    # 距离阈值 (米)
    DANGER_DISTANCE = 0.8
    MIN_DISTANCE = 1.2
    
    # PD 增益 (和 Baseline 完全一样)
    KP_T = 2      # 转向
    KP_F = 1      # 前进
    KP_Y = 0.5    # 横移
    
    # 目标面积 (和 Baseline 一样)
    TARGET_AREA = 30000
    
    # 2D 可视化参数
    VIS_2D_SIZE = 512  # 画布大小
    VIS_2D_NAVMESH_SAMPLES = 50  # 每轴采样点数
    
    def __init__(self, result_path, target_id=None):
        super().__init__()
        print("Initialize Oracle Baseline Agent v3 (simplified)")

        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        self.target_id = target_id
        
        self.rgb_list = []
        self.vis_2d_list = []  # 2D 可视化帧
        
        self.target_human = None
        self.robot = None
        self.other_humans = []
        self.sim = None  # 用于 navmesh
        
        # 2D 可视化缓存
        self.navmesh_img = None  # 缓存的 navmesh 背景
        self.world_bounds = None  # (x_min, x_max, z_min, z_max)
        self.robot_trajectory = []  # 记录轨迹
        self.target_trajectory = []
        
        self.target_pos_history = deque(maxlen=5)
        self.prev_error_t = 0
        self.prev_error_f = 0
        self.current_waypoint = None  # 当前 navmesh waypoint
        
        self.reset()

    def set_sim_agents(self, target_human, robot, other_humans, sim=None):
        self.target_human = target_human
        self.robot = robot
        self.other_humans = other_humans
        self.sim = sim
        self.target_pos_history.clear()
        # 重置2D可视化状态（新episode）
        self.navmesh_img = None
        self.world_bounds = None
        self.robot_trajectory = []
        self.target_trajectory = []

    def reset(self, episode: NavigationEpisode = None, success: bool = False):
        if len(self.rgb_list) != 0 and episode is not None:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            # RGB 视频
            output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
            imageio.mimsave(output_video_path, self.rgb_list)
            self.rgb_list = []
            # 2D 可视化视频
            if len(self.vis_2d_list) > 0:
                vis_2d_path = os.path.join(save_dir, "{}_2d.mp4".format(episode.episode_id))
                imageio.mimsave(vis_2d_path, self.vis_2d_list)
                self.vis_2d_list = []
        
        self.target_pos_history.clear()
        self.prev_error_t = 0
        self.prev_error_f = 0
        self.robot_trajectory = []
        self.target_trajectory = []

    def _get_gt_distance(self):
        """获取 GT 距离"""
        if self.target_human is None or self.robot is None:
            return 2.0
        
        robot_pos = np.array([float(x) for x in self.robot.base_pos])
        target_pos = np.array([float(x) for x in self.target_human.base_pos])
        self.target_pos_history.append(target_pos.copy())
        
        return np.sqrt((robot_pos[0] - target_pos[0])**2 + (robot_pos[2] - target_pos[2])**2)

    def _is_target_approaching(self):
        """检测目标是否朝我们来"""
        if len(self.target_pos_history) < 3:
            return False
        
        robot_pos = np.array([float(x) for x in self.robot.base_pos])
        prev_pos = self.target_pos_history[-3]
        curr_pos = self.target_pos_history[-1]
        target_vel = curr_pos - prev_pos
        
        to_robot = robot_pos - curr_pos
        to_robot_2d = np.array([to_robot[0], to_robot[2]])
        norm = np.linalg.norm(to_robot_2d)
        if norm < 0.01:
            return False
        
        target_vel_2d = np.array([target_vel[0], target_vel[2]])
        return np.dot(target_vel_2d, to_robot_2d / norm) > 0.05

    def act(self, observations, detector, episode_id):
        self.episode_id = episode_id
        
        rgb = observations["agent_1_articulated_agent_jaw_rgb"]
        rgb_ = rgb[:, :, :3]
        height, width = rgb_.shape[:2]
        self.rgb_list.append(rgb_)
        
        gt_distance = self._get_gt_distance()
        
        # ============================================================
        # 1. PD 控制 (和 Baseline 完全一样)
        # ============================================================
        action = [0.0, 0.0, 0.0]
        target_visible = False
        
        # 1.1 尝试 panoptic
        # if self.target_id is not None and "agent_1_articulated_agent_jaw_panoptic" in observations:
        #     panoptic = observations["agent_1_articulated_agent_jaw_panoptic"]
        #     target_mask = (panoptic == self.target_id)
        #     if hasattr(target_mask, "ndim") and target_mask.ndim == 3:
        #         target_mask = target_mask[:, :, 0]
            
        #     if np.any(target_mask):
        #         rows = np.any(target_mask, axis=1)
        #         cols = np.any(target_mask, axis=0)
        #         r_idx = np.where(rows)[0]
        #         c_idx = np.where(cols)[0]
        #         cmin, cmax = int(c_idx[0]), int(c_idx[-1])
        #         rmin, rmax = int(r_idx[0]), int(r_idx[-1])
                
        #         center_x = (cmin + cmax) / (2 * width)
        #         bbox_area = (cmax - cmin) * (rmax - rmin)
                
        #         action = self._pd_control(center_x, bbox_area)
        #         target_visible = True
        
        # 1.2 回退到 detector
        # if not target_visible and detector['agent_1_main_humanoid_detector_sensor']['facing']:
        #     box = detector['agent_1_main_humanoid_detector_sensor']['box']
        #     center_x = (box[0] + box[2]) / (2 * width)
        #     bbox_area = (box[2] - box[0]) * (box[3] - box[1])
            
        #     action = self._pd_control(center_x, bbox_area)
        #     target_visible = True
        
        # 1.3 目标不可见 - 搜索
        if not target_visible:
            action = self._search_action()
        
        # ============================================================
        # 2. GT 距离修正 (Oracle 独有)
        # ============================================================
        move_speed, y_speed, yaw_speed = action
        
        # 2.1 太近 - 后退
        # if gt_distance < self.DANGER_DISTANCE:
        #     move_speed = -0.8
        # elif gt_distance < self.MIN_DISTANCE:
        #     back = (self.MIN_DISTANCE - gt_distance) / (self.MIN_DISTANCE - self.DANGER_DISTANCE)
        #     move_speed = min(move_speed, -0.4 * back)
        
        # 2.2 目标朝我们来 - 后退
        # if self._is_target_approaching() and gt_distance < 1.5:
        #     move_speed = min(move_speed, -0.3)
        
        # 渲染2D可视化帧（在获取waypoint之后）
        self._render_2d_frame()
        
        return [move_speed, y_speed, yaw_speed]

    def _pd_control(self, center_x, bbox_area):
        """
        PD 控制 - 和 Baseline 完全一样！
        
        error_t = 0.5 - center_x
        - 目标在左边 (center_x < 0.5) → error_t > 0 → yaw > 0 → 左转
        - 目标在右边 (center_x > 0.5) → error_t < 0 → yaw < 0 → 右转
        """
        error_t = 0.5 - center_x
        error_f = (self.TARGET_AREA - bbox_area) / 10000
        if abs(error_f) < 0.5:
            error_f = 0
        
        yaw_speed = self.KP_T * error_t
        move_speed = self.KP_F * error_f
        y_speed = self.KP_Y * error_t
        
        self.prev_error_t = error_t
        self.prev_error_f = error_f
        
        return [move_speed, y_speed, yaw_speed]

    def _get_navmesh_waypoint(self, target_pos):
        """
        使用 navmesh 获取到目标的下一个路径点
        
        好处：
        1. 绕过障碍物
        2. 避免卡墙
        3. 找到可行路径
        
        返回：下一个 waypoint，或 None
        """
        if self.sim is None or self.robot is None:
            return None
        
        robot_pos = np.array([float(x) for x in self.robot.base_pos])
        
        path = habitat_sim.ShortestPath()
        path.requested_start = robot_pos
        path.requested_end = np.array(target_pos)
        
        if self.sim.pathfinder.find_path(path) and len(path.points) > 1:
            return np.array(path.points[1])  # 返回下一个 waypoint
        return None

    def _search_action(self):
        """
        搜索 - 用 GT 位置 + Navmesh 判断往哪走
        
        优先级：
        1. 有 navmesh → 跟随 navmesh waypoint (绕障碍物)
        2. 无 navmesh → 直接往目标方向走
        """
        if self.target_human is None or self.robot is None:
            return [0.0, 0.0, 0.5]  # 默认旋转
        
        robot_pos = np.array([float(x) for x in self.robot.base_pos])
        target_pos = np.array([float(x) for x in self.target_human.base_pos])
        
        # 尝试用 navmesh 获取路径点
        waypoint = self._get_navmesh_waypoint(target_pos)
        self.current_waypoint = waypoint  # 保存用于可视化
        print("[debug]navmesh waypoint: ", waypoint)
        
        if waypoint is not None:
            # 用 navmesh waypoint (绕障碍物)
            to_waypoint = waypoint - robot_pos
            goal_2d = np.array([to_waypoint[0], to_waypoint[2]])
        else:
            # 直接往目标走
            to_target = target_pos - robot_pos
            goal_2d = np.array([to_target[0], to_target[2]])
        
        # 机器人朝向
        # try:
        #     import magnum as mn
        #     rot = self.robot.base_rot
        #     forward = rot.transform_vector(mn.Vector3(0, 0, -1))
        #     robot_forward = np.array([float(forward.x), float(forward.z)])
        # except:
        robot_forward = np.array([0.0, -1.0])
        
        # 归一化
        goal_dir = goal_2d / (np.linalg.norm(goal_2d) + 1e-6)
        robot_dir = robot_forward / (np.linalg.norm(robot_forward) + 1e-6)
        
        # 叉积判断转向
        cross = robot_dir[0] * goal_dir[1] - robot_dir[1] * goal_dir[0]
        
        # yaw > 0 = 左转，所以取负号
        yaw_speed = -cross * 2.0
        yaw_speed = np.clip(yaw_speed, -2.0, 2.0)
        
        # 边转边走
        move_speed = 0.3
        
        return [move_speed, 0.0, 0.0]

    # ================================================================
    # 2D 可视化
    # ================================================================
    
    def _init_navmesh_vis(self):
        """
        第一帧初始化：采样 navmesh 可导航区域，生成背景图
        """
        if self.sim is None:
            return
        
        pf = self.sim.pathfinder
        if not pf.is_loaded:
            return
        
        # 获取边界
        lower, upper = pf.get_bounds()
        x_min, z_min = lower[0], lower[2]
        x_max, z_max = upper[0], upper[2]
        
        # 加一点padding
        pad = 0.5
        self.world_bounds = (x_min - pad, x_max + pad, z_min - pad, z_max + pad)
        
        size = self.VIS_2D_SIZE
        samples = self.VIS_2D_NAVMESH_SAMPLES
        
        # 背景：深灰色
        img = np.full((size, size, 3), 40, dtype=np.uint8)
        
        # 采样 navmesh
        x_range = np.linspace(self.world_bounds[0], self.world_bounds[1], samples)
        z_range = np.linspace(self.world_bounds[2], self.world_bounds[3], samples)
        
        # 使用一个固定的 y 高度采样（地面高度通常在 0 附近）
        y_sample = lower[1] + 0.1
        
        for xi, x in enumerate(x_range):
            for zi, z in enumerate(z_range):
                if pf.is_navigable([x, y_sample, z]):
                    px, py = self._world_to_pixel(x, z)
                    # 画一个小方块表示可导航区域
                    cell_size = max(1, size // samples)
                    cv2.rectangle(img, 
                                  (px - cell_size//2, py - cell_size//2),
                                  (px + cell_size//2, py + cell_size//2),
                                  (80, 80, 80), -1)  # 浅灰色
        
        self.navmesh_img = img
    
    def _world_to_pixel(self, x, z):
        """世界坐标 (x, z) -> 像素坐标 (px, py)"""
        if self.world_bounds is None:
            return self.VIS_2D_SIZE // 2, self.VIS_2D_SIZE // 2
        
        x_min, x_max, z_min, z_max = self.world_bounds
        size = self.VIS_2D_SIZE
        
        # 归一化到 [0, 1]
        nx = (x - x_min) / (x_max - x_min + 1e-6)
        nz = (z - z_min) / (z_max - z_min + 1e-6)
        
        # 映射到像素 (翻转X和Z，使得从Y-往Y+看，即Y+朝屏幕外)
        px = int((1 - nx) * (size - 1))  # X轴翻转
        py = int((1 - nz) * (size - 1))
        
        return np.clip(px, 0, size-1), np.clip(py, 0, size-1)
    
    def _render_2d_frame(self):
        """渲染当前帧的2D平面图"""
        if self.robot is None or self.target_human is None:
            return
        
        # 第一帧初始化navmesh
        if self.navmesh_img is None:
            self._init_navmesh_vis()
        
        if self.navmesh_img is None:
            # navmesh加载失败，用纯黑背景
            img = np.zeros((self.VIS_2D_SIZE, self.VIS_2D_SIZE, 3), dtype=np.uint8)
        else:
            img = self.navmesh_img.copy()
        
        # 绘制边界框和标注
        if self.world_bounds is not None:
            x_min, x_max, z_min, z_max = self.world_bounds
            size = self.VIS_2D_SIZE
            border_color = (0, 255, 255)  # 黄色边框 (BGR)
            
            # 画四条边界线（略微内缩以便可见）
            margin = 2
            cv2.rectangle(img, (margin, margin), (size - margin - 1, size - margin - 1), 
                          border_color, 1)
            
            # 标注文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            text_color = (0, 255, 255)  # 黄色
            
            # X轴标注 (底部，X轴已翻转：左边是x_max，右边是x_min)
            cv2.putText(img, f"X:{x_max:.1f}", (5, size - 8), 
                        font, font_scale, text_color, 1)
            cv2.putText(img, f"X:{x_min:.1f}", (size - 60, size - 8), 
                        font, font_scale, text_color, 1)
            
            # Z轴标注 (左侧，注意Z轴在图像中是翻转的：上方是z_max，下方是z_min)
            cv2.putText(img, f"Z:{z_max:.1f}", (5, 15), 
                        font, font_scale, text_color, 1)
            cv2.putText(img, f"Z:{z_min:.1f}", (5, size - 20), 
                        font, font_scale, text_color, 1)
        
        # 获取位置
        robot_pos = self.robot.base_pos
        target_pos = self.target_human.base_pos
        
        robot_x, robot_z = float(robot_pos[0]), float(robot_pos[2])
        target_x, target_z = float(target_pos[0]), float(target_pos[2])
        
        # 记录轨迹
        self.robot_trajectory.append((robot_x, robot_z))
        self.target_trajectory.append((target_x, target_z))
        
        # 画轨迹（淡化的线）
        if len(self.robot_trajectory) > 1:
            for i in range(1, len(self.robot_trajectory)):
                p1 = self._world_to_pixel(*self.robot_trajectory[i-1])
                p2 = self._world_to_pixel(*self.robot_trajectory[i])
                # 渐变透明度
                alpha = 0.3 + 0.7 * (i / len(self.robot_trajectory))
                color = (int(100 * alpha), int(200 * alpha), int(255 * alpha))  # 蓝色
                cv2.line(img, p1, p2, color, 2)
        
        if len(self.target_trajectory) > 1:
            for i in range(1, len(self.target_trajectory)):
                p1 = self._world_to_pixel(*self.target_trajectory[i-1])
                p2 = self._world_to_pixel(*self.target_trajectory[i])
                alpha = 0.3 + 0.7 * (i / len(self.target_trajectory))
                color = (int(100 * alpha), int(100 * alpha), int(255 * alpha))  # 红色
                cv2.line(img, p1, p2, color, 2)
        
        # 画当前位置
        robot_px = self._world_to_pixel(robot_x, robot_z)
        target_px = self._world_to_pixel(target_x, target_z)
        
        # Robot: 蓝色圆
        cv2.circle(img, robot_px, 8, (255, 200, 100), -1)  # 填充
        cv2.circle(img, robot_px, 8, (255, 255, 255), 2)   # 白色边框
        
        # Target: 红色圆
        cv2.circle(img, target_px, 8, (100, 100, 255), -1)
        cv2.circle(img, target_px, 8, (255, 255, 255), 2)
        
        # Waypoint: 绿色菱形
        if self.current_waypoint is not None:
            wp_x, wp_z = float(self.current_waypoint[0]), float(self.current_waypoint[2])
            wp_px = self._world_to_pixel(wp_x, wp_z)
            # 画菱形 (用四个点)
            d = 6  # 菱形半径
            pts = np.array([
                [wp_px[0], wp_px[1] - d],  # 上
                [wp_px[0] + d, wp_px[1]],  # 右
                [wp_px[0], wp_px[1] + d],  # 下
                [wp_px[0] - d, wp_px[1]],  # 左
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], (0, 255, 0))  # 绿色填充
            cv2.polylines(img, [pts], True, (255, 255, 255), 1)  # 白色边框
            # 画从 robot 到 waypoint 的虚线
            cv2.line(img, robot_px, wp_px, (0, 200, 0), 1, cv2.LINE_AA)
        
        # 添加图例
        cv2.putText(img, "Robot", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        cv2.putText(img, "Target", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
        cv2.putText(img, "Waypoint", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Step: {len(self.robot_trajectory)}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 转换为RGB（cv2默认BGR）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.vis_2d_list.append(img_rgb)
