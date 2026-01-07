"""
Improved Baseline Agent for collecting better training data.

Key improvements over baseline_agent.py:
1. Target motion prediction using exponential moving average
2. Search strategy when target is lost (turn toward last seen direction)
3. Smooth action output to avoid jerky movements
4. Better distance control with adaptive gains
5. Diversity injection for richer training data
"""

import habitat
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
    robot_config = ImprovedBaselineAgent(save_path, target_id)
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

            iter_step = 0
            followed_step = 0
            too_far_count = 0
            status = 'Normal'
            info = env.get_metrics()

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

                if np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos) > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        status = 'Lost'
                        break

                record_info["step"] = iter_step
                record_info["dis_to_human"] = float(np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos))
                record_info["facing"] = info['human_following']
                record_info["base_velocity"] = action
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

            # Save all episodes (not just successful ones) for diverse training data
            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(record_infos, f, indent=2)
            with open(os.path.join(save_dir, "{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result, f, indent=2)

            robot_config.reset(env.current_episode, success=result['success'])


class ImprovedBaselineAgent(AgentConfig):
    """
    Improved baseline agent with:
    - Target motion prediction (EMA of position changes)
    - Search strategy when target lost
    - Smooth action output
    - Adaptive distance control
    """
    
    def __init__(self, result_path, target_id=None):
        super().__init__()
        print("Initialize improved baseline agent")

        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        self.target_id = target_id
        
        self.rgb_list = []

        # PD gains (same as baseline, but we'll adapt dynamically)
        self.kp_t = 2.0   # yaw proportional
        self.kd_t = 0.3   # yaw derivative (added!)
        self.kp_f = 1.0   # forward proportional
        self.kd_f = 0.2   # forward derivative (added!)
        self.kp_y = 0.5   # lateral proportional
        self.kd_y = 0.1   # lateral derivative (added!)

        # Target tracking history
        self.position_history = deque(maxlen=10)  # store (center_x, center_y, bbox_area)
        self.velocity_ema = np.array([0.0, 0.0])  # EMA of target velocity in image space
        self.ema_alpha = 0.3  # smoothing factor
        
        # Lost target state
        self.frames_since_seen = 0
        self.last_seen_direction = 0.0  # -1 = left, +1 = right
        self.last_seen_distance = 1.0   # normalized distance estimate
        
        # Action smoothing
        self.last_action = np.array([0.0, 0.0, 0.0])
        self.action_smoothing = 0.7  # blend factor with previous action
        
        # Error tracking
        self.prev_error_t = 0
        self.prev_error_f = 0
        
        # Diversity injection (small random perturbations for training data variety)
        self.noise_scale = 0.05
        self.add_noise = True

        self.reset()

    def reset(self, episode: NavigationEpisode = None, success: bool = False):
        if len(self.rgb_list) != 0 and episode is not None:
            # Save video for all episodes (more training data)
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
            imageio.mimsave(output_video_path, self.rgb_list)
            self.rgb_list = []
        
        # Reset tracking state
        self.position_history.clear()
        self.velocity_ema = np.array([0.0, 0.0])
        self.frames_since_seen = 0
        self.last_seen_direction = 0.0
        self.last_seen_distance = 1.0
        self.last_action = np.array([0.0, 0.0, 0.0])
        self.prev_error_t = 0
        self.prev_error_f = 0

    def _update_motion_prediction(self, center_x, center_y, bbox_area):
        """Update target motion prediction using EMA."""
        current_pos = np.array([center_x, center_y])
        
        if len(self.position_history) > 0:
            prev_pos = np.array([self.position_history[-1][0], self.position_history[-1][1]])
            velocity = current_pos - prev_pos
            # Exponential moving average of velocity
            self.velocity_ema = self.ema_alpha * velocity + (1 - self.ema_alpha) * self.velocity_ema
        
        self.position_history.append((center_x, center_y, bbox_area))
        
        # Update last seen info
        self.last_seen_direction = np.sign(center_x - 0.5)  # which side of image
        self.last_seen_distance = np.sqrt(bbox_area) / 200  # rough distance proxy
        self.frames_since_seen = 0

    def _predict_target_position(self):
        """Predict where target will be based on motion history."""
        if len(self.position_history) < 2:
            return None
        
        last_pos = np.array([self.position_history[-1][0], self.position_history[-1][1]])
        # Predict next position using velocity EMA
        predicted = last_pos + self.velocity_ema
        return predicted

    def _search_action(self):
        """Generate action when target is lost."""
        self.frames_since_seen += 1
        
        # If recently lost, continue in last direction briefly
        if self.frames_since_seen < 5:
            # Momentum: continue last action but decaying
            decay = 0.8 ** self.frames_since_seen
            return self.last_action * decay
        
        # If lost longer, turn toward last seen direction
        if self.frames_since_seen < 30:
            # Turn toward where we last saw the target
            yaw_speed = self.last_seen_direction * 0.5
            # Move forward slowly while searching
            forward_speed = 0.2
            return np.array([forward_speed, 0.0, yaw_speed])
        
        # If lost very long, do a full search rotation
        yaw_speed = 0.8  # spin to search
        return np.array([0.0, 0.0, yaw_speed])

    def _smooth_action(self, action):
        """Apply exponential smoothing to action output."""
        action = np.array(action)
        smoothed = self.action_smoothing * self.last_action + (1 - self.action_smoothing) * action
        self.last_action = smoothed
        return smoothed

    def _add_training_noise(self, action):
        """Add small random noise for training data diversity."""
        if not self.add_noise:
            return action
        noise = np.random.normal(0, self.noise_scale, size=3)
        return action + noise

    def act(self, observations, detector, episode_id):
        self.episode_id = episode_id
        
        rgb = observations["agent_1_articulated_agent_jaw_rgb"]
        rgb_ = rgb[:, :, :3]
        height, width = rgb_.shape[:2]
        
        action = np.array([0.0, 0.0, 0.0])
        target_visible = False

        # Try panoptic segmentation first (if target_id specified)
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
                center_y = (rmin + rmax) / (2 * height)
                bbox_area = (cmax - cmin) * (rmax - rmin)
                
                action = self._compute_tracking_action(center_x, bbox_area, width, height)
                self._update_motion_prediction(center_x, center_y, bbox_area)
                target_visible = True

        # Fall back to detector
        if not target_visible and detector['agent_1_main_humanoid_detector_sensor']['facing']:
            box = detector['agent_1_main_humanoid_detector_sensor']['box']
            center_x = (box[0] + box[2]) / (2 * width)
            center_y = (box[1] + box[3]) / (2 * height)
            bbox_area = (box[2] - box[0]) * (box[3] - box[1])
            
            action = self._compute_tracking_action(center_x, bbox_area, width, height)
            self._update_motion_prediction(center_x, center_y, bbox_area)
            target_visible = True

        # Target not visible - use search strategy
        if not target_visible:
            action = self._search_action()
        
        # Apply smoothing and noise
        action = self._smooth_action(action)
        action = self._add_training_noise(action)
        
        # Clip to reasonable bounds
        action = np.clip(action, -2.0, 2.0)
        
        self.rgb_list.append(rgb_)
        return action.tolist()

    def _compute_tracking_action(self, center_x, bbox_area, width, height):
        """Compute tracking action with predictive compensation."""
        
        # Basic error
        error_t = 0.5 - center_x
        
        # Adaptive target size based on typical human bbox
        target_area = 25000  # target bbox area (closer than baseline's 30000)
        error_f = (target_area - bbox_area) / 10000
        
        # Dead zone for distance to avoid oscillation
        if abs(error_f) < 0.3:
            error_f = 0
        
        # Predictive compensation: adjust for predicted target motion
        predicted = self._predict_target_position()
        if predicted is not None:
            # If target moving right, pre-rotate right
            predicted_error_t = 0.5 - predicted[0]
            # Blend current and predicted error
            error_t = 0.7 * error_t + 0.3 * predicted_error_t
        
        # PD control
        derivative_t = error_t - self.prev_error_t
        derivative_f = error_f - self.prev_error_f
        
        yaw_speed = self.kp_t * error_t + self.kd_t * derivative_t
        move_speed = self.kp_f * error_f + self.kd_f * derivative_f
        y_speed = self.kp_y * error_t + self.kd_y * derivative_t
        
        self.prev_error_t = error_t
        self.prev_error_f = error_f
        
        return np.array([move_speed, y_speed, yaw_speed])

