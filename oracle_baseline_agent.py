"""
Oracle Baseline Agent - Uses ground truth information for expert trajectory collection.

This agent has access to:
- Ground truth 3D positions of all agents
- Ground truth velocities
- Perfect knowledge of the scene

Purpose: Generate high-quality expert demonstrations for imitation learning.
The trained model will only see images, but learns to mimic this oracle behavior.
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

            # Get all agents
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent  # Target human
            robot_agent = sim.agents_mgr[1].articulated_agent          # Our robot
            
            # Get other humans (distractors) if any
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

            # Initialize oracle agent with sim reference
            robot_config.set_sim_agents(sim, humanoid_agent_main, robot_agent, other_humans)

            while not env.episode_over:
                record_info = {}
                obs = sim.get_sensor_observations()
                detector = env.task._get_observations(env.current_episode)
                
                # Oracle agent uses ground truth positions
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
                # Also record GT info for debugging
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

            # Save all episodes for diverse training data
            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(record_infos, f, indent=2)
            with open(os.path.join(save_dir, "{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result, f, indent=2)

            robot_config.reset(env.current_episode, success=result['success'])


class OracleBaselineAgent(AgentConfig):
    """
    Oracle agent that uses ground truth positions for expert trajectory collection.
    
    Key features:
    1. Uses GT 3D positions - no perception errors
    2. Collision avoidance - backs off when too close
    3. Velocity matching - predicts target motion using GT velocity
    4. Multi-human handling - tracks main target, avoids others
    5. Smooth, optimal control - generates clean expert demonstrations
    """
    
    # Target following parameters
    IDEAL_DISTANCE = 1.8        # meters - ideal following distance
    MIN_DISTANCE = 1.0          # meters - start backing off
    DANGER_DISTANCE = 0.7       # meters - urgent back off
    MAX_DISTANCE = 3.5          # meters - speed up to catch up
    
    # Control gains
    MAX_FORWARD_SPEED = 1.5     # m/s
    MAX_BACKWARD_SPEED = 0.8    # m/s
    MAX_LATERAL_SPEED = 0.5     # m/s
    MAX_YAW_SPEED = 1.5         # rad/s
    
    def __init__(self, result_path, target_id=None):
        super().__init__()
        print("Initialize Oracle baseline agent (uses GT positions)")

        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        self.target_id = target_id
        
        self.rgb_list = []
        
        # Sim references (set by evaluate_agent)
        self.sim = None
        self.target_human = None
        self.robot = None
        self.other_humans = []
        
        # Velocity estimation from position history
        self.target_pos_history = deque(maxlen=5)
        self.robot_pos_history = deque(maxlen=5)
        
        # Action smoothing
        self.last_action = np.array([0.0, 0.0, 0.0])
        self.action_smoothing = 0.5
        
        # Small noise for data diversity
        self.noise_scale = 0.02
        
        self.reset()

    def set_sim_agents(self, sim, target_human, robot, other_humans):
        """Set references to sim agents for GT access."""
        self.sim = sim
        self.target_human = target_human
        self.robot = robot
        self.other_humans = other_humans

    def reset(self, episode: NavigationEpisode = None, success: bool = False):
        if len(self.rgb_list) != 0 and episode is not None:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(self.result_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
            imageio.mimsave(output_video_path, self.rgb_list)
            self.rgb_list = []
        
        self.target_pos_history.clear()
        self.robot_pos_history.clear()
        self.last_action = np.array([0.0, 0.0, 0.0])

    def _get_gt_positions(self):
        """Get ground truth 3D positions."""
        robot_pos = np.array([float(x) for x in self.robot.base_pos])
        target_pos = np.array([float(x) for x in self.target_human.base_pos])
        
        # Store for velocity estimation
        self.robot_pos_history.append(robot_pos.copy())
        self.target_pos_history.append(target_pos.copy())
        
        return robot_pos, target_pos

    def _estimate_target_velocity(self):
        """Estimate target velocity from position history."""
        if len(self.target_pos_history) < 2:
            return np.array([0.0, 0.0, 0.0])
        
        # Simple finite difference (could use more sophisticated filtering)
        dt = 0.1  # assume 10 Hz
        vel = (self.target_pos_history[-1] - self.target_pos_history[-2]) / dt
        return vel

    def _get_robot_heading(self):
        """Get robot's current heading direction."""
        # Robot forward direction in world frame
        rot = self.robot.base_rot
        # Convert quaternion to forward vector (assuming z-forward convention)
        # This depends on the specific coordinate system used
        forward = np.array([0, 0, -1])  # Default forward
        try:
            import magnum as mn
            forward = rot.transform_vector(mn.Vector3(0, 0, -1))
            forward = np.array([forward.x, forward.y, forward.z])
        except:
            pass
        return forward

    def _compute_avoidance_for_others(self, robot_pos):
        """Compute avoidance vector for other humans (distractors)."""
        avoidance = np.array([0.0, 0.0, 0.0])
        
        for human in self.other_humans:
            try:
                human_pos = np.array([float(x) for x in human.base_pos])
                diff = robot_pos - human_pos
                dist = np.linalg.norm(diff)
                
                if dist < 1.5:  # Avoidance radius
                    # Push away from distractor
                    if dist > 0.1:
                        avoidance += (diff / dist) * (1.5 - dist) * 0.5
            except:
                pass
        
        return avoidance

    def act(self, observations, detector, episode_id):
        self.episode_id = episode_id
        
        # Get RGB for saving
        rgb = observations["agent_1_articulated_agent_jaw_rgb"]
        rgb_ = rgb[:, :, :3]
        self.rgb_list.append(rgb_)
        
        # === ORACLE: Use ground truth positions ===
        robot_pos, target_pos = self._get_gt_positions()
        
        # Vector from robot to target (in world frame, XZ plane)
        to_target = target_pos - robot_pos
        to_target_2d = np.array([to_target[0], to_target[2]])  # XZ plane
        distance = np.linalg.norm(to_target_2d)
        
        # Estimate target velocity for prediction
        target_vel = self._estimate_target_velocity()
        target_vel_2d = np.array([target_vel[0], target_vel[2]])
        
        # Predict where target will be in ~0.5 seconds
        prediction_horizon = 0.5
        predicted_target = target_pos + target_vel * prediction_horizon
        to_predicted = predicted_target - robot_pos
        to_predicted_2d = np.array([to_predicted[0], to_predicted[2]])
        
        # Get robot heading
        robot_heading = self._get_robot_heading()
        robot_heading_2d = np.array([robot_heading[0], robot_heading[2]])
        robot_heading_2d = robot_heading_2d / (np.linalg.norm(robot_heading_2d) + 1e-6)
        
        # Compute angle to target
        if distance > 0.01:
            to_target_dir = to_target_2d / distance
        else:
            to_target_dir = robot_heading_2d
        
        # Cross product for turn direction (positive = turn left)
        cross = robot_heading_2d[0] * to_target_dir[1] - robot_heading_2d[1] * to_target_dir[0]
        dot = np.dot(robot_heading_2d, to_target_dir)
        angle_error = np.arctan2(cross, dot)  # Signed angle
        
        # === CONTROL LOGIC ===
        
        # 1. YAW: Turn toward (predicted) target
        yaw_speed = np.clip(angle_error * 2.0, -self.MAX_YAW_SPEED, self.MAX_YAW_SPEED)
        
        # 2. FORWARD/BACKWARD: Distance control with collision avoidance
        if distance < self.DANGER_DISTANCE:
            # Too close! Back off urgently
            forward_speed = -self.MAX_BACKWARD_SPEED
        elif distance < self.MIN_DISTANCE:
            # Getting close, slow back off
            back_factor = (self.MIN_DISTANCE - distance) / (self.MIN_DISTANCE - self.DANGER_DISTANCE)
            forward_speed = -self.MAX_BACKWARD_SPEED * back_factor * 0.5
        elif distance > self.MAX_DISTANCE:
            # Too far, speed up
            forward_speed = self.MAX_FORWARD_SPEED
        elif distance > self.IDEAL_DISTANCE:
            # A bit far, move forward
            approach_factor = (distance - self.IDEAL_DISTANCE) / (self.MAX_DISTANCE - self.IDEAL_DISTANCE)
            forward_speed = self.MAX_FORWARD_SPEED * approach_factor * 0.8
        else:
            # In ideal range, match target speed
            target_speed_toward_robot = -np.dot(target_vel_2d, to_target_dir)
            if target_speed_toward_robot > 0.3:
                # Target approaching us - back off a bit
                forward_speed = -0.3
            else:
                # Match roughly
                forward_speed = np.linalg.norm(target_vel_2d) * 0.5
        
        # 3. LATERAL: Small lateral adjustments for smooth tracking
        # Move laterally if target is to the side and we're not facing them
        if abs(angle_error) > 0.3:
            lateral_speed = np.sign(angle_error) * self.MAX_LATERAL_SPEED * 0.3
        else:
            lateral_speed = 0.0
        
        # 4. Avoid other humans (distractors)
        avoidance = self._compute_avoidance_for_others(robot_pos)
        if np.linalg.norm(avoidance) > 0.1:
            # Add avoidance to forward/lateral
            avoidance_forward = np.dot(avoidance[[0, 2]], robot_heading_2d)
            avoidance_lateral = avoidance[0] * (-robot_heading_2d[1]) + avoidance[2] * robot_heading_2d[0]
            forward_speed += avoidance_forward * 0.5
            lateral_speed += avoidance_lateral * 0.5
        
        # 5. Special case: if target is behind us, prioritize turning
        if dot < 0:  # Target behind
            forward_speed = 0.0
            yaw_speed = np.sign(angle_error) * self.MAX_YAW_SPEED
        
        # === SMOOTHING AND OUTPUT ===
        action = np.array([forward_speed, lateral_speed, yaw_speed])
        
        # Smooth action
        action = self.action_smoothing * self.last_action + (1 - self.action_smoothing) * action
        self.last_action = action.copy()
        
        # Add small noise for diversity
        noise = np.random.normal(0, self.noise_scale, size=3)
        action = action + noise
        
        # Clip to bounds
        action[0] = np.clip(action[0], -self.MAX_BACKWARD_SPEED, self.MAX_FORWARD_SPEED)
        action[1] = np.clip(action[1], -self.MAX_LATERAL_SPEED, self.MAX_LATERAL_SPEED)
        action[2] = np.clip(action[2], -self.MAX_YAW_SPEED, self.MAX_YAW_SPEED)
        
        return action.tolist()

