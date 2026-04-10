#!/usr/bin/env python3
"""
AGX Orin 摄像头采集端 - gRPC Client
采集USB摄像头画面，通过gRPC发送到云端推理
支持 HTTP 视频流用于远程查看

服务器返回的是 waypoints（累积位移），不是速度！
- x: 前进方向位移 (m)，相对起点
- y: 左右方向位移 (m)，相对起点
- theta: 朝向角 (rad)，相对起点
"""

import cv2
import grpc
import time
import argparse
import numpy as np
import os
import sys
from typing import Optional, Tuple
from threading import Thread, Lock
import math

# 导入生成的 gRPC 代码
import inference_pb2
import inference_pb2_grpc

# TrajectoryController (延迟导入，仅 --publish-vel 模式需要)
_TrajectoryController = None

def _import_trajectory_controller():
    global _TrajectoryController
    if _TrajectoryController is not None:
        return _TrajectoryController
    script_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'lightrobo_sdk_example_wt1_realsense',
        'src', 'lightrobo_sdk_example', 'script'
    ))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from track_vla_client import TrajectoryController
    _TrajectoryController = TrajectoryController
    return _TrajectoryController

# ROS2 (延迟导入，仅 --ros-topic / --publish-vel 模式需要)
_rclpy = None
_Node = None
_Image = None


class CameraStreamer:
    """摄像头采集 + gRPC 客户端 + HTTP 视频流"""
    
    def __init__(
        self,
        server_addr: str = "localhost:50051",
        camera_idx: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 10,
        jpeg_quality: int = 80,
        http_port: int = 8080
    ):
        self.server_addr = server_addr
        self.camera_idx = camera_idx
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.http_port = http_port
        
        self.channel = None
        self.stub = None
        self.cap = None
        self.frame_id = 0
        
        # HTTP 流相关
        self._current_frame = None
        self._frame_lock = Lock()
        self._http_server = None
        
        # ROS2 共享状态
        self._ros_node = None
        self._ros_initialized = False
        
        # 速度发布相关
        self._vel_publisher = None
        self._vel_controller = None
        self._vel_timer_running = False
        self._wp_last_update = 0.0
        self._wp_stopped = False
        
    def connect(self) -> bool:
        """连接gRPC服务器"""
        print(f"[Client] 连接服务器: {self.server_addr}")
        
        self.channel = grpc.insecure_channel(
            self.server_addr,
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ]
        )
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
        
        # 健康检查
        try:
            response = self.stub.HealthCheck(inference_pb2.Empty())
            print(f"[Client] 服务器状态: healthy={response.healthy}, device={response.device}")
            return response.healthy
        except grpc.RpcError as e:
            print(f"[Client] 连接失败: {e}")
            return False
    
    def open_camera(self, ros_topic: str = None) -> bool:
        """打开摄像头（V4L2 或 ROS2 topic）"""
        if ros_topic:
            return self._open_ros_topic(ros_topic)
        
        print(f"[Client] 打开摄像头: {self.camera_idx}")
        
        self.cap = cv2.VideoCapture(self.camera_idx)
        if not self.cap.isOpened():
            print(f"[Client] 无法打开摄像头 {self.camera_idx}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[Client] 摄像头参数: {actual_w}x{actual_h} @ {actual_fps}fps")
        return True
    
    def _ensure_ros_initialized(self) -> bool:
        """确保 rclpy 已初始化且 ROS2 node 已创建（幂等）"""
        global _rclpy, _Node, _Image
        if self._ros_initialized:
            return True
        try:
            import rclpy
            from rclpy.node import Node as RosNode
            from sensor_msgs.msg import Image as RosImage
            _rclpy = rclpy
            _Node = RosNode
            _Image = RosImage
        except ImportError:
            print("[Client] 错误: rclpy 未安装")
            return False

        if not _rclpy.ok():
            _rclpy.init()

        self._ros_node = _Node('camera_streamer_client')
        self._ros_spin_thread = Thread(
            target=lambda: _rclpy.spin(self._ros_node), daemon=True
        )
        self._ros_spin_thread.start()
        self._ros_initialized = True
        return True

    def _init_vel_publisher(self) -> bool:
        """初始化 TrajectoryController + /velocity_commands publisher + 50Hz 定时器"""
        if not self._ensure_ros_initialized():
            return False

        TC = _import_trajectory_controller()
        self._vel_controller = TC(
            max_vx=0.6, max_vy=0.3, max_wz=0.5,
            max_acc_vx=2.0, max_acc_vy=1.0, max_acc_wz=3.0,
            kp_forward=3.0, kp_lateral=0.5,
            kp_steer=0.5, kp_heading=0.3,
            kd_steer=0.05,
            lookahead_steps=5, vy_deadzone=0.08, wp_dt=0.1,
        )

        from std_msgs.msg import Float32MultiArray
        self._Float32MultiArray = Float32MultiArray
        self._vel_publisher = self._ros_node.create_publisher(
            Float32MultiArray, '/velocity_commands', 10
        )

        self._vel_timer_running = True
        self._vel_timer_thread = Thread(target=self._vel_publish_loop, daemon=True)
        self._vel_timer_thread.start()

        print("[Client] 速度发布已启用: /velocity_commands @ 50Hz")
        return True

    def _update_waypoints(self, waypoints):
        """线程安全地更新 waypoints 并记录时间"""
        self._vel_controller.update_waypoints(waypoints)
        self._wp_last_update = time.monotonic()
        self._wp_stopped = False

    def _vel_publish_loop(self):
        """50Hz 循环: compute_velocity → publish，含轨迹超时保护"""
        interval = 1.0 / 50.0
        TRAJECTORY_TIMEOUT = 1.5
        while self._vel_timer_running:
            now = time.monotonic()
            if (self._wp_last_update > 0
                    and not self._wp_stopped
                    and (now - self._wp_last_update) > TRAJECTORY_TIMEOUT):
                self._vel_controller.stop()
                self._wp_stopped = True
                print("[Vel] 轨迹超时，停车")

            vx, vy, wz = self._vel_controller.compute_velocity()
            msg = self._Float32MultiArray()
            msg.data = [vx, vy, wz]
            self._vel_publisher.publish(msg)
            time.sleep(interval)

    def _open_ros_topic(self, topic: str) -> bool:
        """通过 ROS2 topic 获取图像"""
        if not self._ensure_ros_initialized():
            return False

        print(f"[Client] 订阅 ROS2 topic: {topic}")
        self._ros_frame = None
        self._ros_frame_lock = Lock()
        self._ros_topic = topic
        
        def _img_callback(msg):
            h, w = msg.height, msg.width
            encoding = msg.encoding
            if encoding in ('rgb8', 'RGB8'):
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif encoding in ('bgr8', 'BGR8'):
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            elif encoding in ('bgra8', 'BGRA8'):
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif encoding in ('rgba8', 'RGBA8'):
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)
            
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            with self._ros_frame_lock:
                self._ros_frame = frame
        
        self._ros_sub = self._ros_node.create_subscription(
            _Image, topic, _img_callback, 10
        )

        # 等待第一帧
        for _ in range(50):
            time.sleep(0.1)
            with self._ros_frame_lock:
                if self._ros_frame is not None:
                    h, w = self._ros_frame.shape[:2]
                    print(f"[Client] ROS2 topic 就绪: {w}x{h}")
                    return True
        
        print(f"[Client] 等待 ROS2 topic 超时: {topic}")
        return False
    
    def _read_ros_frame(self):
        """从 ROS2 topic 读取最新帧"""
        with self._ros_frame_lock:
            if self._ros_frame is not None:
                return True, self._ros_frame.copy()
            return False, None
    
    def encode_frame(self, frame: np.ndarray) -> bytes:
        """JPEG编码 (输入 BGR，cv2.imencode 内部处理 BGR→YCbCr)"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buffer.tobytes()
    
    def infer(self, frame: np.ndarray, instruction: str) -> Tuple[Optional[np.ndarray], float]:
        """发送帧到云端推理，返回 waypoints（累积位移序列）"""
        self.frame_id += 1
        
        # 编码
        image_data = self.encode_frame(frame)
        
        # 发送请求
        request = inference_pb2.InferRequest(
            image_data=image_data,
            instruction=instruction,
            frame_id=self.frame_id,
            timestamp_ms=int(time.time() * 1000)
        )
        
        try:
            response = self.stub.Infer(request)
            
            if response.success:
                # 服务器返回的是累积位移: [x, y, theta] * n_waypoints
                waypoints = np.array(response.waypoints).reshape(response.n_waypoints, 3)
                # 模型原生坐标: x=forward, y=right, yaw=CW
                # 控制器坐标:   x=forward, y=left,  yaw=CCW
                waypoints[:, 1] *= -1
                waypoints[:, 2] *= -1
                return waypoints, response.inference_time_ms
            else:
                print(f"[Client] 推理错误: {response.error}")
                return None, 0
        except grpc.RpcError as e:
            print(f"[Client] gRPC错误: {e}")
            return None, 0
    
    def draw_trajectory(self, frame: np.ndarray, waypoints: np.ndarray, scale: float = 120.0) -> np.ndarray:
        """
        在画面上绘制轨迹
        
        参数:
            frame: 原始图像
            waypoints: 累积位移序列 (N, 3) - [x, y, theta]
                       x: 前进方向位移 (m)
                       y: 左侧方向位移 (m)
                       theta: 朝向角 (rad)
            scale: 像素/米 的缩放比例 (默认120，与项目其他地方一致)
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        # 与项目其他地方一致的投影参数
        cx = w // 2              # 图像宽度中心
        cy = int(h * 0.86)       # 图像高度的 86% 处作为机器人位置
        
        arrow_len = 20  # 方向箭头长度
        
        # 绘制轨迹点
        points = [(cx, cy)]  # 起点
        for i, (x, y, theta) in enumerate(waypoints):
            # 转换到屏幕坐标
            # 世界坐标: x=前（向上），y=左（向左）
            # 屏幕坐标: px向右增加，py向下增加
            px = int(cx - y * scale)  # y左 → px左
            py = int(cy - x * scale)  # x前 → py上
            points.append((px, py))
            
            # 画点
            # 颜色渐变：绿色（近）→ 红色（远）
            progress = i / max(len(waypoints) - 1, 1)
            color = (
                int(255 * progress),      # R: 远处变红
                int(255 * (1 - progress)), # G: 近处绿
                0
            )
            cv2.circle(vis, (px, py), 5, color, -1)
            cv2.circle(vis, (px, py), 7, (255, 255, 255), 1)
            
            # 画方向箭头（显示 theta）
            arrow_dx = int(-arrow_len * math.sin(theta))
            arrow_dy = int(-arrow_len * math.cos(theta))
            arrow_end = (px + arrow_dx, py + arrow_dy)
            
            # 箭头颜色：根据转向角度
            turn_intensity = min(abs(theta) / 0.5, 1.0)  # 0.5 rad ≈ 30° 作为最大
            arrow_color = (
                int(255 * turn_intensity),
                int(255 * (1 - turn_intensity * 0.5)),
                0
            )
            cv2.arrowedLine(vis, (px, py), arrow_end, arrow_color, 2, tipLength=0.4)
        
        # 连线（轨迹路径）
        for i in range(len(points) - 1):
            # 渐变线条颜色
            progress = i / max(len(points) - 2, 1)
            line_color = (
                int(100 * progress),
                int(200 * (1 - progress * 0.5)),
                0
            )
            cv2.line(vis, points[i], points[i + 1], line_color, 2)
        
        # 画机器人位置（底部中心 = 起点）
        cv2.circle(vis, (cx, cy), 12, (255, 0, 0), -1)
        cv2.circle(vis, (cx, cy), 14, (255, 255, 255), 2)
        # 机器人朝向箭头（向上 = 初始朝向）
        cv2.arrowedLine(vis, (cx, cy), (cx, cy - 35), (255, 100, 100), 3, tipLength=0.3)
        cv2.putText(vis, "Robot", (cx - 25, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示第一个 waypoint 信息
        if len(waypoints) > 0:
            x, y, theta = waypoints[0]
            wp_info = f"WP0: x={x:.2f}m y={y:.2f}m th={math.degrees(theta):.1f}deg"
            cv2.putText(vis, wp_info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return vis
    
    def _update_http_frame(self, frame: np.ndarray):
        """更新 HTTP 流的当前帧"""
        with self._frame_lock:
            self._current_frame = frame.copy()
    
    def _generate_frames(self):
        """生成 MJPEG 帧流"""
        while True:
            with self._frame_lock:
                if self._current_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self._current_frame.copy()
            
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30fps
    
    def _start_http_server(self):
        """启动 HTTP 视频流服务器"""
        try:
            from flask import Flask, Response
        except ImportError:
            print("[Client] 警告: Flask 未安装，HTTP 流功能不可用")
            print("[Client] 安装: pip install flask")
            return
        
        app = Flask(__name__)
        streamer = self
        
        @app.route('/')
        def index():
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>OpenTrackVLA Live Stream</title>
                <style>
                    body { 
                        background: #1a1a2e; 
                        color: #eee; 
                        font-family: monospace;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        padding: 20px;
                    }
                    h1 { color: #00ff88; }
                    img { 
                        border: 2px solid #00ff88; 
                        border-radius: 8px;
                        max-width: 100%;
                    }
                    .info { 
                        margin-top: 10px; 
                        color: #888; 
                    }
                    .legend {
                        margin-top: 15px;
                        padding: 10px;
                        background: #2a2a4e;
                        border-radius: 5px;
                        font-size: 12px;
                    }
                </style>
            </head>
            <body>
                <h1>🤖 OpenTrackVLA Live Stream</h1>
                <img src="/video_feed" alt="Video Stream">
                <p class="info">实时推理可视化 | 累积位移轨迹</p>
                <div class="legend">
                    <b>图例:</b><br>
                    🔴 机器人位置（起点）| 
                    🟢→🔴 waypoints（近→远）| 
                    ➤ 朝向箭头
                </div>
            </body>
            </html>
            '''
        
        @app.route('/video_feed')
        def video_feed():
            return Response(
                streamer._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        print(f"[Client] HTTP 视频流启动: http://0.0.0.0:{self.http_port}")
        app.run(host='0.0.0.0', port=self.http_port, threaded=True)
    
    def run(self, instruction: str = "Follow the person", display: bool = True,
            http_stream: bool = False, ros_topic: str = None, publish_vel: bool = False):
        """主循环：帧采集+可视化在主线程，gRPC推理在后台线程"""
        if not self.connect():
            return

        if not self.open_camera(ros_topic=ros_topic):
            return

        self._use_ros = ros_topic is not None

        if publish_vel:
            if not self._init_vel_publisher():
                print("[Client] 速度发布初始化失败，退出")
                return

        if http_stream:
            http_thread = Thread(target=self._start_http_server, daemon=True)
            http_thread.start()
            time.sleep(1)

        # 后台推理共享状态
        self._latest_waypoints = None
        self._latest_rtt = 0.0
        self._latest_server_time = 0.0
        self._infer_lock = Lock()
        self._infer_frame = None
        self._infer_frame_lock = Lock()
        self._infer_running = True

        infer_thread = Thread(target=self._inference_loop, args=(instruction,), daemon=True)
        infer_thread.start()

        print(f"[Client] 开始推理循环，指令: '{instruction}'")
        if display:
            print("[Client] 按 'q' 退出, 'r' 重置历史")
        else:
            print("[Client] 按 Ctrl+C 退出")

        display_interval = 1.0 / 30.0
        last_display_time = 0.0

        try:
            while True:
                if self._use_ros:
                    ret, frame = self._read_ros_frame()
                else:
                    ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue

                with self._infer_frame_lock:
                    self._infer_frame = frame.copy()

                now = time.time()
                if now - last_display_time < display_interval:
                    time.sleep(0.002)
                    continue
                last_display_time = now

                with self._infer_lock:
                    waypoints = self._latest_waypoints
                    rtt = self._latest_rtt
                    server_time = self._latest_server_time

                if waypoints is not None:
                    vis = self.draw_trajectory(frame, waypoints)
                    if self._vel_controller:
                        vx, vy, wz = self._vel_controller.compute_velocity()
                    else:
                        vx = vy = wz = 0.0
                    vel_str = f"  vel=({vx:+.2f},{vy:+.2f},{wz:+.2f})" if publish_vel else ""
                    info = f"RTT:{rtt:.0f}ms Srv:{server_time:.0f}ms F:{self.frame_id}{vel_str}"
                    cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                else:
                    vis = frame

                if http_stream:
                    self._update_http_frame(vis)

                if display:
                    cv2.imshow('OpenTrackVLA Streamer', vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        print("[Client] 发送重置请求...")

        except KeyboardInterrupt:
            print("\n[Client] 收到中断信号")
        finally:
            self._infer_running = False
            self.close(display)

    def _inference_loop(self, instruction: str):
        """后台线程：持续取最新帧 → gRPC 推理 → 更新 waypoints"""
        while self._infer_running:
            with self._infer_frame_lock:
                frame = self._infer_frame

            if frame is None:
                time.sleep(0.01)
                continue

            start = time.time()
            result = self.infer(frame, instruction)
            rtt = (time.time() - start) * 1000

            if result[0] is not None:
                waypoints, server_time = result

                if self._vel_controller is not None:
                    self._update_waypoints(waypoints.tolist())

                with self._infer_lock:
                    self._latest_waypoints = waypoints
                    self._latest_rtt = rtt
                    self._latest_server_time = server_time

                n = len(waypoints)
                w0 = waypoints[0] if n > 0 else [0, 0, 0]
                wL = waypoints[-1] if n > 0 else [0, 0, 0]
                dist = math.sqrt(wL[0]**2 + wL[1]**2)
                if self._vel_controller:
                    vx, vy, wz = self._vel_controller.compute_velocity()
                    print(f"[Infer] F{self.frame_id}: RTT={rtt:.0f}ms wp={n} "
                          f"p0=({w0[0]:+.4f},{w0[1]:+.4f},{math.degrees(w0[2]):+.1f}°) "
                          f"pN=({wL[0]:+.4f},{wL[1]:+.4f},{math.degrees(wL[2]):+.1f}°) "
                          f"D={dist:.4f}m "
                          f"vel=({vx:+.3f},{vy:+.3f},{wz:+.3f})")
                else:
                    print(f"[Infer] F{self.frame_id}: RTT={rtt:.0f}ms wp={n} "
                          f"p0=({w0[0]:+.4f},{w0[1]:+.4f},{math.degrees(w0[2]):+.1f}°) "
                          f"pN=({wL[0]:+.4f},{wL[1]:+.4f},{math.degrees(wL[2]):+.1f}°) "
                          f"D={dist:.4f}m")
    
    def close(self, display: bool = True):
        """清理资源"""
        self._vel_timer_running = False
        if self._vel_controller:
            self._vel_controller.stop()
        if self.cap:
            self.cap.release()
        if self._ros_node:
            self._ros_node.destroy_node()
        if _rclpy and _rclpy.ok():
            try:
                _rclpy.shutdown()
            except Exception:
                pass
        if self.channel:
            self.channel.close()
        if display:
            cv2.destroyAllWindows()
        print("[Client] 已关闭")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AGX Orin 摄像头采集端')
    parser.add_argument('--server', type=str, default='localhost:50051', help='gRPC服务器地址')
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引')
    parser.add_argument('--width', type=int, default=640, help='图像宽度')
    parser.add_argument('--height', type=int, default=480, help='图像高度')
    parser.add_argument('--fps', type=int, default=10, help='目标帧率')
    parser.add_argument('--quality', type=int, default=80, help='JPEG质量 (1-100)')
    parser.add_argument('--instruction', type=str, default='Follow the person', help='文本指令')
    parser.add_argument('--no-display', action='store_true', help='不显示画面（headless模式）')
    parser.add_argument('--http-stream', action='store_true', help='启用HTTP视频流（用于远程查看）')
    parser.add_argument('--http-port', type=int, default=8080, help='HTTP流端口')
    parser.add_argument('--ros-topic', type=str, default=None,
                        help='通过 ROS2 topic 获取图像（替代 V4L2），例如 /camera/realsense2_camera/color/image_raw')
    parser.add_argument('--publish-vel', action='store_true', default=True,
                        help='将推理 waypoints 转化为速度指令，发布到 /velocity_commands（默认开启）')
    parser.add_argument('--no-publish-vel', action='store_false', dest='publish_vel',
                        help='禁用速度指令发布')
    
    args = parser.parse_args()
    
    streamer = CameraStreamer(
        server_addr=args.server,
        camera_idx=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.quality,
        http_port=args.http_port
    )
    
    streamer.run(
        instruction=args.instruction, 
        display=not args.no_display,
        http_stream=args.http_stream,
        ros_topic=args.ros_topic,
        publish_vel=args.publish_vel
    )
