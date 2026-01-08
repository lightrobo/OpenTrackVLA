#!/usr/bin/env python3
"""
AGX Orin 摄像头采集端 - gRPC Client
采集USB摄像头画面，通过gRPC发送到云端推理
支持 HTTP 视频流用于远程查看
"""

import cv2
import grpc
import time
import argparse
import numpy as np
from typing import Optional, Tuple
from threading import Thread, Lock
import io

# 导入生成的 gRPC 代码
import inference_pb2
import inference_pb2_grpc


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
    
    def open_camera(self) -> bool:
        """打开摄像头"""
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
    
    def encode_frame(self, frame: np.ndarray) -> bytes:
        """JPEG编码"""
        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buffer.tobytes()
    
    def infer(self, frame: np.ndarray, instruction: str) -> Optional[np.ndarray]:
        """发送帧到云端推理"""
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
                waypoints = np.array(response.waypoints).reshape(response.n_waypoints, 3)
                return waypoints, response.inference_time_ms
            else:
                print(f"[Client] 推理错误: {response.error}")
                return None, 0
        except grpc.RpcError as e:
            print(f"[Client] gRPC错误: {e}")
            return None, 0
    
    def draw_waypoints(self, frame: np.ndarray, waypoints: np.ndarray, scale: float = 120.0) -> np.ndarray:
        """在画面上绘制waypoints"""
        vis = frame.copy()
        h, w = vis.shape[:2]
        cx, cy = w // 2, h - 50  # 底部中心作为原点
        
        points = []
        for i, (x, y, theta) in enumerate(waypoints):
            # 转换坐标：x前进变为向上，y左右保持
            px = int(cx - y * scale)  # 左右
            py = int(cy - x * scale)  # 前后
            points.append((px, py))
            
            # 画点
            color = (0, 255, 0) if i == 0 else (0, 255 - i * 20, i * 20)
            cv2.circle(vis, (px, py), 6, color, -1)
            cv2.circle(vis, (px, py), 8, (255, 255, 255), 1)
        
        # 连线
        for i in range(len(points) - 1):
            cv2.line(vis, points[i], points[i + 1], (0, 200, 0), 2)
        
        # 画机器人位置（底部中心）
        cv2.circle(vis, (cx, cy), 10, (255, 0, 0), -1)
        cv2.putText(vis, "Robot", (cx - 25, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
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
                </style>
            </head>
            <body>
                <h1>🤖 OpenTrackVLA Live Stream</h1>
                <img src="/video_feed" alt="Video Stream">
                <p class="info">实时推理可视化 | Waypoints 叠加显示</p>
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
    
    def run(self, instruction: str = "Follow the person", display: bool = True, http_stream: bool = False):
        """主循环"""
        if not self.connect():
            return
        
        if not self.open_camera():
            return
        
        # 启动 HTTP 流服务器
        if http_stream:
            http_thread = Thread(target=self._start_http_server, daemon=True)
            http_thread.start()
            time.sleep(1)  # 等待服务器启动
        
        print(f"[Client] 开始推理循环，指令: '{instruction}'")
        if display:
            print("[Client] 按 'q' 退出, 'r' 重置历史")
        else:
            print("[Client] 按 Ctrl+C 退出")
        
        frame_interval = 1.0 / self.fps
        last_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[Client] 读取帧失败")
                    continue
                
                # 控制帧率
                current_time = time.time()
                if current_time - last_time < frame_interval:
                    continue
                last_time = current_time
                
                # 推理
                start = time.time()
                result = self.infer(frame, instruction)
                rtt = (time.time() - start) * 1000
                
                if result[0] is not None:
                    waypoints, server_time = result
                    vis = self.draw_waypoints(frame, waypoints)
                    
                    # 显示信息
                    info = f"RTT: {rtt:.0f}ms | Server: {server_time:.0f}ms | Frame: {self.frame_id}"
                    cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 终端输出
                    if not display:
                        print(f"[Client] Frame {self.frame_id}: RTT={rtt:.0f}ms, Server={server_time:.0f}ms, Waypoints={len(waypoints)}")
                else:
                    vis = frame
                    cv2.putText(vis, "Inference Failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 更新 HTTP 流
                if http_stream:
                    self._update_http_frame(vis)
                
                # 本地显示
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
            self.close(display)
    
    def close(self, display: bool = True):
        """清理资源"""
        if self.cap:
            self.cap.release()
        if self.channel:
            self.channel.close()
        # 只有在显示模式下才调用 destroyAllWindows
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
        http_stream=args.http_stream
    )
