#!/usr/bin/env python3
"""
云端推理服务器 - gRPC Server
直接复用 trained_agent.py 的推理逻辑
"""

import grpc
from concurrent import futures
import time
import sys
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import io

# 添加项目根目录
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# 导入生成的 gRPC 代码
import inference_pb2
import inference_pb2_grpc

# 直接导入 trained_agent 的推理类
from trained_agent import GTBBoxAgent


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC 服务实现 - 直接复用 GTBBoxAgent"""
    
    def __init__(self, model_dir: str, device: str = "cuda"):
        # 设置环境变量，让 GTBBoxAgent 加载正确的模型
        if model_dir:
            os.environ['HF_MODEL_DIR'] = model_dir
        
        # 创建一个临时目录用于 GTBBoxAgent（它需要 result_path）
        tmp_dir = "/tmp/grpc_inference"
        os.makedirs(tmp_dir, exist_ok=True)
        
        # 实例化 GTBBoxAgent，复用其完整的推理逻辑
        self.agent = GTBBoxAgent(result_path=tmp_dir)
        self.device = device
        
        print(f"[Server] GTBBoxAgent 初始化完成")
        print(f"[Server] 模型目录: {model_dir}")
        print(f"[Server] 设备: {self.agent.planner_device}")
    
    def Infer(self, request, context):
        """单帧推理 - 直接调用 GTBBoxAgent._planner_action"""
        start_time = time.time()
        
        try:
            # 解码JPEG
            image = Image.open(io.BytesIO(request.image_data))
            rgb_frame = np.array(image.convert('RGB'))
            
            # 直接调用 GTBBoxAgent 的推理方法
            instruction = request.instruction or "follow the person"
            action = self.agent._planner_action(rgb_frame, instruction)
            
            inference_time = (time.time() - start_time) * 1000
            
            if action is None:
                return inference_pb2.InferResponse(
                    frame_id=request.frame_id,
                    error="Planner returned None",
                    success=False
                )
            
            # 获取预测的轨迹（如果有）
            traj = self.agent._last_predicted_traj
            if traj is not None:
                waypoints = traj.flatten().tolist()
                n_waypoints = traj.shape[0]
            else:
                # 如果没有轨迹，返回 action 作为单个 waypoint
                waypoints = action
                n_waypoints = 1
            
            return inference_pb2.InferResponse(
                frame_id=request.frame_id,
                waypoints=waypoints,
                n_waypoints=n_waypoints,
                inference_time_ms=inference_time,
                success=True
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return inference_pb2.InferResponse(
                frame_id=request.frame_id,
                error=str(e),
                success=False
            )
    
    def StreamInfer(self, request_iterator, context):
        """流式推理"""
        for request in request_iterator:
            yield self.Infer(request, context)
    
    def HealthCheck(self, request, context):
        """健康检查"""
        model_loaded = self.agent.planner_model is not None
        return inference_pb2.HealthResponse(
            healthy=model_loaded,
            model_status="loaded" if model_loaded else "not_loaded",
            device=str(self.agent.planner_device)
        )
    
    def reset(self):
        """重置历史状态"""
        self.agent._coarse_hist_tokens.clear()
        self.agent._last_predicted_traj = None


def serve(args):
    """启动服务"""
    print(f"[Server] 初始化推理服务...")
    
    servicer = InferenceServicer(
        model_dir=args.model_dir,
        device=args.device
    )
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    
    server.add_insecure_port(f'[::]:{args.port}')
    server.start()
    
    print(f"[Server] gRPC服务启动: 0.0.0.0:{args.port}")
    print(f"[Server] 等待连接...")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[Server] 收到中断信号，正在关闭...")
        server.stop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='云端推理服务器')
    parser.add_argument('--model-dir', type=str, default=None, help='HuggingFace模型目录路径（可选，也可通过 HF_MODEL_DIR 环境变量设置）')
    parser.add_argument('--port', type=int, default=50051, help='gRPC端口')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--workers', type=int, default=4, help='工作线程数')
    
    args = parser.parse_args()
    serve(args)
