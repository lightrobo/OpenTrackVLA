#!/usr/bin/env python3
"""
云端推理服务器 - gRPC Server
在云端开发机上运行，接收图像进行推理
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

# 添加项目根目录（云端需要有 OpenTrackVLA 项目代码）
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# 导入生成的 gRPC 代码
import inference_pb2
import inference_pb2_grpc

import torch

# 延迟导入模型（等torch加载完成）
OpenTrackVLAForWaypoint = None
VisionFeatureCacher = None
VisionCacheConfig = None
grid_pool_tokens = None


class InferenceModel:
    """推理模型封装"""
    
    def __init__(self, model_dir: str, device: str = "cuda", history: int = 31):
        global OpenTrackVLAForWaypoint, VisionFeatureCacher, VisionCacheConfig, grid_pool_tokens
        
        from open_trackvla_hf import OpenTrackVLAForWaypoint
        from cache_gridpool import VisionFeatureCacher, VisionCacheConfig, grid_pool_tokens
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.history = history
        
        print(f"[Server] 使用设备: {self.device}")
        print(f"[Server] 加载模型: {model_dir}")
        
        # 加载模型
        self.model = OpenTrackVLAForWaypoint.from_pretrained(
            model_dir,
            torch_dtype=torch.float32 if self.device.type == "cpu" else torch.bfloat16
        ).to(self.device).eval()
        print("[Server] OpenTrackVLA模型加载完成")
        
        # 初始化vision encoder
        vision_cfg = VisionCacheConfig(
            image_size=384,
            batch_size=1,
            device=str(self.device)
        )
        self.vision_encoder = VisionFeatureCacher(vision_cfg)
        self.vision_encoder.eval()
        print("[Server] Vision encoder初始化完成")
        
        # 历史缓存
        self.coarse_history = []
    
    def encode_frame(self, rgb_frame: np.ndarray):
        """编码单帧"""
        pil_img = Image.fromarray(rgb_frame.astype(np.uint8))
        
        tok_dino, Hp, Wp = self.vision_encoder._encode_dino([pil_img])
        tok_sigl = self.vision_encoder._encode_siglip([pil_img], out_hw=(Hp, Wp))
        
        Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)
        Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)[0].float()
        Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4)[0].float()
        
        return Vcoarse, Vfine
    
    def predict(self, rgb_frame: np.ndarray, instruction: str) -> np.ndarray:
        """预测waypoints"""
        Vcoarse_curr, Vfine_curr = self.encode_frame(rgb_frame)
        
        # 更新历史
        self.coarse_history.append(Vcoarse_curr.cpu())
        
        # 构建历史序列
        H = self.history
        hist = list(self.coarse_history)
        if len(hist) < H:
            pad_needed = H - len(hist)
            first = hist[0] if hist else Vcoarse_curr.cpu()
            hist = [first] * pad_needed + hist
        else:
            hist = hist[-H:]
        
        # 构建tokens
        coarse_list = []
        coarse_tidx = []
        for t, tok4 in enumerate(hist):
            tok4 = tok4.to(self.device)
            coarse_list.append(tok4)
            coarse_tidx.append(
                torch.full((tok4.size(0),), fill_value=t, dtype=torch.long, device=self.device)
            )
        
        coarse_tokens = torch.cat(coarse_list, dim=0).unsqueeze(0)
        coarse_tidx = torch.cat(coarse_tidx, dim=0).unsqueeze(0)
        
        fine_tokens = Vfine_curr.to(self.device).unsqueeze(0)
        fine_tidx = torch.full((1, fine_tokens.size(1)), fill_value=H, dtype=torch.long, device=self.device)
        
        # 推理
        with torch.inference_mode():
            waypoints = self.model(
                coarse_tokens, coarse_tidx,
                fine_tokens, fine_tidx,
                [instruction]
            )
        
        return waypoints[0].detach().cpu().float().numpy()
    
    def reset_history(self):
        """重置历史"""
        self.coarse_history = []


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC 服务实现"""
    
    def __init__(self, model: InferenceModel):
        self.model = model
    
    def Infer(self, request, context):
        """单帧推理"""
        start_time = time.time()
        
        try:
            # 解码JPEG
            image = Image.open(io.BytesIO(request.image_data))
            rgb_frame = np.array(image.convert('RGB'))
            
            # 推理
            waypoints = self.model.predict(rgb_frame, request.instruction)
            
            inference_time = (time.time() - start_time) * 1000
            
            return inference_pb2.InferResponse(
                frame_id=request.frame_id,
                waypoints=waypoints.flatten().tolist(),
                n_waypoints=waypoints.shape[0],
                inference_time_ms=inference_time,
                success=True
            )
        except Exception as e:
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
        return inference_pb2.HealthResponse(
            healthy=True,
            model_status="loaded",
            device=str(self.model.device)
        )


def serve(args):
    """启动服务"""
    print(f"[Server] 初始化模型...")
    model = InferenceModel(
        model_dir=args.model_dir,
        device=args.device,
        history=args.history
    )
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(model), server
    )
    
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
    parser.add_argument('--model-dir', type=str, required=True, help='模型目录路径')
    parser.add_argument('--port', type=int, default=50051, help='gRPC端口')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--history', type=int, default=31, help='历史帧数量')
    parser.add_argument('--workers', type=int, default=4, help='工作线程数')
    
    args = parser.parse_args()
    serve(args)

