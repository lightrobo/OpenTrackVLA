#!/usr/bin/env python3
"""
OpenTrackVLA摄像头实时推理脚本
从USB摄像头读取视频流，运行OpenTrackVLA模型进行waypoint预测
"""

import cv2
import sys
import os
import time
import argparse
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from PIL import Image
import torch

# 添加项目根目录到 Python 路径
# 获取脚本所在目录的父目录（项目根目录）
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# 验证 PyTorch 安装
try:
    import torch
    print(f"[init] PyTorch 版本: {torch.__version__}")
    print(f"[init] CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[init] CUDA 版本: {torch.version.cuda}")
        print(f"[init] GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"[init] GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("[init] ⚠️  CUDA 不可用，将使用 CPU（推理会很慢）")
        # 检查是否是 ARM64 平台
        import platform
        if platform.machine() in ['aarch64', 'arm64']:
            print("[init] ℹ️  检测到 ARM64 平台，CPU 模式是正常的")
except ImportError as e:
    print(f"❌ PyTorch 导入失败: {e}")
    print("请安装 PyTorch: pip install torch torchvision torchaudio")
    sys.exit(1)

# 导入项目模块（必须在 torch 导入之后）
try:
    from open_trackvla_hf import OpenTrackVLAForWaypoint
    from cache_gridpool import VisionFeatureCacher, VisionCacheConfig, grid_pool_tokens
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"项目根目录: {project_root}")
    print("请确保在OpenTrackVLA项目根目录下运行此脚本，或设置 PYTHONPATH")
    import traceback
    traceback.print_exc()
    sys.exit(1)


class CameraInference:
    def __init__(
        self,
        model_dir: str,
        dinov3_model_path: Optional[str] = None,
        siglip_model_path: Optional[str] = None,
        history: int = 31,
        device: str = "cuda",
        camera_idx: int = 0
    ):
        """
        初始化摄像头推理器
        
        Args:
            model_dir: OpenTrackVLA模型目录（HuggingFace格式）
            dinov3_model_path: DINOv3模型路径（可选，会从环境变量读取）
            siglip_model_path: SigLIP模型路径（可选）
            history: 历史帧数量（默认31）
            device: 计算设备（cuda/cpu）
            camera_idx: 摄像头索引
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.history = history
        self.camera_idx = camera_idx
        
        print(f"[init] 使用设备: {self.device}")
        print(f"[init] 加载模型: {model_dir}")
        
        # 检查模型目录是否存在
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        
        # 加载OpenTrackVLA模型
        print("[init] 开始加载模型权重（可能需要几分钟）...")
        import time
        start_time = time.time()
        
        try:
            self.model = OpenTrackVLAForWaypoint.from_pretrained(
                model_dir,
                torch_dtype=torch.float32 if self.device.type == "cpu" else torch.bfloat16
            )
            load_time = time.time() - start_time
            print(f"[init] 模型权重加载完成 (耗时: {load_time:.2f}秒)")
            
            print(f"[init] 移动模型到设备: {self.device}...")
            self.model = self.model.to(self.device).eval()
            print("[init] OpenTrackVLA模型加载完成")
        except Exception as e:
            print(f"[init] ❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 设置DINOv3模型路径
        if dinov3_model_path:
            os.environ['DINOV3_MODEL_PATH'] = dinov3_model_path
            print(f"[init] 设置DINOv3路径: {dinov3_model_path}")
        
        # 初始化vision encoder
        vision_cfg = VisionCacheConfig(
            image_size=384,
            batch_size=1,
            device=str(self.device),
            siglip_model_name=siglip_model_path if siglip_model_path else None
        )
        self.vision_encoder = VisionFeatureCacher(vision_cfg)
        self.vision_encoder.eval()
        print("[init] Vision encoder初始化完成")
        
        # 帧缓冲区
        self.frame_buffer: List[np.ndarray] = []
        self.coarse_history: List[torch.Tensor] = []
        
    def encode_frame(self, rgb_frame: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码单帧图像为vision tokens
        
        Args:
            rgb_frame: RGB图像 (H, W, 3), uint8
            
        Returns:
            Vcoarse: (4, C) coarse tokens
            Vfine: (64, C) fine tokens
        """
        # 转换为PIL Image
        pil_img = Image.fromarray(rgb_frame.astype(np.uint8))
        
        # 编码DINOv3和SigLIP
        tok_dino, Hp, Wp = self.vision_encoder._encode_dino([pil_img])
        tok_sigl = self.vision_encoder._encode_siglip([pil_img], out_hw=(Hp, Wp))
        
        # 拼接并pooling
        Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)  # (1, P, C_total)
        Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)[0].float()  # (64, C)
        Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4)[0].float()  # (4, C)
        
        return Vcoarse, Vfine
    
    def predict_waypoints(
        self,
        instruction: str = "Follow the person"
    ) -> Optional[np.ndarray]:
        """
        基于当前帧缓冲区预测waypoints
        
        Args:
            instruction: 文本指令
            
        Returns:
            waypoints: (n_waypoints, 3) numpy array [x, y, theta], 如果失败返回None
        """
        if len(self.frame_buffer) == 0:
            return None
        
        try:
            # 获取当前帧
            current_frame = self.frame_buffer[-1]
            Vcoarse_curr, Vfine_curr = self.encode_frame(current_frame)
            
            # 更新coarse历史
            self.coarse_history.append(Vcoarse_curr.cpu())
            
            # 构建历史序列（左padding）
            H = self.history
            hist = list(self.coarse_history)
            if len(hist) < H:
                pad_needed = H - len(hist)
                first = hist[0] if hist else Vcoarse_curr.cpu()
                hist = [first] * pad_needed + hist
            else:
                hist = hist[-H:]
            
            # 构建tokens和indices
            coarse_list = []
            coarse_tidx = []
            for t, tok4 in enumerate(hist):
                tok4 = tok4.to(self.device)
                coarse_list.append(tok4)
                coarse_tidx.append(
                    torch.full(
                        (tok4.size(0),),
                        fill_value=t,
                        dtype=torch.long,
                        device=self.device
                    )
                )
            
            coarse_tokens = torch.cat(coarse_list, dim=0).unsqueeze(0)  # (1, H*4, C)
            coarse_tidx = torch.cat(coarse_tidx, dim=0).unsqueeze(0)    # (1, H*4)
            
            # Fine tokens for current frame (time index H)
            fine_tokens = Vfine_curr.to(self.device).unsqueeze(0)  # (1, 64, C)
            fine_tidx = torch.full(
                (1, fine_tokens.size(1)),
                fill_value=H,
                dtype=torch.long,
                device=self.device
            )
            
            # 推理
            with torch.inference_mode():
                waypoints = self.model(
                    coarse_tokens, coarse_tidx,
                    fine_tokens, fine_tidx,
                    [instruction]
                )  # (1, n_waypoints, 3)
            
            return waypoints[0].detach().cpu().float().numpy()
            
        except Exception as e:
            print(f"[predict] 错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def draw_waypoints(
        self,
        frame: np.ndarray,
        waypoints: np.ndarray,
        scale: float = 120.0
    ) -> np.ndarray:
        """
        在帧上绘制waypoints轨迹
        
        Args:
            frame: RGB图像
            waypoints: (n_waypoints, 3) [x, y, theta]
            scale: 像素缩放因子
            
        Returns:
            绘制后的图像
        """
        if waypoints is None or len(waypoints) == 0:
            return frame
        
        h, w = frame.shape[:2]
        base_x = w // 2
        base_y = int(h * 0.86)
        
        # 转换为像素坐标
        pts = []
        for wp in waypoints:
            x, y = float(wp[0]), float(wp[1])
            px = base_x - int(y * scale)
            py = base_y - int(x * scale)
            pts.append((px, py))
        
        # 绘制轨迹
        frame_out = frame.copy()
        for i in range(1, len(pts)):
            cv2.line(
                frame_out,
                pts[i-1],
                pts[i],
                color=(0, 255, 200),
                thickness=6
            )
        
        # 绘制起点
        if pts:
            cv2.circle(frame_out, pts[0], 6, (0, 255, 0), -1)
        
        return frame_out
    
    def run(self, instruction: str = "Follow the person"):
        """
        运行摄像头推理循环
        """
        # 打开摄像头
        cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {self.camera_idx}")
            return
        
        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"\n摄像头已打开")
        print(f"分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"指令: {instruction}")
        print("\n按 'q' 退出")
        print("按 's' 保存当前帧")
        print("按 'r' 重置历史缓冲区\n")
        
        frame_count = 0
        inference_interval = 5  # 每5帧推理一次（降低计算负载）
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("错误: 无法读取帧")
                    break
                
                # BGR转RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 添加到缓冲区
                self.frame_buffer.append(frame_rgb)
                if len(self.frame_buffer) > self.history + 10:  # 保持一定长度
                    self.frame_buffer.pop(0)
                
                frame_count += 1
                
                # 定期推理
                waypoints = None
                if frame_count >= self.history and frame_count % inference_interval == 0:
                    waypoints = self.predict_waypoints(instruction)
                
                # 绘制waypoints
                frame_display = self.draw_waypoints(frame, waypoints)
                
                # 显示信息
                cv2.putText(
                    frame_display,
                    f"Frame: {frame_count} | Buffer: {len(self.frame_buffer)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame_display,
                    instruction[:50],
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # 显示
                cv2.imshow('OpenTrackVLA Inference', frame_display)
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"frame_{frame_count}.jpg"
                    cv2.imwrite(filename, frame_display)
                    print(f"已保存: {filename}")
                elif key == ord('r'):
                    self.frame_buffer.clear()
                    self.coarse_history.clear()
                    print("历史缓冲区已重置")
                
        except KeyboardInterrupt:
            print("\n收到中断信号")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("摄像头已关闭")


def main():
    parser = argparse.ArgumentParser(description="OpenTrackVLA摄像头实时推理")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="~/resources/models/vla/opentrackvla-qwen06b",
        help="OpenTrackVLA模型目录（HuggingFace格式）"
    )
    parser.add_argument(
        "--dinov3_path",
        type=str,
        default="~/resources/models/vision_tower/dinov3-vits16",
        help="DINOv3模型路径"
    )
    parser.add_argument(
        "--siglip_path",
        type=str,
        default=None,
        help="SigLIP模型路径（可选，默认从HuggingFace Hub下载）"
    )
    parser.add_argument(
        "--history",
        type=int,
        default=31,
        help="历史帧数量"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="摄像头索引"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Follow the person",
        help="文本指令"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="计算设备"
    )
    
    args = parser.parse_args()
    
    # 展开路径
    model_dir = os.path.expanduser(args.model_dir)
    dinov3_path = os.path.expanduser(args.dinov3_path) if args.dinov3_path else None
    siglip_path = os.path.expanduser(args.siglip_path) if args.siglip_path else None
    
    # 检查路径
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录不存在: {model_dir}")
        sys.exit(1)
    
    if dinov3_path and not os.path.exists(dinov3_path):
        print(f"警告: DINOv3路径不存在: {dinov3_path}")
        print("将使用HuggingFace默认路径")
        dinov3_path = None
    
    if siglip_path and not os.path.exists(siglip_path):
        print(f"警告: SigLIP路径不存在: {siglip_path}")
        print("将使用HuggingFace默认路径")
        siglip_path = None
    
    # 创建推理器并运行
    inferencer = CameraInference(
        model_dir=model_dir,
        dinov3_model_path=dinov3_path,
        siglip_model_path=siglip_path,
        history=args.history,
        device=args.device,
        camera_idx=args.camera
    )
    
    inferencer.run(instruction=args.instruction)


if __name__ == "__main__":
    main()