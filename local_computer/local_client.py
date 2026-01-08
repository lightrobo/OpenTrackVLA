#!/usr/bin/env python3
"""
本地测试客户端 - 用于测试gRPC连接
可以发送单张图片或视频文件进行测试
在本地电脑上运行
"""

import grpc
import time
import argparse
import numpy as np
from pathlib import Path
import cv2

# 导入生成的 gRPC 代码
import inference_pb2
import inference_pb2_grpc


def test_health(stub):
    """测试健康检查"""
    print("[Test] 健康检查...")
    try:
        response = stub.HealthCheck(inference_pb2.Empty())
        print(f"  - healthy: {response.healthy}")
        print(f"  - model_status: {response.model_status}")
        print(f"  - device: {response.device}")
        return response.healthy
    except grpc.RpcError as e:
        print(f"  - 错误: {e}")
        return False


def test_single_image(stub, image_path: str, instruction: str):
    """测试单张图片推理"""
    print(f"[Test] 单帧推理: {image_path}")
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"  - 无法读取图片: {image_path}")
        return
    
    # BGR -> RGB, 编码为JPEG
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    
    # 发送请求
    request = inference_pb2.InferRequest(
        image_data=buffer.tobytes(),
        instruction=instruction,
        frame_id=1,
        timestamp_ms=int(time.time() * 1000)
    )
    
    start = time.time()
    try:
        response = stub.Infer(request)
        rtt = (time.time() - start) * 1000
        
        if response.success:
            waypoints = np.array(response.waypoints).reshape(response.n_waypoints, 3)
            print(f"  - 成功! RTT: {rtt:.0f}ms, 服务器耗时: {response.inference_time_ms:.0f}ms")
            print(f"  - Waypoints ({response.n_waypoints}):")
            for i, (x, y, theta) in enumerate(waypoints):
                print(f"      [{i}] x={x:.3f}, y={y:.3f}, theta={theta:.3f}")
        else:
            print(f"  - 失败: {response.error}")
    except grpc.RpcError as e:
        print(f"  - gRPC错误: {e}")


def test_video(stub, video_path: str, instruction: str, max_frames: int = 100):
    """测试视频推理"""
    print(f"[Test] 视频推理: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  - 无法打开视频: {video_path}")
        return
    
    frame_count = 0
    total_rtt = 0
    total_server_time = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 编码
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        request = inference_pb2.InferRequest(
            image_data=buffer.tobytes(),
            instruction=instruction,
            frame_id=frame_count,
            timestamp_ms=int(time.time() * 1000)
        )
        
        start = time.time()
        try:
            response = stub.Infer(request)
            rtt = (time.time() - start) * 1000
            
            if response.success:
                total_rtt += rtt
                total_server_time += response.inference_time_ms
                frame_count += 1
                
                if frame_count % 10 == 0:
                    print(f"  - 已处理 {frame_count} 帧, 平均RTT: {total_rtt/frame_count:.0f}ms")
        except grpc.RpcError as e:
            print(f"  - 帧 {frame_count} gRPC错误: {e}")
            break
    
    cap.release()
    
    if frame_count > 0:
        print(f"[Test] 完成! 共 {frame_count} 帧")
        print(f"  - 平均RTT: {total_rtt/frame_count:.0f}ms")
        print(f"  - 平均服务器耗时: {total_server_time/frame_count:.0f}ms")
        print(f"  - 有效帧率: {1000 / (total_rtt/frame_count):.1f} fps")


def test_latency(stub, iterations: int = 10):
    """延迟基准测试"""
    print(f"[Test] 延迟测试 ({iterations}次)...")
    
    # 创建一个小的测试图像
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', dummy_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    
    rtts = []
    for i in range(iterations):
        request = inference_pb2.InferRequest(
            image_data=buffer.tobytes(),
            instruction="test",
            frame_id=i,
            timestamp_ms=int(time.time() * 1000)
        )
        
        start = time.time()
        try:
            response = stub.Infer(request)
            rtt = (time.time() - start) * 1000
            rtts.append(rtt)
        except grpc.RpcError:
            pass
    
    if rtts:
        print(f"  - 最小RTT: {min(rtts):.0f}ms")
        print(f"  - 最大RTT: {max(rtts):.0f}ms")
        print(f"  - 平均RTT: {sum(rtts)/len(rtts):.0f}ms")
        print(f"  - 中位数RTT: {sorted(rtts)[len(rtts)//2]:.0f}ms")


def main():
    parser = argparse.ArgumentParser(description='gRPC推理测试客户端')
    parser.add_argument('--server', type=str, default='localhost:50051', help='gRPC服务器地址')
    parser.add_argument('--image', type=str, help='测试图片路径')
    parser.add_argument('--video', type=str, help='测试视频路径')
    parser.add_argument('--instruction', type=str, default='Follow the person', help='文本指令')
    parser.add_argument('--latency-test', action='store_true', help='运行延迟测试')
    parser.add_argument('--max-frames', type=int, default=100, help='视频最大帧数')
    
    args = parser.parse_args()
    
    print(f"[Test] 连接服务器: {args.server}")
    
    channel = grpc.insecure_channel(
        args.server,
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    
    # 健康检查
    if not test_health(stub):
        print("[Test] 服务器不健康，退出")
        return
    
    # 延迟测试
    if args.latency_test:
        test_latency(stub)
    
    # 图片测试
    if args.image:
        test_single_image(stub, args.image, args.instruction)
    
    # 视频测试
    if args.video:
        test_video(stub, args.video, args.instruction, args.max_frames)
    
    # 如果没有指定测试，只做健康检查
    if not args.image and not args.video and not args.latency_test:
        print("[Test] 提示: 使用 --image, --video, 或 --latency-test 进行测试")
    
    channel.close()


if __name__ == '__main__':
    main()

