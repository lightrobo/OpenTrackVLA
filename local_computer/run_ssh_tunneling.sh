#!/bin/bash
# SSH 隧道脚本
# 在本地电脑上运行，建立到云端开发机的端口转发

# ============ 配置区域 ============
REMOTE_USER="root"          # 云端用户名
REMOTE_HOST="172.18.16.32"  # 云端服务器地址
REMOTE_PORT=50051                    # 云端gRPC端口
LOCAL_PORT=50051                     # 本地暴露端口
SSH_PORT=2222                          # SSH端口

# 可选：SSH密钥路径
# SSH_KEY="~/.ssh/id_rsa"
# ================================

echo "=========================================="
echo "  OpenTrackVLA SSH Tunnel"
echo "=========================================="
echo ""
echo "配置:"
echo "  云端服务器: ${REMOTE_USER}@${REMOTE_HOST}"
echo "  远程端口:   ${REMOTE_PORT}"
echo "  本地端口:   ${LOCAL_PORT}"
echo ""

# 检查是否已有隧道在运行
if lsof -i :${LOCAL_PORT} > /dev/null 2>&1; then
    echo "⚠️  端口 ${LOCAL_PORT} 已被占用"
    echo "   请先关闭占用该端口的进程"
    exit 1
fi

# 构建SSH命令
# 绑定到 0.0.0.0 使得局域网内其他设备（如 AGX Orin）可以连接
SSH_CMD="ssh -N -L 0.0.0.0:${LOCAL_PORT}:localhost:${REMOTE_PORT}"

# 如果指定了SSH密钥
if [ -n "${SSH_KEY}" ]; then
    SSH_CMD="${SSH_CMD} -i ${SSH_KEY}"
fi

# 添加连接保持参数
SSH_CMD="${SSH_CMD} -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

# 添加远程主机
SSH_CMD="${SSH_CMD} -p ${SSH_PORT} ${REMOTE_USER}@${REMOTE_HOST}"

echo "执行命令:"
echo "  ${SSH_CMD}"
echo ""
echo "建立隧道中..."
echo "按 Ctrl+C 断开连接"
echo ""

# 执行SSH隧道
${SSH_CMD}

echo ""
echo "隧道已断开"

