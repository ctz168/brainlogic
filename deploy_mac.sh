#!/bin/bash
# 类人脑双系统全闭环AI架构 - Mac 一键启动脚本
# Mac One-Click Deployment Script

echo "===================================================="
echo "    Brain-Like AI Dual-System Deployment (Mac)    "
echo "===================================================="

# 1. 环境检查
echo "[1/4] 检查环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装。"
    exit 1
fi

# 2. 安装依赖
echo "[2/4] 安装核心依赖..."
pip3 install torch torchvision torchaudio
pip3 install transformers flask accelerate sentencepiece
pip3 install wikipedia-api faiss-cpu

# 3. 初始化权重与配置
echo "[3/4] 初始化模型权重 (默认 Qwen3.5-0.8B)..."
mkdir -p weights
# 实际应从 GitHub 或 HuggingFace 拉取，这里假设已存在
python3 core/config.py --init

# 4. 启动服务
echo "[4/4] 启动 Web 监控界面..."
echo "访问地址: http://localhost:5000"
export FLASK_APP=web/app.py
python3 -m flask run --port=5000
