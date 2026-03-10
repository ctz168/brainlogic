from flask import Flask, render_template, Response, request, jsonify
import json
import time
import torch
from core.config import BrainLikeConfig
from core.base_model import BrainLikeQwenModel
from modules.refresh_engine import RefreshEngine

app = Flask(__name__)

# 全局模型与引擎
model = None
engine = None

def init_brain():
    global model, engine
    config = BrainLikeConfig()
    model = BrainLikeQwenModel(config)
    engine = RefreshEngine(model, config)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    # 简单的 token 转换（示例）
    input_ids = torch.tensor([[123, 456]]) 
    
    def generate():
        for cycle in range(10): # 模拟 10 个刷新周期
            result = engine.run_cycle(input_ids[0, 0])
            data = {
                "token": "Hello", # 实际应为解码后的 token
                "cycle": cycle,
                "memory_count": 100, # 模拟实时统计
                "stdp_ltp": 0.05
            }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.01) # 10ms 节奏
            
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    # init_brain() # 在实际运行中启用
    app.run(host='0.0.0.0', port=5000)
