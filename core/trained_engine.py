#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 训练后推理引擎
Trained Inference Engine

使用训练后的动态权重进行推理
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, Optional, Generator

import torch

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TrainedInferenceEngine:
    """训练后的推理引擎"""
    
    def __init__(self, base_model_path: str = None, trained_path: str = None):
        self.base_model_path = base_model_path or str(PROJECT_ROOT / "models/Qwen3.5-0.8B")
        self.trained_path = trained_path or str(PROJECT_ROOT / "output/integrated_trained")
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self.config = None
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """初始化"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("="*60)
        logger.info("加载训练后的模型")
        logger.info("="*60)
        
        self.device = torch.device("cpu")
        
        # 加载配置
        config_path = os.path.join(self.trained_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"训练配置: {self.config}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.trained_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        logger.info("加载基础模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 加载动态权重
        weights_path = os.path.join(self.trained_path, 'dynamic_weights.pt')
        if os.path.exists(weights_path):
            logger.info("加载训练后的动态权重...")
            dynamic_weights = torch.load(weights_path, map_location=self.device)
            
            # 应用动态权重
            applied = 0
            for name, param in self.model.named_parameters():
                if name in dynamic_weights:
                    param.data = dynamic_weights[name]
                    applied += 1
            
            logger.info(f"应用了 {applied} 个动态权重")
        else:
            logger.warning("未找到训练后的权重，使用基础模型")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self._initialized = True
        logger.info("模型加载完成！")
        
        return True
    
    def generate_stream(self, prompt: str, max_new_tokens: int = 100) -> Generator[str, None, None]:
        """流式生成"""
        if not self._initialized:
            if not self.initialize():
                yield "初始化失败"
                return
        
        # 构建提示词
        full_prompt = f"问题：{prompt}\n\n请一步步思考并回答："
        
        # 编码
        inputs = self.tokenizer(
            full_prompt,
            return_tensors='pt',
            max_length=128,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 只返回回答部分
        if "请一步步思考并回答：" in full_output:
            answer = full_output.split("请一步步思考并回答：")[-1]
        else:
            answer = full_output
        
        # 流式输出
        for char in answer:
            yield char
    
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """生成完整回答"""
        return "".join(self.generate_stream(prompt, max_new_tokens))


# 全局实例
_engine: Optional[TrainedInferenceEngine] = None

def get_engine() -> TrainedInferenceEngine:
    global _engine
    if _engine is None:
        _engine = TrainedInferenceEngine()
    return _engine
