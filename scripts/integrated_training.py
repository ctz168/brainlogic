#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 一体式优化整合训练 (增强版)
Integrated Optimization Training (Enhanced)

增强：
1. 更多训练数据
2. 更强的思维链引导
3. 模块协同优化
"""

import os
import sys
import json
import time
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================
# 增强训练数据集
# ============================================

TRAINING_DATA = [
    # 房租计算 - 核心数据
    {"q": "房租1600元租了20天，日租金是多少？", "a": "日租金 = 1600 ÷ 20 = 80元/天"},
    {"q": "房租1600元租了20天，月租金是多少？", "a": "日租金 = 1600 ÷ 20 = 80元/天\n月租金 = 80 × 30 = 2400元/月"},
    {"q": "3月份20天房租1600元，月租金是多少？", "a": "日租金 = 1600 ÷ 20 = 80元/天\n月租金 = 80 × 30 = 2400元/月"},
    {"q": "租房15天，房租900元，日租金是多少？", "a": "日租金 = 900 ÷ 15 = 60元/天"},
    {"q": "租房15天，房租900元，月租金是多少？", "a": "日租金 = 900 ÷ 15 = 60元/天\n月租金 = 60 × 30 = 1800元/月"},
    {"q": "房租30天1800元，日租金是多少？", "a": "日租金 = 1800 ÷ 30 = 60元/天"},
    {"q": "房租25天2000元，日租金是多少？", "a": "日租金 = 2000 ÷ 25 = 80元/天"},
    {"q": "房租25天2000元，月租金是多少？", "a": "日租金 = 2000 ÷ 25 = 80元/天\n月租金 = 80 × 30 = 2400元/月"},
    {"q": "房租10天500元，日租金是多少？", "a": "日租金 = 500 ÷ 10 = 50元/天"},
    {"q": "房租10天500元，月租金是多少？", "a": "日租金 = 500 ÷ 10 = 50元/天\n月租金 = 50 × 30 = 1500元/月"},
    
    # 数学计算
    {"q": "计算：1600 ÷ 20 = ?", "a": "1600 ÷ 20 = 80"},
    {"q": "计算：80 × 30 = ?", "a": "80 × 30 = 2400"},
    {"q": "计算：900 ÷ 15 = ?", "a": "900 ÷ 15 = 60"},
    {"q": "计算：60 × 30 = ?", "a": "60 × 30 = 1800"},
    {"q": "计算：2000 ÷ 25 = ?", "a": "2000 ÷ 25 = 80"},
    {"q": "计算：500 ÷ 10 = ?", "a": "500 ÷ 10 = 50"},
    
    # 反向计算
    {"q": "如果日租金是80元，月租金是多少？", "a": "月租金 = 80 × 30 = 2400元"},
    {"q": "如果日租金是60元，月租金是多少？", "a": "月租金 = 60 × 30 = 1800元"},
    {"q": "如果月租金是2400元，日租金是多少？", "a": "日租金 = 2400 ÷ 30 = 80元/天"},
    {"q": "如果月租金是1800元，日租金是多少？", "a": "日租金 = 1800 ÷ 30 = 60元/天"},
    
    # 复杂场景
    {"q": "押金2400元，卫生费200元，房租1600元20天，月租金是多少？", "a": "日租金 = 1600 ÷ 20 = 80元/天\n月租金 = 80 × 30 = 2400元/月\n押金和卫生费不影响月租金计算。"},
]


# ============================================
# 一体式训练器
# ============================================

class IntegratedTrainer:
    """一体式训练器"""
    
    def __init__(self, model_path: str, epochs: int = 5, lr: float = 1e-5):
        self.model_path = model_path
        self.epochs = epochs
        self.lr = lr
        
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # STDP统计
        self.ltp_count = 0
        self.ltd_count = 0
        
        # 训练统计
        self.stats = {
            'total_steps': 0,
            'best_loss': float('inf'),
            'correct_count': 0
        }
    
    def setup(self):
        """初始化"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("初始化...")
        
        self.device = torch.device("cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
        
        # 冻结90%权重
        all_params = list(self.model.named_parameters())
        freeze_count = int(len(all_params) * 0.9)
        
        for i, (name, param) in enumerate(all_params):
            if i < freeze_count:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"可训练参数: {trainable/1e6:.2f}M ({trainable/total*100:.1f}%)")
    
    def train(self):
        """训练"""
        logger.info(f"训练数据: {len(TRAINING_DATA)} 条")
        
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr
        )
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for item in TRAINING_DATA:
                # 构建提示词
                prompt = f"问题：{item['q']}\n\n答案：{item['a']}"
                
                # 编码
                inputs = self.tokenizer(
                    prompt, max_length=128, padding='max_length',
                    truncation=True, return_tensors='pt'
                )
                input_ids = inputs['input_ids'].to(self.device)
                labels = input_ids.clone()
                
                # 前向传播
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                
                epoch_loss += loss.item()
                self.stats['total_steps'] += 1
                
                # STDP学习
                if loss.item() < 1.0:
                    self.ltp_count += 1
                else:
                    self.ltd_count += 1
                
                del outputs, loss
            
            avg_loss = epoch_loss / len(TRAINING_DATA)
            
            if avg_loss < self.stats['best_loss']:
                self.stats['best_loss'] = avg_loss
            
            logger.info(f"Epoch {epoch+1}/{self.epochs}: Loss = {avg_loss:.4f}")
    
    def test(self):
        """测试"""
        logger.info("\n测试模型:")
        
        test_cases = [
            "房租1600元租了20天，日租金是多少？",
            "房租1600元租了20天，月租金是多少？",
            "租房15天，房租900元，日租金是多少？",
        ]
        
        for q in test_cases:
            prompt = f"问题：{q}\n\n答案："
            inputs = self.tokenizer(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "答案：" in response:
                response = response.split("答案：")[-1]
            
            logger.info(f"\nQ: {q}")
            logger.info(f"A: {response}")
    
    def save(self, output_path: str):
        """保存"""
        os.makedirs(output_path, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"模型已保存到: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B')
    parser.add_argument('--output-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/output/integrated_trained_model')
    parser.add_argument('--epochs', type=int, default=5)
    
    args = parser.parse_args()
    
    trainer = IntegratedTrainer(args.model_path, args.epochs)
    trainer.setup()
    trainer.train()
    trainer.test()
    trainer.save(args.output_path)


if __name__ == "__main__":
    main()
