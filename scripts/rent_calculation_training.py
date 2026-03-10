#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 房租计算专项训练模块
Rent Calculation Special Training Module

目标：通过训练让模型学会正确推理房租计算问题
约束：仅训练10%动态权重，90%静态权重冻结
"""

import os
import sys
import json
import time
import logging
import gc
import re
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# 房租计算专项训练数据
# ============================================

RENT_TRAINING_DATA = [
    # 基础计算
    {
        "context": "3月12日起租，3月份20天房租1600元。",
        "question": "日租金是多少？",
        "reasoning": "日租金 = 房租金额 ÷ 天数 = 1600 ÷ 20 = 80元/天",
        "answer": "日租金是80元/天。"
    },
    {
        "context": "3月12日起租，3月份20天房租1600元。",
        "question": "月租金是多少？",
        "reasoning": "日租金 = 1600 ÷ 20 = 80元/天\n月租金 = 日租金 × 30 = 80 × 30 = 2400元/月",
        "answer": "月租金是2400元/月。"
    },
    {
        "context": "3月12日起租，3月份20天房租1600元。押金2400元。",
        "question": "押金是多少？",
        "reasoning": "根据信息，押金是2400元。",
        "answer": "押金是2400元。"
    },
    {
        "context": "3月12日起租，3月份20天房租1600元。卫生费200元。离租卫生干净退200元卫生费。",
        "question": "卫生费怎么退？",
        "reasoning": "根据约定，离租时如果卫生干净，可以退还200元卫生费。",
        "answer": "离租时如果卫生干净，可以退还200元卫生费。"
    },
    # 更多变体
    {
        "context": "租房15天，房租900元。",
        "question": "日租金是多少？",
        "reasoning": "日租金 = 900 ÷ 15 = 60元/天",
        "answer": "日租金是60元/天。"
    },
    {
        "context": "租房15天，房租900元。",
        "question": "月租金是多少？",
        "reasoning": "日租金 = 900 ÷ 15 = 60元/天\n月租金 = 60 × 30 = 1800元/月",
        "answer": "月租金是1800元/月。"
    },
    {
        "context": "租房10天，房租500元。",
        "question": "日租金是多少？月租金是多少？",
        "reasoning": "日租金 = 500 ÷ 10 = 50元/天\n月租金 = 50 × 30 = 1500元/月",
        "answer": "日租金是50元/天，月租金是1500元/月。"
    },
    {
        "context": "小王租房25天，房租2000元。押金3000元。",
        "question": "日租金和月租金各是多少？",
        "reasoning": "日租金 = 2000 ÷ 25 = 80元/天\n月租金 = 80 × 30 = 2400元/月",
        "answer": "日租金是80元/天，月租金是2400元/月。"
    },
    {
        "context": "房租30天1800元。",
        "question": "日租金是多少？",
        "reasoning": "日租金 = 1800 ÷ 30 = 60元/天",
        "answer": "日租金是60元/天。"
    },
    {
        "context": "房租30天1800元。",
        "question": "月租金是多少？",
        "reasoning": "这就是一个月的房租，所以月租金是1800元。",
        "answer": "月租金是1800元。"
    },
    # 复杂场景
    {
        "context": "3月12日起租，3月份20天房租1600元。押金:两千四百元；卫生费200元。离租卫生干净退200元卫生费。合计2600元。",
        "question": "月租金是多少？",
        "reasoning": "首先计算日租金：日租金 = 1600 ÷ 20 = 80元/天\n然后计算月租金：月租金 = 80 × 30 = 2400元/月",
        "answer": "月租金是2400元/月。"
    },
    {
        "context": "3月12日起租，3月份20天房租1600元。押金:两千四百元；卫生费200元。离租卫生干净退200元卫生费。合计2600元。",
        "question": "卫生费要怎样才能退？",
        "reasoning": "根据约定，离租时如果卫生干净，可以退还200元卫生费。",
        "answer": "离租时如果卫生干净，可以退还200元卫生费。"
    },
]

# 数学推理训练数据
MATH_TRAINING_DATA = [
    {
        "question": "计算：1600 ÷ 20 = ?",
        "reasoning": "1600 ÷ 20 = 80",
        "answer": "1600 ÷ 20 = 80"
    },
    {
        "question": "计算：80 × 30 = ?",
        "reasoning": "80 × 30 = 2400",
        "answer": "80 × 30 = 2400"
    },
    {
        "question": "如果日租金是80元，月租金是多少？",
        "reasoning": "月租金 = 日租金 × 30 = 80 × 30 = 2400元",
        "answer": "月租金是2400元。"
    },
    {
        "question": "如果房租1600元租了20天，日租金是多少？",
        "reasoning": "日租金 = 房租 ÷ 天数 = 1600 ÷ 20 = 80元/天",
        "answer": "日租金是80元/天。"
    },
]


class RentCalculationTrainer:
    """房租计算专项训练器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cpu")
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        # 训练统计
        self.train_stats = {
            'total_steps': 0,
            'total_loss': 0,
            'best_loss': float('inf')
        }
    
    def setup(self):
        """初始化"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info("加载模型...")
        
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
        self._freeze_weights()
        
        # 创建优化器（仅优化可训练参数）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.01)
        
        logger.info(f"可训练参数: {sum(p.numel() for p in trainable_params)/1e6:.2f}M")
    
    def _freeze_weights(self):
        """冻结90%权重"""
        all_params = list(self.model.named_parameters())
        freeze_count = int(len(all_params) * 0.9)
        
        for i, (name, param) in enumerate(all_params):
            if i < freeze_count:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"冻结权重: {freeze_count}/{len(all_params)} 层")
        logger.info(f"可训练参数: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total*100:.1f}%)")
    
    def train_on_data(self, data: List[Dict], epochs: int = 3):
        """在数据上训练"""
        logger.info(f"\n开始训练，数据量: {len(data)}，轮数: {epochs}")
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for item in data:
                # 构建训练样本
                if 'context' in item:
                    # 带上下文的问题
                    prompt = f"""{item['context']}

问题：{item['question']}

请一步步思考：
{item['reasoning']}

答案：{item['answer']}"""
                else:
                    # 纯问题
                    prompt = f"""问题：{item['question']}

请一步步思考：
{item['reasoning']}

答案：{item['answer']}"""
                
                # 编码
                input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n好的，我来解答。<|im_end|>"
                
                encodings = self.tokenizer(
                    input_text, max_length=256, padding='max_length',
                    truncation=True, return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                labels = input_ids.clone()
                
                # 前向传播
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()
                
                epoch_loss += loss.item()
                self.train_stats['total_steps'] += 1
                
                del outputs, loss
                gc.collect()
            
            avg_loss = epoch_loss / len(data)
            self.train_stats['total_loss'] += avg_loss
            
            if avg_loss < self.train_stats['best_loss']:
                self.train_stats['best_loss'] = avg_loss
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        
        return avg_loss
    
    def train_all(self, epochs: int = 3):
        """训练所有数据"""
        logger.info("="*60)
        logger.info("开始房租计算专项训练")
        logger.info("="*60)
        
        start_time = time.time()
        
        # 训练房租计算
        logger.info("\n[1/2] 房租计算训练")
        self.train_on_data(RENT_TRAINING_DATA, epochs)
        
        # 训练数学推理
        logger.info("\n[2/2] 数学推理训练")
        self.train_on_data(MATH_TRAINING_DATA, epochs)
        
        total_time = time.time() - start_time
        
        logger.info(f"\n训练完成！总耗时: {total_time:.1f}秒")
        logger.info(f"最佳Loss: {self.train_stats['best_loss']:.4f}")
        
        return self.train_stats
    
    def test_generation(self, test_cases: List[Dict]):
        """测试生成效果"""
        logger.info("\n" + "="*60)
        logger.info("测试生成效果")
        logger.info("="*60)
        
        self.model.eval()
        
        for case in test_cases:
            context = case.get('context', '')
            question = case['question']
            expected = case.get('answer', '')
            
            # 构建提示
            if context:
                prompt = f"{context}\n\n问题：{question}\n\n请一步步思考并回答："
            else:
                prompt = f"问题：{question}\n\n请一步步思考并回答："
            
            input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            encodings = self.tokenizer(
                input_text, return_tensors='pt', max_length=128, truncation=True
            )
            input_ids = encodings['input_ids'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "assistant" in generated:
                response = generated.split("assistant")[-1].strip()
            else:
                response = generated.strip()
            
            logger.info(f"\n问题: {question}")
            logger.info(f"期望: {expected}")
            logger.info(f"回答: {response}")
    
    def save_model(self, output_path: str):
        """保存模型"""
        os.makedirs(output_path, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # 保存训练统计
        with open(os.path.join(output_path, 'training_stats.json'), 'w') as f:
            json.dump(self.train_stats, f, indent=2)
        
        logger.info(f"模型已保存到: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='房租计算专项训练')
    parser.add_argument('--model-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B')
    parser.add_argument('--output-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/output/rent_trained_model')
    parser.add_argument('--epochs', type=int, default=3)
    
    args = parser.parse_args()
    
    # 初始化训练器
    trainer = RentCalculationTrainer(args.model_path)
    trainer.setup()
    
    # 训练
    trainer.train_all(epochs=args.epochs)
    
    # 测试
    test_cases = [
        {
            "context": "3月12日起租，3月份20天房租1600元。押金:两千四百元；卫生费200元。离租卫生干净退200元卫生费。合计2600元。",
            "question": "月租金是多少？"
        },
        {
            "context": "3月12日起租，3月份20天房租1600元。押金:两千四百元；卫生费200元。离租卫生干净退200元卫生费。合计2600元。",
            "question": "卫生费要怎样才能退？"
        },
        {
            "context": "租房15天，房租900元。",
            "question": "日租金和月租金各是多少？"
        },
    ]
    
    trainer.test_generation(test_cases)
    
    # 保存
    trainer.save_model(args.output_path)


if __name__ == "__main__":
    main()
