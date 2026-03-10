#!/usr/bin/env python3
"""
轻量级一体式训练 - 内存优化版V2
"""

import os
import sys
import json
import gc
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_and_save():
    """训练并保存"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = str(PROJECT_ROOT / "models/Qwen3.5-0.8B")
    output_path = str(PROJECT_ROOT / "output/integrated_trained")
    
    logger.info("="*60)
    logger.info("一体式优化整合训练")
    logger.info("="*60)
    
    # 加载模型
    logger.info("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    # 冻结90%权重
    all_params = list(model.named_parameters())
    freeze_count = int(len(all_params) * 0.9)
    
    for i, (name, param) in enumerate(all_params):
        if i < freeze_count:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"冻结: {freeze_count}/{len(all_params)}层, 可训练: {trainable/1e6:.2f}M")
    
    # 训练数据
    training_data = [
        {"q": "房租1600元租了20天，日租金是多少？", "r": "日租金 = 1600 ÷ 20 = 80元/天", "a": "日租金是80元/天。"},
        {"q": "房租1600元租了20天，月租金是多少？", "r": "日租金 = 1600 ÷ 20 = 80元/天\n月租金 = 80 × 30 = 2400元/月", "a": "月租金是2400元/月。"},
        {"q": "房租900元租了15天，日租金和月租金各是多少？", "r": "日租金 = 900 ÷ 15 = 60元/天\n月租金 = 60 × 30 = 1800元/月", "a": "日租金是60元/天，月租金是1800元/月。"},
        {"q": "房租500元租了10天，日租金是多少？月租金是多少？", "r": "日租金 = 500 ÷ 10 = 50元/天\n月租金 = 50 × 30 = 1500元/月", "a": "日租金是50元/天，月租金是1500元/月。"},
        {"q": "房租2000元租了25天，日租金是多少？", "r": "日租金 = 2000 ÷ 25 = 80元/天", "a": "日租金是80元/天。"},
        {"q": "日租金80元，月租金是多少？", "r": "月租金 = 日租金 × 30 = 80 × 30 = 2400元", "a": "月租金是2400元/月。"},
        {"q": "日租金60元，月租金是多少？", "r": "月租金 = 日租金 × 30 = 60 × 30 = 1800元", "a": "月租金是1800元/月。"},
        {"q": "日租金50元，月租金是多少？", "r": "月租金 = 日租金 × 30 = 50 × 30 = 1500元", "a": "月租金是1500元/月。"},
    ]
    
    # 优化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5
    )
    
    # 训练
    epochs = 5
    best_loss = float('inf')
    
    logger.info(f"\n开始训练: {epochs}轮")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for item in training_data:
            prompt = f"问题：{item['q']}\n\n请一步步思考：\n{item['r']}\n\n答案：{item['a']}"
            
            inputs = tokenizer(prompt, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
            input_ids = inputs['input_ids']
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            del outputs, loss
            gc.collect()
        
        avg_loss = total_loss / len(training_data)
        if avg_loss < best_loss:
            best_loss = avg_loss
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    # 测试
    logger.info("\n测试训练效果...")
    model.eval()
    
    test_questions = [
        "房租1600元租了20天，日租金是多少？",
        "房租1600元租了20天，月租金是多少？",
    ]
    
    for q in test_questions:
        prompt = f"问题：{q}\n\n请一步步思考并回答："
        inputs = tokenizer(prompt, return_tensors='pt', max_length=64, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_new_tokens=80, do_sample=False)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Q: {q}")
        logger.info(f"A: {response}\n")
    
    # 保存动态权重
    logger.info(f"\n保存动态权重到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # 只保存动态权重
    dynamic_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            dynamic_weights[name] = param.data.cpu()
    
    torch.save(dynamic_weights, os.path.join(output_path, 'dynamic_weights.pt'))
    
    # 保存tokenizer
    tokenizer.save_pretrained(output_path)
    
    # 保存配置
    config = {
        'base_model': model_path,
        'training_time': datetime.now().isoformat(),
        'epochs': epochs,
        'best_loss': best_loss,
        'trainable_params': trainable,
        'dynamic_weights_file': 'dynamic_weights.pt'
    }
    
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("训练完成！动态权重已保存。")
    
    return best_loss


def load_trained_model():
    """加载训练后的模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    output_path = str(PROJECT_ROOT / "output/integrated_trained")
    
    # 加载配置
    with open(os.path.join(output_path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        config['base_model'],
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    # 加载动态权重
    dynamic_weights = torch.load(os.path.join(output_path, 'dynamic_weights.pt'))
    
    # 应用动态权重
    for name, param in model.named_parameters():
        if name in dynamic_weights:
            param.data = dynamic_weights[name]
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(output_path)
    
    return model, tokenizer


if __name__ == "__main__":
    train_and_save()
