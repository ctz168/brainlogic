#!/usr/bin/env python3
"""
训练后测评脚本
"""

import os
import sys
import json
import time
import logging
import re
import gc
from pathlib import Path
from datetime import datetime

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

# 训练数据
TRAIN_DATA = [
    # 数学
    {"q": "123 + 456 = ?", "a": "579"},
    {"q": "25 × 4 = ?", "a": "100"},
    {"q": "144 ÷ 12 = ?", "a": "12"},
    {"q": "1000 - 382 = ?", "a": "618"},
    {"q": "15 × 15 = ?", "a": "225"},
    # 逻辑
    {"q": "小明比小红高，小红比小华高，谁最高？", "a": "小明最高。"},
    {"q": "1,3,5,7,? 下一个数是什么？", "a": "9。这是奇数序列。"},
    {"q": "2,4,8,16,? 下一个数是什么？", "a": "32。这是2的幂次序列。"},
    # 常识
    {"q": "太阳从哪个方向升起？", "a": "东方。"},
    {"q": "一年有多少个月？", "a": "12个月。"},
    {"q": "水在多少度沸腾？", "a": "100摄氏度。"},
    # 语言
    {"q": "'画蛇添足'是什么意思？", "a": "比喻做多余的事，反而把事情弄坏。"},
    {"q": "'高兴'的反义词是什么？", "a": "悲伤。"},
    {"q": "'守株待兔'比喻什么？", "a": "比喻不主动努力，心存侥幸。"},
]

# 测评数据
TEST_DATA = {
    "math": [
        {"q": "123 + 456 = ?", "a": "579"},
        {"q": "25 × 4 = ?", "a": "100"},
        {"q": "144 ÷ 12 = ?", "a": "12"},
        {"q": "1000 - 382 = ?", "a": "618"},
        {"q": "15 × 15 = ?", "a": "225"},
    ],
    "logic": [
        {"q": "小明比小红高，小红比小华高，谁最高？", "a": "小明"},
        {"q": "1,3,5,7,? 下一个数是什么？", "a": "9"},
        {"q": "2,4,8,16,? 下一个数是什么？", "a": "32"},
    ],
    "common": [
        {"q": "太阳从哪个方向升起？", "a": "东方"},
        {"q": "一年有多少个月？", "a": "12"},
        {"q": "水在多少度沸腾？", "a": "100"},
    ],
    "language": [
        {"q": "'画蛇添足'是什么意思？", "a": "做多余的事"},
        {"q": "'高兴'的反义词是什么？", "a": "悲伤"},
        {"q": "'守株待兔'比喻什么？", "a": "侥幸"},
    ],
}

def train_and_evaluate(model_path: str, epochs: int = 2):
    """训练并测评"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    logger.info("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True
    )
    model = model.to(device)
    
    # 冻结90%权重
    all_params = list(model.named_parameters())
    freeze_count = int(len(all_params) * 0.9)
    for i, (name, param) in enumerate(all_params):
        if i < freeze_count:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数: {trainable/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5, weight_decay=0.01
    )
    
    # 训练前测评
    logger.info("\n" + "="*50)
    logger.info("训练前测评")
    logger.info("="*50)
    
    before_results = evaluate(model, tokenizer, device)
    
    # 训练
    logger.info("\n" + "="*50)
    logger.info(f"开始训练 ({epochs}轮)")
    logger.info("="*50)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for item in TRAIN_DATA:
            input_text = f"<|im_start|>user\n{item['q']}<|im_end|>\n<|im_start|>assistant\n{item['a']}<|im_end|>"
            
            enc = tokenizer(input_text, max_length=64, padding='max_length',
                           truncation=True, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            
            total_loss += loss.item()
            
            del outputs, loss
            gc.collect()
        
        avg_loss = total_loss / len(TRAIN_DATA)
        logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # 训练后测评
    logger.info("\n" + "="*50)
    logger.info("训练后测评")
    logger.info("="*50)
    
    model.eval()
    after_results = evaluate(model, tokenizer, device)
    
    # 对比
    improvement = after_results['overall'] - before_results['overall']
    
    logger.info("\n" + "="*50)
    logger.info("对比结果")
    logger.info("="*50)
    logger.info(f"训练前: {before_results['overall']*100:.1f}%")
    logger.info(f"训练后: {after_results['overall']*100:.1f}%")
    logger.info(f"提升: {improvement*100:+.1f}%")
    
    for cat in before_results['categories']:
        b = before_results['categories'][cat]
        a = after_results['categories'][cat]
        logger.info(f"  {cat}: {b*100:.1f}% → {a*100:.1f}% ({(a-b)*100:+.1f}%)")
    
    return {
        'before': before_results,
        'after': after_results,
        'improvement': improvement,
        'epochs': epochs
    }

def evaluate(model, tokenizer, device):
    """测评"""
    results = {}
    total_correct = 0
    total_questions = 0
    
    for category, items in TEST_DATA.items():
        correct = 0
        total = len(items)
        
        for item in items:
            q = item['q']
            expected = item['a']
            
            input_text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            enc = tokenizer(input_text, return_tensors='pt', max_length=32, truncation=True)
            input_ids = enc['input_ids'].to(device)
            
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=30, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
            
            gen = tokenizer.decode(out[0], skip_special_tokens=True)
            answer = gen.split("assistant")[-1].strip() if "assistant" in gen else gen.strip()
            
            is_correct = expected in answer or any(kw in answer for kw in expected.split())
            
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            logger.info(f"  {status} {q[:20]}... → {answer[:30]}...")
        
        results[category] = correct / total if total > 0 else 0
        total_correct += correct
        total_questions += total
    
    results['overall'] = total_correct / total_questions if total_questions > 0 else 0
    results['categories'] = {k: v for k, v in results.items() if k != 'overall'}
    
    logger.info(f"\n正确率: {results['overall']*100:.1f}% ({total_correct}/{total_questions})")
    
    return results

if __name__ == "__main__":
    model_path = '/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B'
    output_path = '/home/z/my-project/download/brain_like_ai/output'
    
    result = train_and_evaluate(model_path, epochs=2)
    
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'train_eval_result.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info("\n结果已保存")
