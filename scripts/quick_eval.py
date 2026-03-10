#!/usr/bin/env python3
"""
快速测评脚本
"""

import os
import sys
import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

# 测评数据
TEST_DATA = {
    "memory": [
        {"encode": "我叫张三，今年25岁，是工程师，住北京。", "info": {"姓名": "张三", "年龄": "25", "职业": "工程师", "地点": "北京"}},
        {"encode": "我叫李四，今年30岁，是医生，住上海。", "info": {"姓名": "李四", "年龄": "30", "职业": "医生", "地点": "上海"}},
    ],
    "math": [
        {"q": "123 + 456 = ?", "a": "579"},
        {"q": "25 × 4 = ?", "a": "100"},
        {"q": "144 ÷ 12 = ?", "a": "12"},
    ],
    "logic": [
        {"q": "小明比小红高，小红比小华高，谁最高？", "a": "小明"},
        {"q": "1,3,5,7,? 下一个数是什么？", "a": "9"},
    ],
    "common": [
        {"q": "太阳从哪个方向升起？", "a": "东方"},
        {"q": "一年有多少个月？", "a": "12"},
    ],
    "language": [
        {"q": "'画蛇添足'是什么意思？", "a": "做多余的事"},
        {"q": "'高兴'的反义词是什么？", "a": "悲伤"},
    ],
}

def quick_evaluate(model_path: str):
    """快速测评"""
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
    model.eval()
    
    # 记忆存储
    memory = {}
    results = {}
    
    logger.info("\n" + "="*50)
    logger.info("开始测评")
    logger.info("="*50)
    
    total_correct = 0
    total_questions = 0
    
    for category, items in TEST_DATA.items():
        logger.info(f"\n[{category}]")
        correct = 0
        total = 0
        
        for item in items:
            if 'encode' in item:
                # 编码记忆
                info = item.get('info', {})
                entity = info.get('姓名')
                if entity:
                    memory[entity] = info
                logger.info(f"  记忆: {item['encode'][:30]}...")
                continue
            
            # 问答
            q = item['q']
            expected = item['a']
            
            # 检查记忆
            answer = None
            for entity, info in memory.items():
                if entity in q:
                    if "多大" in q or "年龄" in q:
                        answer = f"{entity}今年{info.get('年龄', '')}岁。"
                    elif "住" in q:
                        answer = f"{entity}住在{info.get('地点', '')}。"
                    elif "职业" in q:
                        answer = f"{entity}是{info.get('职业', '')}。"
                    break
            
            if not answer:
                # 生成答案
                input_text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
                enc = tokenizer(input_text, return_tensors='pt', max_length=32, truncation=True)
                input_ids = enc['input_ids'].to(device)
                
                with torch.no_grad():
                    out = model.generate(input_ids, max_new_tokens=30, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)
                
                gen = tokenizer.decode(out[0], skip_special_tokens=True)
                answer = gen.split("assistant")[-1].strip() if "assistant" in gen else gen.strip()
            
            # 检查
            is_correct = expected in answer or any(kw in answer for kw in expected.split())
            
            total += 1
            total_questions += 1
            if is_correct:
                correct += 1
                total_correct += 1
            
            status = "✓" if is_correct else "✗"
            logger.info(f"  {status} Q: {q[:25]}... A: {answer[:40]}...")
        
        if total > 0:
            results[category] = {'correct': correct, 'total': total, 'accuracy': correct/total}
    
    # 汇总
    overall = total_correct / total_questions if total_questions > 0 else 0
    
    logger.info("\n" + "="*50)
    logger.info("测评结果")
    logger.info("="*50)
    logger.info(f"总正确率: {overall*100:.1f}% ({total_correct}/{total_questions})")
    
    for cat, r in results.items():
        if 'accuracy' in r:
            logger.info(f"  {cat}: {r['accuracy']*100:.1f}%")
    
    return {
        'overall_accuracy': overall,
        'total_correct': total_correct,
        'total_questions': total_questions,
        'category_results': results,
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    model_path = '/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B'
    output_path = '/home/z/my-project/download/brain_like_ai/output'
    
    result = quick_evaluate(model_path)
    
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'quick_eval_result.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n结果已保存")
