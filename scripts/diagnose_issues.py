#!/usr/bin/env python3
"""
问题诊断与修复脚本

问题1: 记忆不行 - 没有上下文记忆
问题2: 推理不对 - 简单数学推理都做错
"""

import os
import sys
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

# ============================================
# 问题诊断
# ============================================

DIAGNOSTIC_CASES = [
    # 记忆测试
    {
        "name": "上下文记忆测试",
        "dialogue": [
            {"role": "user", "content": "3月12日起租，3月份20天房租1600元。押金2400元。"},
            {"role": "assistant", "content": "好的，我记住了。"},
            {"role": "user", "content": "房租是多少？"},
        ],
        "expected": "1600元",
        "problem": "模型没有记住上下文"
    },
    # 推理测试
    {
        "name": "数学推理测试",
        "dialogue": [
            {"role": "user", "content": "3月份20天房租1600元，月租金是多少？"},
        ],
        "expected": "2400元 (1600/20*30=2400)",
        "problem": "推理完全错误"
    },
    {
        "name": "简单计算测试",
        "dialogue": [
            {"role": "user", "content": "20天房租1600元，每天多少钱？"},
        ],
        "expected": "80元 (1600/20=80)",
        "problem": "基础除法都算错"
    },
]


def diagnose_memory_issue():
    """诊断记忆问题"""
    print("="*60)
    print("问题1诊断: 记忆不行")
    print("="*60)
    
    print("\n【问题描述】")
    print("用户: 3月12日起租，3月份20天房租1600元...")
    print("模型: 好的...")
    print("用户: 房租是多少？")
    print("模型: 对不起，我无法为您提供...")  # 完全忘记了！
    
    print("\n【根本原因】")
    print("1. 海马体记忆系统未集成到对话流程")
    print("2. 每次对话都是独立的，没有历史上下文")
    print("3. 会话记忆没有被保存和使用")
    
    print("\n【解决方案】")
    print("1. 实现对话历史管理")
    print("2. 将历史对话编码到海马体")
    print("3. 每次回复前召回相关记忆")


def diagnose_reasoning_issue():
    """诊断推理问题"""
    print("\n" + "="*60)
    print("问题2诊断: 推理不对")
    print("="*60)
    
    print("\n【问题描述】")
    print("用户: 3月份20天房租1600元，月租金是多少？")
    print("期望: 2400元 (1600÷20×30=2400)")
    print("实际: 押金+可用剩余期限...完全无关的计算")
    
    print("\n【根本原因】")
    print("1. 模型(0.8B)太小，推理能力有限")
    print("2. 没有专门的数学推理训练")
    print("3. 模型可能被安全过滤干扰")
    
    print("\n【解决方案】")
    print("1. 增加数学推理专项训练")
    print("2. 实现计算器插件")
    print("3. 使用思维链(CoT)提示")


def test_with_context(model_path: str):
    """测试带上下文的对话"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("\n" + "="*60)
    print("带上下文对话测试")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True
    )
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    # 模拟对话历史
    dialogue_history = [
        {"role": "user", "content": "3月12日起租，3月份20天房租1600元。押金2400元。"},
        {"role": "assistant", "content": "好的，我记住了这些信息。"},
    ]
    
    # 构建带历史的提示
    def build_prompt(history, new_question):
        prompt = "<|im_start|>system\n你是一个智能助手，请记住用户告诉你的信息。<|im_end|>\n"
        for turn in history:
            prompt += f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{new_question}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    
    # 测试1: 房租是多少
    question1 = "房租是多少？"
    prompt1 = build_prompt(dialogue_history, question1)
    
    print(f"\n问题: {question1}")
    
    enc = tokenizer(prompt1, return_tensors='pt', max_length=256, truncation=True)
    input_ids = enc['input_ids'].to(device)
    
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=50, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)
    
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    print(f"回答: {response}")
    
    # 测试2: 月租金计算
    question2 = "20天房租1600元，月租金是多少？请计算：月租金 = 日租金 × 30天"
    prompt2 = build_prompt(dialogue_history, question2)
    
    print(f"\n问题: {question2}")
    
    enc = tokenizer(prompt2, return_tensors='pt', max_length=256, truncation=True)
    input_ids = enc['input_ids'].to(device)
    
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=100, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)
    
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    print(f"回答: {response}")


def test_with_cot(model_path: str):
    """测试思维链推理"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("\n" + "="*60)
    print("思维链(CoT)推理测试")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True
    )
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    # 使用思维链提示
    cot_prompt = """<|im_start|>user
请一步步计算：
已知：20天房租是1600元
求：月租金是多少？

请按以下步骤计算：
1. 先计算日租金 = 1600 ÷ 20 = ?
2. 再计算月租金 = 日租金 × 30 = ?

请给出计算过程和最终答案。
<|im_end|>
<|im_start|>assistant
"""
    
    print("\n使用思维链提示...")
    
    enc = tokenizer(cot_prompt, return_tensors='pt', max_length=128, truncation=True)
    input_ids = enc['input_ids'].to(device)
    
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=150, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)
    
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    print(f"回答: {response}")


def test_with_calculator(model_path: str):
    """测试带计算器的推理"""
    print("\n" + "="*60)
    print("计算器辅助推理测试")
    print("="*60)
    
    # 先用Python计算正确答案
    days = 20
    rent = 1600
    daily_rent = rent / days
    monthly_rent = daily_rent * 30
    
    print(f"\n【正确计算】")
    print(f"日租金 = {rent} ÷ {days} = {daily_rent}元")
    print(f"月租金 = {daily_rent} × 30 = {monthly_rent}元")
    
    print(f"\n正确答案: 月租金是{monthly_rent:.0f}元")


def main():
    model_path = '/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B'
    
    # 诊断
    diagnose_memory_issue()
    diagnose_reasoning_issue()
    
    # 测试
    print("\n" + "="*60)
    print("开始测试...")
    print("="*60)
    
    test_with_context(model_path)
    test_with_cot(model_path)
    test_with_calculator(model_path)
    
    # 总结
    print("\n" + "="*60)
    print("诊断总结")
    print("="*60)
    
    print("""
【问题1: 记忆不行】
原因: 海马体未集成到对话流程
解决: 
  1. 实现对话历史管理
  2. 每轮对话编码到海马体
  3. 回复前召回相关记忆

【问题2: 推理不对】
原因: 模型太小(0.8B)，推理能力有限
解决:
  1. 使用思维链(CoT)提示
  2. 集成计算器插件
  3. 增加推理训练数据
  4. 考虑使用更大模型

【立即可行的方案】
1. 添加对话历史到提示中
2. 使用思维链提示引导推理
3. 对计算类问题使用外部计算器
""")


if __name__ == "__main__":
    main()
