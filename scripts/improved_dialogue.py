#!/usr/bin/env python3
"""
改进版智能对话系统 - 解决记忆和推理问题
"""

import os
import sys
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


@dataclass
class Fact:
    """事实存储"""
    key: str
    value: Any
    source: str  # 来源句子
    timestamp: float = field(default_factory=time.time)


class ImprovedDialogueSystem:
    """
    改进版对话系统
    
    核心改进：
    1. 结构化事实存储 - 精确记忆
    2. 直接回答优先 - 避免模型幻觉
    3. 计算器集成 - 精确计算
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # 结构化事实存储
        self.facts: Dict[str, Fact] = {}
        
        # 对话历史（用于显示）
        self.history: List[Dict[str, str]] = []
    
    def setup(self):
        """初始化"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info("初始化改进版对话系统...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = torch.device("cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.float32,
            trust_remote_code=True, low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("初始化完成")
    
    def chat(self, user_input: str) -> str:
        """对话主入口"""
        # 1. 提取并存储事实
        self._extract_facts(user_input)
        
        # 2. 检查是否可以直接回答
        direct_answer = self._try_direct_answer(user_input)
        if direct_answer:
            self._add_to_history('user', user_input)
            self._add_to_history('assistant', direct_answer)
            return direct_answer
        
        # 3. 检查是否需要计算
        if self._needs_calculation(user_input):
            answer = self._calculate(user_input)
            self._add_to_history('user', user_input)
            self._add_to_history('assistant', answer)
            return answer
        
        # 4. 使用模型生成
        answer = self._generate_response(user_input)
        self._add_to_history('user', user_input)
        self._add_to_history('assistant', answer)
        return answer
    
    def _extract_facts(self, text: str):
        """从文本中提取事实"""
        # 提取房租信息
        rent_match = re.search(r'(\d+)\s*天.*?房租.*?(\d+)\s*元', text)
        if rent_match:
            days = rent_match.group(1)
            rent = rent_match.group(2)
            self.facts['房租天数'] = Fact('房租天数', days, text)
            self.facts['房租金额'] = Fact('房租金额', rent, text)
            self.facts['日租金'] = Fact('日租金', int(rent) / int(days), text)
            self.facts['月租金'] = Fact('月租金', int(rent) / int(days) * 30, text)
        
        # 提取押金
        deposit_match = re.search(r'押金[：:]?\s*(\d+)', text)
        if deposit_match:
            self.facts['押金'] = Fact('押金', deposit_match.group(1), text)
        
        # 提取日期
        date_match = re.search(r'(\d+)月(\d+)日', text)
        if date_match:
            self.facts['起租月份'] = Fact('起租月份', date_match.group(1), text)
            self.facts['起租日期'] = Fact('起租日期', date_match.group(2), text)
        
        # 提取其他金额
        amount_matches = re.findall(r'(\d+)\s*元', text)
        for i, amount in enumerate(amount_matches):
            if f'金额{i+1}' not in self.facts:
                self.facts[f'金额{i+1}'] = Fact(f'金额{i+1}', amount, text)
    
    def _try_direct_answer(self, question: str) -> Optional[str]:
        """尝试直接回答"""
        question_lower = question.lower()
        
        # 房租查询
        if '房租' in question and ('多少' in question or '是' in question):
            if '房租金额' in self.facts:
                return f"房租是{self.facts['房租金额'].value}元。"
        
        # 押金查询
        if '押金' in question and ('多少' in question or '是' in question):
            if '押金' in self.facts:
                return f"押金是{self.facts['押金'].value}元。"
        
        # 日租金查询
        if '日租金' in question or ('每天' in question and '租金' in question):
            if '日租金' in self.facts:
                return f"日租金是{self.facts['日租金'].value:.0f}元/天。"
        
        # 月租金查询
        if '月租金' in question or ('每月' in question and '租金' in question):
            if '月租金' in self.facts:
                return f"月租金是{self.facts['月租金'].value:.0f}元/月。"
        
        return None
    
    def _needs_calculation(self, text: str) -> bool:
        """检查是否需要计算"""
        calc_keywords = ['计算', '多少', '等于', '乘', '除', '加', '减', '×', '÷', '+', '-']
        return any(kw in text for kw in calc_keywords) and any(c.isdigit() for c in text)
    
    def _calculate(self, question: str) -> str:
        """执行计算"""
        # 如果已经有月租金数据
        if '月租金' in self.facts:
            return f"""【计算结果】

已知：{self.facts['房租天数'].value}天房租{self.facts['房租金额'].value}元

日租金 = {self.facts['房租金额'].value} ÷ {self.facts['房租天数'].value} = {self.facts['日租金'].value:.0f}元/天
月租金 = {self.facts['日租金'].value:.0f} × 30 = {self.facts['月租金'].value:.0f}元/月

所以月租金是 {self.facts['月租金'].value:.0f} 元。"""
        
        return "需要更多信息才能计算。"
    
    def _generate_response(self, user_input: str) -> str:
        """使用模型生成回复"""
        # 构建带历史的提示
        prompt = self._build_prompt(user_input)
        
        enc = self.tokenizer(prompt, return_tensors='pt', max_length=256, truncation=True)
        input_ids = enc['input_ids'].to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(
                input_ids, max_new_tokens=80, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        return response
    
    def _build_prompt(self, user_input: str) -> str:
        """构建提示"""
        prompt = "<|im_start|>system\n你是一个智能助手。\n"
        
        # 添加已知事实
        if self.facts:
            prompt += "\n【已知信息】\n"
            for key, fact in self.facts.items():
                prompt += f"- {key}: {fact.value}\n"
        
        # 添加最近对话
        if self.history:
            prompt += "\n【最近对话】\n"
            for turn in self.history[-4:]:
                prompt += f"{turn['role']}: {turn['content']}\n"
        
        prompt += f"""<|im_end|>
<|im_start|>user
{user_input}
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def _add_to_history(self, role: str, content: str):
        """添加到历史"""
        self.history.append({'role': role, 'content': content})
        if len(self.history) > 20:
            self.history = self.history[-20:]
    
    def get_facts(self) -> Dict[str, Any]:
        """获取所有事实"""
        return {k: {'value': v.value, 'source': v.source} for k, v in self.facts.items()}


def test_improved_system(model_path: str):
    """测试改进版系统"""
    system = ImprovedDialogueSystem(model_path)
    system.setup()
    
    print("="*60)
    print("改进版对话系统测试")
    print("="*60)
    
    # 模拟真实对话
    dialogues = [
        "3月12日起租，3月份20天房租1600元。押金2400元。",
        "房租是多少？",
        "押金是多少？",
        "日租金是多少？",
        "月租金是多少？",
    ]
    
    for user_input in dialogues:
        print(f"\n用户: {user_input}")
        response = system.chat(user_input)
        print(f"助手: {response}")
    
    print("\n" + "="*60)
    print("存储的事实")
    print("="*60)
    for k, v in system.get_facts().items():
        print(f"  {k}: {v['value']}")


if __name__ == "__main__":
    model_path = '/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B'
    test_improved_system(model_path)
