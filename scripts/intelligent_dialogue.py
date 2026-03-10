#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 智能对话系统
Intelligent Dialogue System with Memory and CoT

解决问题：
1. 记忆不行 - 添加对话历史管理
2. 推理不对 - 使用思维链(CoT)提示
"""

import os
import sys
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


@dataclass
class DialogueTurn:
    """对话轮次"""
    role: str  # user 或 assistant
    content: str
    timestamp: float = field(default_factory=time.time)
    memory_encoded: bool = False


class IntelligentDialogueSystem:
    """
    智能对话系统
    
    特点：
    1. 对话历史管理 - 解决记忆问题
    2. 思维链推理 - 解决推理问题
    3. 计算器集成 - 精确计算
    """
    
    def __init__(self, model_path: str, max_history: int = 10):
        self.model_path = model_path
        self.max_history = max_history
        
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # 对话历史
        self.dialogue_history: List[DialogueTurn] = []
        
        # 记忆存储（简化版）
        self.memory_store: Dict[str, Any] = {}
        
        # 统计
        self.stats = {
            'total_turns': 0,
            'memory_recalls': 0,
            'cot_uses': 0,
            'calculator_uses': 0
        }
    
    def setup(self):
        """初始化"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info("初始化智能对话系统...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = torch.device("cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("智能对话系统初始化完成")
    
    def chat(self, user_input: str) -> str:
        """
        对话主入口
        
        Args:
            user_input: 用户输入
            
        Returns:
            模型回复
        """
        self.stats['total_turns'] += 1
        
        # 1. 检查是否需要计算
        if self._needs_calculation(user_input):
            return self._handle_calculation(user_input)
        
        # 2. 检查是否需要推理
        if self._needs_reasoning(user_input):
            return self._handle_reasoning(user_input)
        
        # 3. 检查是否是记忆查询
        if self._is_memory_query(user_input):
            return self._handle_memory_query(user_input)
        
        # 4. 普通对话
        return self._handle_normal_chat(user_input)
    
    def _needs_calculation(self, text: str) -> bool:
        """检查是否需要计算"""
        calc_patterns = [
            r'\d+\s*[+\-×÷*/]\s*\d+',  # 数学表达式
            r'多少.*\d+',  # 多少钱、多少天等
            r'计算',  # 计算关键词
            r'等于',  # 等于
            r'房租.*\d+',  # 房租相关
            r'租金.*\d+',  # 租金相关
        ]
        return any(re.search(p, text) for p in calc_patterns)
    
    def _needs_reasoning(self, text: str) -> bool:
        """检查是否需要推理"""
        reasoning_patterns = [
            r'为什么',
            r'怎么.*算',
            r'如何.*算',
            r'推理',
            r'分析',
            r'月租.*是多少',
            r'日租.*是多少',
        ]
        return any(re.search(p, text) for p in reasoning_patterns)
    
    def _is_memory_query(self, text: str) -> bool:
        """检查是否是记忆查询"""
        memory_patterns = [
            r'我(刚才|之前|上次)说',
            r'你(还记得|记住)',
            r'房租是多少',
            r'押金是多少',
            r'什么时候',
        ]
        return any(re.search(p, text) for p in memory_patterns)
    
    def _handle_calculation(self, user_input: str) -> str:
        """处理计算类问题"""
        self.stats['calculator_uses'] += 1
        
        # 提取数字和关系
        numbers = re.findall(r'\d+', user_input)
        
        # 特殊处理房租计算
        if '房租' in user_input or '租金' in user_input:
            return self._calculate_rent(user_input, numbers)
        
        # 使用思维链处理
        return self._handle_reasoning(user_input)
    
    def _calculate_rent(self, user_input: str, numbers: List[str]) -> str:
        """计算房租"""
        # 从历史中获取信息
        context = self._get_relevant_context(user_input)
        
        # 尝试从输入或历史中提取信息
        days = None
        rent = None
        
        # 从当前输入提取
        days_match = re.search(r'(\d+)\s*天', user_input)
        rent_match = re.search(r'(\d+)\s*元', user_input)
        
        if days_match:
            days = int(days_match.group(1))
        if rent_match:
            rent = int(rent_match.group(1))
        
        # 从历史中提取
        for turn in reversed(self.dialogue_history):
            if days is None:
                days_m = re.search(r'(\d+)\s*天', turn.content)
                if days_m:
                    days = int(days_m.group(1))
            if rent is None:
                rent_m = re.search(r'(\d+)\s*元', turn.content)
                if rent_m:
                    rent = int(rent_m.group(1))
        
        # 计算
        if days and rent:
            daily_rent = rent / days
            monthly_rent = daily_rent * 30
            
            result = f"""【计算结果】

日租金 = {rent}元 ÷ {days}天 = {daily_rent:.0f}元/天
月租金 = {daily_rent:.0f}元/天 × 30天 = {monthly_rent:.0f}元

所以月租金是 {monthly_rent:.0f} 元。"""
            
            # 存储到记忆
            self._store_memory('日租金', daily_rent)
            self._store_memory('月租金', monthly_rent)
            
            return result
        
        # 无法计算，使用思维链
        return self._handle_reasoning(user_input)
    
    def _handle_reasoning(self, user_input: str) -> str:
        """使用思维链处理推理"""
        self.stats['cot_uses'] += 1
        
        # 构建思维链提示
        prompt = self._build_cot_prompt(user_input)
        
        # 生成回复
        response = self._generate(prompt, max_new_tokens=200)
        
        # 添加到历史
        self._add_turn('user', user_input)
        self._add_turn('assistant', response)
        
        return response
    
    def _handle_memory_query(self, user_input: str) -> str:
        """处理记忆查询"""
        self.stats['memory_recalls'] += 1
        
        # 从历史中查找相关信息
        context = self._get_relevant_context(user_input)
        
        # 构建带上下文的提示
        prompt = self._build_context_prompt(user_input, context)
        
        # 生成回复
        response = self._generate(prompt, max_new_tokens=100)
        
        # 添加到历史
        self._add_turn('user', user_input)
        self._add_turn('assistant', response)
        
        return response
    
    def _handle_normal_chat(self, user_input: str) -> str:
        """处理普通对话"""
        # 构建提示
        prompt = self._build_normal_prompt(user_input)
        
        # 生成回复
        response = self._generate(prompt, max_new_tokens=100)
        
        # 添加到历史
        self._add_turn('user', user_input)
        self._add_turn('assistant', response)
        
        return response
    
    def _build_cot_prompt(self, user_input: str) -> str:
        """构建思维链提示"""
        # 获取相关上下文
        context = self._get_relevant_context(user_input)
        
        prompt = "<|im_start|>system\n你是一个智能助手，擅长数学计算和逻辑推理。请一步步思考问题。\n"
        
        # 添加历史上下文
        if context:
            prompt += f"\n【已知信息】\n{context}\n"
        
        prompt += f"""<|im_end|>
<|im_start|>user
{user_input}

请按以下步骤思考：
1. 理解问题：需要计算什么？
2. 提取数据：有哪些已知数字？
3. 确定方法：用什么公式计算？
4. 执行计算：一步步算出结果
5. 验证答案：结果是否合理？

请给出详细的计算过程。
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def _build_context_prompt(self, user_input: str, context: str) -> str:
        """构建带上下文的提示"""
        prompt = "<|im_start|>system\n你是一个智能助手，请根据对话历史回答问题。\n"
        
        # 添加对话历史
        prompt += "\n【对话历史】\n"
        for turn in self.dialogue_history[-self.max_history:]:
            role = "用户" if turn.role == "user" else "助手"
            prompt += f"{role}: {turn.content}\n"
        
        prompt += f"""<|im_end|>
<|im_start|>user
{user_input}
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def _build_normal_prompt(self, user_input: str) -> str:
        """构建普通提示"""
        prompt = "<|im_start|>system\n你是一个智能助手。\n"
        
        # 添加最近的对话历史
        if self.dialogue_history:
            prompt += "\n【对话历史】\n"
            for turn in self.dialogue_history[-3:]:  # 只保留最近3轮
                role = "用户" if turn.role == "user" else "助手"
                prompt += f"{role}: {turn.content}\n"
        
        prompt += f"""<|im_end|>
<|im_start|>user
{user_input}
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def _get_relevant_context(self, query: str) -> str:
        """获取相关上下文"""
        context_parts = []
        
        # 从对话历史中提取相关信息
        for turn in self.dialogue_history:
            # 简单的关键词匹配
            if any(kw in turn.content for kw in ['房租', '租金', '押金', '元', '天']):
                context_parts.append(turn.content)
        
        # 从记忆存储中获取
        for key, value in self.memory_store.items():
            if key in query or any(kw in key for kw in ['房租', '租金', '押金']):
                context_parts.append(f"{key}: {value}")
        
        return "\n".join(context_parts[-5:]) if context_parts else ""
    
    def _store_memory(self, key: str, value: Any):
        """存储记忆"""
        self.memory_store[key] = value
        logger.debug(f"存储记忆: {key} = {value}")
    
    def _add_turn(self, role: str, content: str):
        """添加对话轮次"""
        turn = DialogueTurn(role=role, content=content)
        self.dialogue_history.append(turn)
        
        # 限制历史长度
        if len(self.dialogue_history) > self.max_history * 2:
            self.dialogue_history = self.dialogue_history[-self.max_history:]
    
    def _generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """生成回复"""
        enc = self.tokenizer(
            prompt, return_tensors='pt', 
            max_length=512, truncation=True
        )
        input_ids = enc['input_ids'].to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(out[0], skip_special_tokens=True)
        
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        return response
    
    def clear_history(self):
        """清空对话历史"""
        self.dialogue_history.clear()
        self.memory_store.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'history_length': len(self.dialogue_history),
            'memory_count': len(self.memory_store)
        }


# ============================================
# 测试
# ============================================

def test_dialogue_system(model_path: str):
    """测试对话系统"""
    system = IntelligentDialogueSystem(model_path)
    system.setup()
    
    print("="*60)
    print("智能对话系统测试")
    print("="*60)
    
    # 模拟用户对话
    dialogues = [
        "3月12日起租，3月份20天房租1600元。押金2400元。",
        "房租是多少？",
        "月租金是多少？",
        "日租金是多少？",
    ]
    
    for user_input in dialogues:
        print(f"\n用户: {user_input}")
        response = system.chat(user_input)
        print(f"助手: {response}")
    
    print("\n" + "="*60)
    print("统计信息")
    print("="*60)
    stats = system.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    model_path = '/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B'
    test_dialogue_system(model_path)
