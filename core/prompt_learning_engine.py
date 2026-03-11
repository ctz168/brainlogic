#!/usr/bin/env python3
"""
纯提示学习引擎
Prompt Learning Engine

核心思想：
不修改权重，只通过prompt注入正确的推理模式
更稳定，不会破坏模型原有能力
"""

import os
import sys
import logging
import time
import re
from typing import Dict, List, Optional, Any, Generator, Tuple
from pathlib import Path

import torch

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from core.config import BrainLikeConfig, DEFAULT_CONFIG
from modules.hippocampus import HippocampusSystem


class PromptLearningEngine:
    """
    纯提示学习引擎
    
    核心创新：
    1. 不修改权重 - 保持模型稳定
    2. 存储正确模式 - 构建知识库
    3. 动态注入prompt - 引导正确推理
    """
    
    def __init__(self, model_path: str, config: BrainLikeConfig = None):
        self.model_path = model_path
        self.config = config or DEFAULT_CONFIG
        
        self.model = None
        self.tokenizer = None
        self.hippocampus = None
        
        # 知识库 - 存储正确的推理模式
        self.knowledge_base: List[Dict] = []
        
        # 会话状态
        self.session: Dict[str, Any] = {}
        self.history: List[Dict] = []
        
        # 统计
        self.total_memories = 0
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"提示学习引擎运行设备: {self.device}")
        self._initialized = False
    
    def initialize(self) -> bool:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"初始化提示学习引擎: {self.model_path}")
        
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
        self.model.eval()
        
        self.hippocampus = HippocampusSystem(self.config)
        
        self._initialized = True
        logger.info("提示学习引擎初始化完成！")
        return True
    
    def _detect_correction(self, user_input: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """检测用户纠正"""
        correction_keywords = ['不对', '错误', '错了', '不是', '应该是', '正确的是', '答案是']
        
        is_correction = any(kw in user_input for kw in correction_keywords)
        
        if is_correction:
            # 提取数字
            numbers = re.findall(r'\d+', user_input)
            if numbers:
                correct_answer = numbers[-1]
                return True, correct_answer, user_input
        
        return False, None, None
    
    def _learn_from_correction(self, question: str, correct_answer: str, explanation: str):
        """从纠正中学习 - 存储到知识库"""
        logger.info(f"学习正确模式: {question[:30]}... → {correct_answer}")
        
        # 提取推理模式
        pattern = self._extract_pattern(question, correct_answer, explanation)
        
        # 存储到知识库
        self.knowledge_base.append({
            'question': question,
            'answer': correct_answer,
            'explanation': explanation,
            'pattern': pattern,
            'timestamp': time.time()
        })
        
        logger.info(f"知识库已更新，共 {len(self.knowledge_base)} 条知识")
    
    def _extract_pattern(self, question: str, answer: str, explanation: str) -> str:
        """提取推理模式"""
        # 房租计算模式
        if '房租' in question and '天' in question:
            nums = re.findall(r'\d+', question)
            if len(nums) >= 2:
                days = nums[0] if '天' in question.split(nums[0])[1][:10] else nums[1]
                rent = nums[1] if nums[0] == days else nums[0]
                return f"房租计算: {rent}元租{days}天 → 日租金={rent}÷{days}元，月租金=日租金×30"
        
        # 奇偶数模式
        if '奇数' in question or '偶数' in question:
            nums = re.findall(r'\d+', question)
            if len(nums) >= 2:
                return f"范围奇偶数: {nums[0]}到{nums[1]}之间的{'最小奇数' if '最小奇数' in question else '最大奇数' if '最大奇数' in question else '数'}"
        
        return f"问题: {question[:30]}... 答案: {answer}"
    
    def _find_relevant_knowledge(self, question: str) -> List[Dict]:
        """找到相关的知识"""
        relevant = []
        
        for kb in self.knowledge_base:
            # 简单的关键词匹配
            score = 0
            q_words = set(question)
            kb_words = set(kb['question'])
            
            # 共同关键词
            common = q_words & kb_words
            score = len(common)
            
            # 特殊模式匹配
            if '房租' in question and '房租' in kb['question']:
                score += 5
            if '奇数' in question and '奇数' in kb['question']:
                score += 5
            if '偶数' in question and '偶数' in kb['question']:
                score += 5
            
            if score > 0:
                relevant.append((score, kb))
        
        # 按分数排序
        relevant.sort(key=lambda x: x[0], reverse=True)
        
        return [r[1] for r in relevant[:3]]
    
    def _build_system_prompt(self, question: str) -> str:
        """构建系统提示"""
        base_prompt = "你是AI助手，请准确计算并简洁回答。"
        
        # 添加相关知识
        relevant_knowledge = self._find_relevant_knowledge(question)
        
        if relevant_knowledge:
            knowledge_text = "\n\n【已学习的正确模式】"
            for i, kb in enumerate(relevant_knowledge, 1):
                knowledge_text += f"\n{i}. {kb['pattern']}"
                knowledge_text += f"\n   示例: {kb['question'][:40]}... → {kb['answer']}"
            
            base_prompt += knowledge_text
        
        return base_prompt
    
    def _store_memory(self, content: str, hidden_state: torch.Tensor, importance: float = 1.0):
        """存储记忆"""
        memory_id = self.hippocampus.encode_episode(
            features=hidden_state.squeeze(0),
            timestamp_ms=time.time() * 1000,
            semantic_info={
                'content': content,
                'importance': importance
            }
        )
        self.total_memories += 1
        return memory_id
    
    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        return "".join(list(self.generate_stream(prompt, max_new_tokens)))
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 400
    ) -> Generator[str, None, None]:
        if not self._initialized:
            if not self.initialize():
                yield "初始化失败"
                return
        
        # 1. 检测纠正
        is_correction, correct_answer, explanation = self._detect_correction(prompt)
        
        # 2. 如果是纠正，学习
        if is_correction and correct_answer:
            if self.history:
                last_question = self.history[-1]['q']
                self._learn_from_correction(last_question, correct_answer, explanation)
                yield f"[已学习此模式]\n\n"
        
        # 3. 构建动态系统提示
        system_prompt = self._build_system_prompt(prompt)
        
        # 4. 构建消息
        messages = [{"role": "system", "content": system_prompt}]
        
        for h in self.history[-3:]:
            messages.append({"role": "user", "content": h['q']})
            messages.append({"role": "assistant", "content": h['a'][:100]})
        
        messages.append({"role": "user", "content": prompt})
        
        # 5. 生成
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        
        generated = []
        past = None
        response = ""
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                out = self.model(
                    input_ids=input_ids if past is None else input_ids[:, -1:],
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True
                )
                
                logits = out.logits[:, -1, :].clone()
                hidden = out.hidden_states[-1][:, -1, :]
                past = out.past_key_values
                
                logits = logits / 0.6
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                
                tid = next_tok.item()
                if tid == self.tokenizer.eos_token_id:
                    break
                
                tok_text = self.tokenizer.decode([tid], skip_special_tokens=False)
                if "<|im_" in tok_text:
                    break
                
                generated.append(tid)
                text = self.tokenizer.decode([tid], skip_special_tokens=True)
                response += text
                yield text
                
                input_ids = torch.cat([input_ids, next_tok], dim=-1)
        
        # 6. 存储记忆
        self._store_memory(
            content=f"Q: {prompt} A: {response[:100]}",
            hidden_state=hidden,
            importance=2.0 if is_correction else 1.0
        )
        
        # 7. 保存历史
        self.history.append({'q': prompt, 'a': response})
        if len(self.history) > 20:
            self.history = self.history[-20:]
    
    def get_statistics(self) -> Dict:
        return {
            'initialized': self._initialized,
            'device': str(self.device),
            'knowledge_base_size': len(self.knowledge_base),
            'total_memories': self.total_memories,
            'history_count': len(self.history)
        }
    
    def clear_memory(self):
        self.history = []
        self.knowledge_base = []
        self.hippocampus.clear()
        self.total_memories = 0
