#!/usr/bin/env python3
"""
黎曼平滑逻辑增强引擎
Riemann Smoothing Logic Enhancement Engine

核心创新：
1. RiemannSmoothingLayer - 基于黎曼几何的逻辑平滑
2. LogicDensityProcessor - 逻辑稠密度增强
3. 解析延拓 - 预测逻辑轨迹
"""

import os
import sys
import logging
import time
import re
import math
from typing import Dict, List, Optional, Any, Generator, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LogitsProcessor

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from core.config import BrainLikeConfig, DEFAULT_CONFIG
from modules.hippocampus import HippocampusSystem


# ============================================================================
# 第一部分：黎曼平滑层
# ============================================================================

class RiemannSmoothingLayer(nn.Module):
    """
    黎曼平滑层
    
    原理：
    1. 计算二阶导数（拉普拉斯算子）
    2. 捕捉逻辑跳跃
    3. 平滑修正补偿
    
    效果：在离散的逻辑点之间拉起平滑曲线
    """
    
    def __init__(self, dim: int, alpha: float = 0.1):
        super().__init__()
        self.dim = dim
        self.alpha = alpha  # 逻辑稠密度权重
        
        # 拉普拉斯算子权重核
        self.register_buffer('laplacian_kernel', torch.tensor([-1.0, 2.0, -1.0]))
        
        logger.info(f"黎曼平滑层初始化: dim={dim}, alpha={alpha}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
        
        Returns:
            平滑后的隐藏状态
        """
        if x.size(1) < 3:
            return x
        
        # 1. 计算二阶差分（拉普拉斯算子）
        # 捕捉逻辑跳跃
        diff = x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :]
        
        # 2. 平滑修正
        # 在离散的逻辑点之间拉起平滑曲线
        x_smoothed = x.clone()
        x_smoothed[:, 1:-1, :] = x[:, 1:-1, :] + self.alpha * diff
        
        return x_smoothed


# ============================================================================
# 第二部分：逻辑稠密度处理器
# ============================================================================

class LogicDensityProcessor(LogitsProcessor):
    """
    逻辑稠密度处理器
    
    原理：
    1. 解析延拓：计算逻辑动量
    2. 逻辑吸引子：预测下一个逻辑点
    3. 平滑惩罚：让逻辑轨迹更符合连续流
    """
    
    def __init__(self, alpha: float = 0.1, window_size: int = 3, vocab_size: int = 248077):
        self.alpha = alpha  # 逻辑平滑强度
        self.window_size = window_size
        self.vocab_size = vocab_size
        
        logger.info(f"逻辑稠密度处理器初始化: alpha={alpha}, window={window_size}")
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: 已生成的token序列 [batch_size, seq_len]
            scores: 当前步的logits [batch_size, vocab_size]
        
        Returns:
            处理后的logits
        """
        if input_ids.shape[1] < self.window_size:
            return scores
        
        # 1. 计算逻辑动量
        recent_tokens = input_ids[0, -self.window_size:].float()
        
        # 2. 解析延拓：预测下一个逻辑点
        # 使用二阶差分预测
        logic_momentum = 2 * recent_tokens[-1] - recent_tokens[-2]
        
        # 3. 计算候选token与预测逻辑点的距离
        candidate_ids = torch.arange(scores.size(1), device=scores.device).float()
        dist_to_attractor = torch.abs(candidate_ids - logic_momentum)
        
        # 4. 归一化距离
        dist_normalized = dist_to_attractor / (dist_to_attractor.max() + 1e-8)
        
        # 5. 应用平滑惩罚
        scores = scores - self.alpha * dist_normalized
        
        return scores


# ============================================================================
# 第三部分：组合增强处理器
# ============================================================================

class CombinedLogicProcessor(LogitsProcessor):
    """
    组合逻辑处理器
    
    整合多种增强方法：
    1. 康托集增强
    2. 斐波那契增强
    3. 素数增强
    4. 黎曼平滑
    """
    
    def __init__(
        self,
        vocab_size: int,
        cantor_strength: float = 0.3,
        fib_strength: float = 0.2,
        prime_strength: float = 0.2,
        smooth_strength: float = 0.1
    ):
        self.vocab_size = vocab_size
        self.cantor_strength = cantor_strength
        self.fib_strength = fib_strength
        self.prime_strength = prime_strength
        self.smooth_strength = smooth_strength
        
        # 预计算增强索引
        self.cantor_indices = self._compute_cantor()
        self.fib_indices = self._compute_fibonacci()
        self.prime_indices = self._compute_primes()
        
        logger.info(f"组合逻辑处理器: 康托集{len(self.cantor_indices)}, "
                   f"斐波那契{len(self.fib_indices)}, 素数{len(self.prime_indices)}")
    
    def _compute_cantor(self) -> List[int]:
        indices = []
        for i in range(min(self.vocab_size, 100000)):
            n = i
            is_cantor = True
            while n > 0:
                if n % 3 == 1:
                    is_cantor = False
                    break
                n //= 3
            if is_cantor:
                indices.append(i)
        return indices
    
    def _compute_fibonacci(self) -> List[int]:
        indices = []
        a, b = 1, 1
        while b < self.vocab_size:
            indices.append(b)
            a, b = b, a + b
        return indices
    
    def _compute_primes(self) -> List[int]:
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0: return False
            return True
        return [i for i in range(min(self.vocab_size, 20000)) if is_prime(i)]
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.clone()
        
        # 1. 康托集增强
        for idx in self.cantor_indices:
            if idx < scores.size(1):
                scores[0, idx] += self.cantor_strength
        
        # 2. 斐波那契增强
        for idx in self.fib_indices:
            if idx < scores.size(1):
                scores[0, idx] += self.fib_strength
        
        # 3. 素数增强
        for idx in self.prime_indices:
            if idx < scores.size(1):
                scores[0, idx] += self.prime_strength
        
        # 4. 黎曼平滑
        if input_ids.shape[1] >= 3:
            recent = input_ids[0, -3:].float()
            momentum = 2 * recent[-1] - recent[-2]
            candidates = torch.arange(scores.size(1), device=scores.device).float()
            dist = torch.abs(candidates - momentum) / (self.vocab_size + 1e-8)
            scores = scores - self.smooth_strength * dist
        
        return scores


# ============================================================================
# 第四部分：黎曼平滑学习引擎
# ============================================================================

class RiemannSmoothingEngine:
    """
    黎曼平滑学习引擎
    
    整合：
    1. 黎曼平滑层 - 隐藏状态平滑
    2. 逻辑稠密度处理器 - 生成时增强
    3. 提示学习 - 知识存储
    """
    
    def __init__(self, model_path: str, config: BrainLikeConfig = None):
        self.model_path = model_path
        self.config = config or DEFAULT_CONFIG
        
        self.model = None
        self.tokenizer = None
        self.hippocampus = None
        
        # 增强组件
        self.riemann_layer: Optional[RiemannSmoothingLayer] = None
        self.logic_processor: Optional[CombinedLogicProcessor] = None
        
        # 知识库
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
        
        logger.info(f"黎曼平滑引擎运行设备: {self.device}")
        self._initialized = False
    
    def initialize(self) -> bool:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"初始化黎曼平滑引擎: {self.model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 初始化海马体
        self.hippocampus = HippocampusSystem(self.config)
        
        # 初始化黎曼平滑层
        hidden_size = self.model.config.hidden_size
        self.riemann_layer = RiemannSmoothingLayer(dim=hidden_size, alpha=0.1)
        self.riemann_layer = self.riemann_layer.to(self.device)
        
        # 初始化逻辑处理器
        vocab_size = len(self.tokenizer)
        self.logic_processor = CombinedLogicProcessor(
            vocab_size=vocab_size,
            cantor_strength=0.3,
            fib_strength=0.2,
            prime_strength=0.2,
            smooth_strength=0.1
        )
        
        self._initialized = True
        logger.info("黎曼平滑引擎初始化完成！")
        return True
    
    def _detect_correction(self, user_input: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """检测用户纠正"""
        correction_keywords = ['不对', '错误', '错了', '不是', '应该是', '正确的是', '答案是']
        
        is_correction = any(kw in user_input for kw in correction_keywords)
        
        if is_correction:
            numbers = re.findall(r'\d+', user_input)
            if numbers:
                correct_answer = numbers[-1]
                return True, correct_answer, user_input
        
        return False, None, None
    
    def _learn_from_correction(self, question: str, correct_answer: str, explanation: str):
        """从纠正中学习"""
        logger.info(f"学习正确模式: {question[:30]}... → {correct_answer}")
        
        pattern = self._extract_pattern(question, correct_answer, explanation)
        
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
        if '房租' in question and '天' in question:
            nums = re.findall(r'\d+', question)
            if len(nums) >= 2:
                return f"房租计算: 日租金=总租金÷天数，月租金=日租金×30"
        
        if '奇数' in question or '偶数' in question:
            return f"范围奇偶数: 找范围内符合条件的数"
        
        return f"问题: {question[:30]}... 答案: {answer}"
    
    def _find_relevant_knowledge(self, question: str) -> List[Dict]:
        """找到相关的知识"""
        relevant = []
        
        for kb in self.knowledge_base:
            score = 0
            
            if '房租' in question and '房租' in kb['question']:
                score += 5
            if '奇数' in question and '奇数' in kb['question']:
                score += 5
            
            if score > 0:
                relevant.append((score, kb))
        
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in relevant[:3]]
    
    def _build_system_prompt(self, question: str) -> str:
        """构建系统提示"""
        base_prompt = "你是AI助手，请准确计算并简洁回答。"
        
        relevant_knowledge = self._find_relevant_knowledge(question)
        
        if relevant_knowledge:
            knowledge_text = "\n\n【已学习的正确模式】"
            for i, kb in enumerate(relevant_knowledge, 1):
                knowledge_text += f"\n{i}. {kb['pattern']}"
            
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
        
        # 2. 学习
        if is_correction and correct_answer:
            if self.history:
                last_question = self.history[-1]['q']
                self._learn_from_correction(last_question, correct_answer, explanation)
                yield f"[已学习此模式]\n\n"
        
        # 3. 构建系统提示
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
                
                # 应用黎曼平滑到隐藏状态
                if hidden.dim() == 3:
                    hidden = self.riemann_layer(hidden)
                
                # 应用逻辑稠密度处理器
                logits = self.logic_processor(input_ids, logits)
                
                # 采样
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
            'history_count': len(self.history),
            'riemann_alpha': self.riemann_layer.alpha if self.riemann_layer else 0
        }
    
    def clear_memory(self):
        self.history = []
        self.knowledge_base = []
        self.hippocampus.clear()
        self.total_memories = 0
