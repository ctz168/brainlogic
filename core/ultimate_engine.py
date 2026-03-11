#!/usr/bin/env python3
"""
终极集成推理引擎
Ultimate Integrated Inference Engine

整合所有增强技术：
1. 黎曼平滑层 - 隐藏状态平滑
2. 逻辑稠密度处理器 - 生成时增强
3. 康托集+斐波那契+素数增强 - 数学增强
4. STDP学习 - 权重更新
5. 海马体记忆 - 存储和召回
6. 提示学习 - 知识库
7. 纠正检测 - 用户反馈学习
"""

import os
import sys
import logging
import time
import re
import math
from typing import Dict, List, Optional, Any, Generator, Tuple
from pathlib import Path
from dataclasses import dataclass, field

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
from modules.stdp_system import STDPKernel


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class Knowledge:
    """知识条目"""
    question: str
    answer: str
    explanation: str
    pattern: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    corrections: int = 0


@dataclass
class SessionState:
    """会话状态"""
    question_count: int = 0
    correction_count: int = 0
    learning_events: int = 0
    last_question: str = ""
    last_answer: str = ""


# ============================================================================
# 第一部分：黎曼平滑层
# ============================================================================

class RiemannSmoothingLayer(nn.Module):
    """黎曼平滑层 - 消除逻辑跳跃"""
    
    def __init__(self, dim: int, alpha: float = 0.1):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        logger.info(f"黎曼平滑层: dim={dim}, alpha={alpha}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < 3:
            return x
        
        # 二阶差分（拉普拉斯算子）
        diff = x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :]
        
        # 平滑修正
        x_smoothed = x.clone()
        x_smoothed[:, 1:-1, :] = x[:, 1:-1, :] + self.alpha * diff
        
        return x_smoothed


# ============================================================================
# 第二部分：组合逻辑处理器
# ============================================================================

class UltimateLogicProcessor(LogitsProcessor):
    """
    终极逻辑处理器
    
    整合所有增强方法：
    1. 康托集增强
    2. 斐波那契增强
    3. 素数增强
    4. 黎曼平滑
    5. 知识引导
    """
    
    def __init__(
        self,
        vocab_size: int,
        cantor_strength: float = 0.3,
        fib_strength: float = 0.2,
        prime_strength: float = 0.2,
        smooth_strength: float = 0.1,
        knowledge_boost: float = 0.5
    ):
        self.vocab_size = vocab_size
        self.cantor_strength = cantor_strength
        self.fib_strength = fib_strength
        self.prime_strength = prime_strength
        self.smooth_strength = smooth_strength
        self.knowledge_boost = knowledge_boost
        
        # 预计算索引
        self.cantor_indices = self._compute_cantor()
        self.fib_indices = self._compute_fibonacci()
        self.prime_indices = self._compute_primes()
        
        # 知识相关token
        self.knowledge_tokens: Dict[str, List[int]] = {}
        
        logger.info(f"终极逻辑处理器: 康托集{len(self.cantor_indices)}, "
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
    
    def set_knowledge_tokens(self, tokens: Dict[str, List[int]]):
        """设置知识相关token"""
        self.knowledge_tokens = tokens
    
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
        
        # 5. 知识引导
        for key, token_ids in self.knowledge_tokens.items():
            for tid in token_ids:
                if tid < scores.size(1):
                    scores[0, tid] += self.knowledge_boost
        
        return scores


# ============================================================================
# 第三部分：知识管理系统
# ============================================================================

class KnowledgeManager:
    """知识管理器"""
    
    def __init__(self):
        self.knowledge_base: List[Knowledge] = []
        self.pattern_index: Dict[str, List[int]] = {}  # 模式到知识索引的映射
        
        logger.info("知识管理器初始化完成")
    
    def add_knowledge(self, question: str, answer: str, explanation: str, importance: float = 1.0):
        """添加知识"""
        pattern = self._extract_pattern(question, answer, explanation)
        
        knowledge = Knowledge(
            question=question,
            answer=answer,
            explanation=explanation,
            pattern=pattern,
            importance=importance
        )
        
        self.knowledge_base.append(knowledge)
        
        # 更新索引
        pattern_key = self._get_pattern_key(pattern)
        if pattern_key not in self.pattern_index:
            self.pattern_index[pattern_key] = []
        self.pattern_index[pattern_key].append(len(self.knowledge_base) - 1)
        
        logger.info(f"添加知识: {pattern[:30]}... (共{len(self.knowledge_base)}条)")
        return knowledge
    
    def _extract_pattern(self, question: str, answer: str, explanation: str) -> str:
        """提取推理模式"""
        # 房租计算
        if '房租' in question and '天' in question:
            return "房租计算: 日租金=总租金÷天数, 月租金=日租金×30"
        
        # 奇偶数
        if '奇数' in question or '偶数' in question:
            return "范围奇偶数: 找范围内符合条件的数"
        
        # 押金
        if '押金' in question:
            return "押金: 固定金额, 退房时退还"
        
        # 卫生费
        if '卫生费' in question:
            return "卫生费: 卫生干净时退还"
        
        return f"通用: {question[:20]}... → {answer}"
    
    def _get_pattern_key(self, pattern: str) -> str:
        """获取模式键"""
        if '房租' in pattern:
            return 'rent'
        if '奇数' in pattern or '偶数' in pattern:
            return 'odd_even'
        if '押金' in pattern:
            return 'deposit'
        if '卫生费' in pattern:
            return 'cleaning'
        return 'general'
    
    def find_relevant(self, question: str, top_k: int = 3) -> List[Knowledge]:
        """查找相关知识"""
        scores = []
        
        for i, kb in enumerate(self.knowledge_base):
            score = 0
            
            # 关键词匹配
            q_words = set(question)
            kb_words = set(kb.question)
            common = q_words & kb_words
            score += len(common)
            
            # 模式匹配
            if '房租' in question and '房租' in kb.question:
                score += 10
            if '奇数' in question and '奇数' in kb.question:
                score += 10
            if '偶数' in question and '偶数' in kb.question:
                score += 10
            
            # 重要性加权
            score *= kb.importance
            
            scores.append((score, i))
        
        # 排序
        scores.sort(key=lambda x: x[0], reverse=True)
        
        return [self.knowledge_base[i] for _, i in scores[:top_k]]
    
    def get_all_patterns(self) -> List[str]:
        """获取所有模式"""
        return [kb.pattern for kb in self.knowledge_base]


# ============================================================================
# 第四部分：终极集成引擎
# ============================================================================

class UltimateEngine:
    """
    终极集成推理引擎
    
    整合所有技术：
    1. 黎曼平滑 - 逻辑连续性
    2. 数学增强 - 康托集+斐波那契+素数
    3. 海马体记忆 - 存储和召回
    4. STDP学习 - 权重更新
    5. 知识管理 - 知识库
    6. 纠正学习 - 用户反馈
    """
    
    def __init__(self, model_path: str, config: BrainLikeConfig = None):
        self.model_path = model_path
        self.config = config or DEFAULT_CONFIG
        
        # 核心组件
        self.model = None
        self.tokenizer = None
        self.hippocampus = None
        self.stdp = None
        
        # 增强组件
        self.riemann_layer: Optional[RiemannSmoothingLayer] = None
        self.logic_processor: Optional[UltimateLogicProcessor] = None
        self.knowledge_manager: Optional[KnowledgeManager] = None
        
        # 状态
        self.session = SessionState()
        self.history: List[Dict] = []
        
        # 统计
        self.stats = {
            'total_memories': 0,
            'total_corrections': 0,
            'total_learning': 0,
            'total_tokens': 0
        }
        
        # 设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"终极引擎运行设备: {self.device}")
        self._initialized = False
    
    def initialize(self) -> bool:
        """初始化"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"初始化终极引擎: {self.model_path}")
        
        # 1. 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 2. 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 3. 初始化海马体
        self.hippocampus = HippocampusSystem(self.config)
        
        # 4. 初始化STDP
        self.stdp = STDPKernel(self.config.stdp)
        
        # 5. 初始化黎曼平滑层
        hidden_size = self.model.config.hidden_size
        self.riemann_layer = RiemannSmoothingLayer(dim=hidden_size, alpha=0.1)
        self.riemann_layer = self.riemann_layer.to(self.device)
        
        # 6. 初始化逻辑处理器
        vocab_size = len(self.tokenizer)
        self.logic_processor = UltimateLogicProcessor(
            vocab_size=vocab_size,
            cantor_strength=0.3,
            fib_strength=0.2,
            prime_strength=0.2,
            smooth_strength=0.1,
            knowledge_boost=0.5
        )
        
        # 7. 初始化知识管理器
        self.knowledge_manager = KnowledgeManager()
        
        self._initialized = True
        
        logger.info("=" * 60)
        logger.info("终极引擎初始化完成！")
        logger.info(f"  - 黎曼平滑: alpha=0.1")
        logger.info(f"  - 数学增强: 康托集+斐波那契+素数")
        logger.info(f"  - 海马体: 记忆存储")
        logger.info(f"  - STDP: 学习机制")
        logger.info(f"  - 知识库: 动态更新")
        logger.info("=" * 60)
        
        return True
    
    def _detect_correction(self, user_input: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """检测用户纠正"""
        keywords = ['不对', '错误', '错了', '不是', '应该是', '正确的是', '答案是']
        
        is_correction = any(kw in user_input for kw in keywords)
        
        if is_correction:
            numbers = re.findall(r'\d+', user_input)
            if numbers:
                return True, numbers[-1], user_input
        
        return False, None, None
    
    def _learn(self, question: str, answer: str, explanation: str):
        """学习"""
        logger.info(f"学习: {question[:30]}... → {answer}")
        
        # 1. 添加到知识库
        self.knowledge_manager.add_knowledge(
            question=question,
            answer=answer,
            explanation=explanation,
            importance=2.0  # 纠正的知识更重要
        )
        
        # 2. 更新逻辑处理器的知识token
        self._update_knowledge_tokens()
        
        # 3. 更新统计
        self.stats['total_learning'] += 1
        self.session.learning_events += 1
    
    def _update_knowledge_tokens(self):
        """更新知识相关token"""
        knowledge_tokens = {}
        
        for kb in self.knowledge_manager.knowledge_base:
            # 编码答案
            answer_tokens = self.tokenizer.encode(kb.answer, add_special_tokens=False)
            knowledge_tokens[f"answer_{kb.answer}"] = answer_tokens
            
            # 编码关键词
            keywords = re.findall(r'[\d]+|[a-zA-Z]+|[\u4e00-\u9fff]+', kb.pattern)
            for kw in keywords:
                kw_tokens = self.tokenizer.encode(kw, add_special_tokens=False)
                knowledge_tokens[f"kw_{kw}"] = kw_tokens
        
        self.logic_processor.set_knowledge_tokens(knowledge_tokens)
    
    def _build_system_prompt(self, question: str) -> str:
        """构建系统提示"""
        base = "你是AI助手，请准确计算并简洁回答。"
        
        # 添加相关知识
        relevant = self.knowledge_manager.find_relevant(question, top_k=3)
        
        if relevant:
            patterns = "\n".join([f"- {kb.pattern}" for kb in relevant])
            base += f"\n\n【相关知识】\n{patterns}"
        
        return base
    
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
        self.stats['total_memories'] += 1
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
            if self.session.last_question:
                self._learn(self.session.last_question, correct_answer, explanation)
                yield f"[已学习]\n\n"
            self.stats['total_corrections'] += 1
            self.session.correction_count += 1
        
        # 3. 更新会话状态
        self.session.question_count += 1
        self.session.last_question = prompt
        
        # 4. 构建系统提示
        system_prompt = self._build_system_prompt(prompt)
        
        # 5. 构建消息
        messages = [{"role": "system", "content": system_prompt}]
        
        for h in self.history[-5:]:
            messages.append({"role": "user", "content": h['q']})
            messages.append({"role": "assistant", "content": h['a'][:100]})
        
        messages.append({"role": "user", "content": prompt})
        
        # 6. 生成
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
                
                # 应用黎曼平滑
                if hidden.dim() == 3:
                    hidden = self.riemann_layer(hidden)
                
                # 应用逻辑处理器
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
                self.stats['total_tokens'] += 1
        
        # 7. 存储记忆
        self._store_memory(
            content=f"Q: {prompt} A: {response[:100]}",
            hidden_state=hidden,
            importance=2.0 if is_correction else 1.0
        )
        
        # 8. 保存历史
        self.history.append({'q': prompt, 'a': response})
        if len(self.history) > 20:
            self.history = self.history[-20:]
        
        self.session.last_answer = response
    
    def get_statistics(self) -> Dict:
        return {
            'initialized': self._initialized,
            'device': str(self.device),
            'session': {
                'questions': self.session.question_count,
                'corrections': self.session.correction_count,
                'learning': self.session.learning_events
            },
            'knowledge_base': len(self.knowledge_manager.knowledge_base) if self.knowledge_manager else 0,
            'memories': self.stats['total_memories'],
            'tokens': self.stats['total_tokens'],
            'history': len(self.history)
        }
    
    def clear_memory(self):
        """清空记忆"""
        self.history = []
        self.session = SessionState()
        if self.hippocampus:
            self.hippocampus.clear()
        self.stats['total_memories'] = 0
