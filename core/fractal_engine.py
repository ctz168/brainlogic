#!/usr/bin/env python3
"""
类人脑双系统AI架构 - 分形自相似增强引擎
Fractal Self-Similar Enhanced Engine

核心创新：使用康托集+斐波那契分形结构增强数学逻辑稠密度
"""

import os
import sys
import logging
import time
import re
import math
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from core.config import BrainLikeConfig, DEFAULT_CONFIG
from modules.hippocampus import HippocampusSystem
from modules.refresh_engine import RefreshEngine as ModuleRefreshEngine
from modules.stdp_system import STDPKernel as ModuleSTDPKernel


class FractalMathEnhancer:
    """
    分形自相似数学增强器
    
    原理：
    1. 康托集(Cantor Set)：产生自相似的离散结构
    2. 斐波那契数列：自然界中最常见的自相似模式
    3. 组合增强：提升数学推理的逻辑稠密度
    """
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        
        # 预计算康托集索引
        self.cantor_indices = self._compute_cantor_set(min(vocab_size, 100000))
        
        # 预计算斐波那契索引
        self.fib_indices = self._compute_fibonacci(vocab_size)
        
        logger.info(f"分形增强器初始化: 康托集{len(self.cantor_indices)}点, 斐波那契{len(self.fib_indices)}点")
    
    def _compute_cantor_set(self, max_n: int) -> List[int]:
        """计算康托集索引"""
        indices = []
        for i in range(max_n):
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
    
    def _compute_fibonacci(self, max_n: int) -> List[int]:
        """计算斐波那契索引"""
        indices = []
        a, b = 1, 1
        while b < max_n:
            indices.append(b)
            a, b = b, a + b
        return indices
    
    def enhance(self, logits: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
        """
        增强logits
        
        Args:
            logits: [batch, vocab_size]
            strength: 增强强度
            
        Returns:
            增强后的logits
        """
        logits = logits.clone()
        
        # 康托集增强（主要）
        for idx in self.cantor_indices:
            if idx < logits.shape[-1]:
                logits[0, idx] += strength
        
        # 斐波那契增强（辅助）
        for idx in self.fib_indices:
            if idx < logits.shape[-1]:
                logits[0, idx] += strength * 0.5
        
        return logits


class FractalEnhancedEngine:
    """分形增强引擎"""
    
    def __init__(self, model_path: str, config: BrainLikeConfig = None):
        self.model_path = model_path
        self.config = config or DEFAULT_CONFIG
        
        self.model = None
        self.tokenizer = None
        self.fractal_enhancer = None
        self.refresh_engine = None
        
        # 会话状态
        self.session: Dict[str, Any] = {}
        self.history: List[Dict] = []
        
        # 连续状态
        self._continuous_hidden: Optional[torch.Tensor] = None
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"引擎运行设备: {self.device}")
        self._initialized = False
    
    def initialize(self) -> bool:
        """初始化"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"初始化分形增强引擎: {self.model_path}")
        
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
        
        # 初始化分形增强器
        self.fractal_enhancer = FractalMathEnhancer(len(self.tokenizer))
        
        # 冻结权重
        self._freeze_weights()
        
        # 初始化子模块
        hippo = HippocampusSystem(self.config)
        stdp = ModuleSTDPKernel(self.config.stdp)
        self.refresh_engine = ModuleRefreshEngine(
            self.model, self.config, hippocampus_module=hippo, stdp_module=stdp
        )
        
        self._initialized = True
        logger.info("分形增强引擎初始化完成！")
        return True
    
    def _freeze_weights(self):
        """冻结权重"""
        all_params = list(self.model.named_parameters())
        freeze_count = int(len(all_params) * self.config.weight_split.static_ratio)
        
        for i, (name, param) in enumerate(all_params):
            if i < freeze_count:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"冻结权重: {freeze_count}/{len(all_params)} 层")
        logger.info(f"可训练参数: {trainable/1e6:.2f}M ({trainable/total*100:.1f}%)")
    
    def _extract_session_data(self, text: str):
        """提取会话数据"""
        m = re.search(r'(\d+)\s*天\s*房租\s*(\d+)', text)
        if m:
            self.session['天数'] = int(m.group(1))
            self.session['房租'] = int(m.group(2))
        
        m = re.search(r'押金[：:]*\s*(\d+)', text)
        if m:
            self.session['押金'] = int(m.group(1))
        
        m = re.search(r'卫生费[：:]*\s*(\d+)', text)
        if m:
            self.session['卫生费'] = int(m.group(1))
    
    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        """同步生成"""
        return "".join(list(self.generate_stream(prompt, max_new_tokens)))
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 400
    ) -> Generator[str, None, None]:
        """流式生成 - 带分形增强"""
        if not self._initialized:
            if not self.initialize():
                yield "初始化失败"
                return
        
        self._extract_session_data(prompt)
        
        # 构建消息
        messages = [
            {"role": "system", "content": "你是AI助手，请准确计算并回答。"}
        ]
        
        for h in self.history[-5:]:
            messages.append({"role": "user", "content": h['q']})
            messages.append({"role": "assistant", "content": h['a']})
        
        messages.append({"role": "user", "content": prompt})
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        
        generated = []
        past = None
        response = ""
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.model(
                    input_ids=input_ids if past is None else input_ids[:, -1:],
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True
                )
                
                logits = out.logits[:, -1, :].clone()
                hidden = out.hidden_states[-1][:, -1, :]
                past = out.past_key_values
                
                # 分形增强
                logits = self.fractal_enhancer.enhance(logits, strength=0.3)
                
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
        
        # 保存历史
        self.history.append({'q': prompt, 'a': response})
        if len(self.history) > 10:
            self.history = self.history[-10:]
        
        self._continuous_hidden = hidden.detach().clone()
    
    def get_statistics(self) -> Dict:
        return {
            'initialized': self._initialized,
            'device': str(self.device),
            'session': self.session,
            'history_count': len(self.history)
        }
    
    def clear_memory(self):
        self.session = {}
        self.history = []
        self._continuous_hidden = None
