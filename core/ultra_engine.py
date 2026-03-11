#!/usr/bin/env python3
"""
类人脑双系统AI架构 - 极强增强引擎 v5
Ultra Enhanced Brain-Like Engine

核心优化：
1. 极强数学增强 - 康托集+斐波那契+数字token+计算token
2. 智能Prompt构建 - 带示例引导
3. 会话状态保持
"""

import os
import sys
import logging
import time
import re
import math
from typing import Dict, List, Optional, Any, Generator
from pathlib import Path

import torch

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from core.config import BrainLikeConfig, DEFAULT_CONFIG
from modules.hippocampus import HippocampusSystem
from modules.refresh_engine import RefreshEngine as ModuleRefreshEngine
from modules.stdp_system import STDPKernel as ModuleSTDPKernel


class UltraMathEnhancer:
    """极强数学增强器"""
    
    def __init__(self, tokenizer, vocab_size: int):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        
        # 预计算增强索引
        self.cantor = self._compute_cantor()
        self.fib = self._compute_fibonacci()
        self.digit_tokens = self._compute_digit_tokens()
        self.calc_tokens = self._compute_calc_tokens()
        
        logger.info(f"极强增强器: 康托集{len(self.cantor)}, 斐波那契{len(self.fib)}, "
                   f"数字{len(self.digit_tokens)}, 计算{len(self.calc_tokens)}")
    
    def _compute_cantor(self):
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
    
    def _compute_fibonacci(self):
        indices = []
        a, b = 1, 1
        while b < self.vocab_size:
            indices.append(b)
            a, b = b, a + b
        return indices
    
    def _compute_digit_tokens(self):
        tokens = set()
        for i in range(1000):
            ids = self.tokenizer.encode(str(i), add_special_tokens=False)
            tokens.update(ids)
        return list(tokens)
    
    def _compute_calc_tokens(self):
        words = ['元', '天', '月', '租', '算', '等', '于', '是', '除', '乘', 
                 '加', '减', '答', '解', '结果', '结论', '日', '每', '总']
        tokens = set()
        for w in words:
            ids = self.tokenizer.encode(w, add_special_tokens=False)
            tokens.update(ids)
        return list(tokens)
    
    def enhance(self, logits, strength=1.0):
        """极强增强"""
        logits = logits.clone()
        
        # 康托集
        for idx in self.cantor:
            if idx < logits.shape[-1]:
                logits[0, idx] += 0.5 * strength
        
        # 斐波那契
        for idx in self.fib:
            if idx < logits.shape[-1]:
                logits[0, idx] += 0.3 * strength
        
        # 数字token（最强）
        for idx in self.digit_tokens:
            if idx < logits.shape[-1]:
                logits[0, idx] += 2.0 * strength
        
        # 计算token
        for idx in self.calc_tokens:
            if idx < logits.shape[-1]:
                logits[0, idx] += 1.5 * strength
        
        # 稠密度放大
        first_deriv = torch.gradient(logits[0])[0]
        second_deriv = torch.gradient(first_deriv)[0]
        curvature = torch.abs(second_deriv)
        enhancement = curvature / (curvature.max() + 1e-8)
        logits[0] += enhancement * 0.5 * strength
        
        return logits


class UltraEnhancedEngine:
    """极强增强引擎"""
    
    def __init__(self, model_path: str, config: BrainLikeConfig = None):
        self.model_path = model_path
        self.config = config or DEFAULT_CONFIG
        
        self.model = None
        self.tokenizer = None
        self.enhancer = None
        self.refresh_engine = None
        
        # 会话状态
        self.session: Dict[str, Any] = {}
        self.history: List[Dict] = []
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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"初始化极强增强引擎: {self.model_path}")
        
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
        
        # 初始化增强器
        self.enhancer = UltraMathEnhancer(self.tokenizer, len(self.tokenizer))
        
        # 冻结权重
        self._freeze_weights()
        
        # 初始化子模块
        hippo = HippocampusSystem(self.config)
        stdp = ModuleSTDPKernel(self.config.stdp)
        self.refresh_engine = ModuleRefreshEngine(
            self.model, self.config, hippocampus_module=hippo, stdp_module=stdp
        )
        
        self._initialized = True
        logger.info("极强增强引擎初始化完成！")
        return True
    
    def _freeze_weights(self):
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
    
    def _build_smart_prompt(self, user_input: str) -> str:
        """智能构建prompt - 带示例引导"""
        self._extract_session_data(user_input)
        
        # 检测是否是计算问题
        is_calc = any(kw in user_input for kw in ['计算', '多少', '租', '元', '天', '月'])
        
        if is_calc:
            # 带示例的计算prompt
            system = """你是AI助手，请准确计算。

计算示例：
- 如果10天房租是1000元，则日租金=1000÷10=100元/天，月租金=100×30=3000元
- 如果房租800元租了10天，则日租金=800÷10=80元/天，月租金=80×30=2400元

请按此逻辑计算用户的问题。"""
        else:
            system = "你是AI助手，请准确回答问题。"
        
        # 构建消息
        messages = [{"role": "system", "content": system}]
        
        # 添加历史
        for h in self.history[-3:]:
            messages.append({"role": "user", "content": h['q']})
            messages.append({"role": "assistant", "content": h['a'][:150]})
        
        messages.append({"role": "user", "content": user_input})
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        return "".join(list(self.generate_stream(prompt, max_new_tokens)))
    
    def generate_stream(self, prompt: str, max_new_tokens: int = 400) -> Generator[str, None, None]:
        if not self._initialized:
            if not self.initialize():
                yield "初始化失败"
                return
        
        text = self._build_smart_prompt(prompt)
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
                
                # 极强增强
                logits = self.enhancer.enhance(logits, strength=1.0)
                
                # 低温度采样
                logits = logits / 0.4
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
