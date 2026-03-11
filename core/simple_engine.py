#!/usr/bin/env python3
"""
简化版连续流引擎 - 使用正确的 ChatML 格式
"""

import os
import sys
import logging
import time
from typing import Dict, List, Optional, Any, Generator
from pathlib import Path

import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

from core.config import BrainLikeConfig, DEFAULT_CONFIG
from modules.hippocampus import HippocampusSystem
from modules.refresh_engine import RefreshEngine as ModuleRefreshEngine
from modules.stdp_system import STDPKernel as ModuleSTDPKernel, STDPType


class SimpleContinuousEngine:
    """简化版连续流引擎"""
    
    def __init__(self, model_path: str, config: BrainLikeConfig = None):
        self.model_path = model_path
        self.config = config or DEFAULT_CONFIG
        self.model = None
        self.tokenizer = None
        self.refresh_engine = None
        
        # 连续状态
        self._continuous_hidden: Optional[torch.Tensor] = None
        self._conversation_history: List[Dict] = []
        self._session_context: Dict[str, Any] = {}
        
        self._initialized = False
        
    def initialize(self) -> bool:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"加载模型: {self.model_path}")
        
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
        self.model.eval()
        
        # 冻结大部分权重
        all_params = list(self.model.named_parameters())
        freeze_count = int(len(all_params) * 0.9)
        for i, (name, param) in enumerate(all_params):
            if i < freeze_count:
                param.requires_grad = False
        
        # 初始化子模块
        hippo = HippocampusSystem(self.config)
        stdp = ModuleSTDPKernel(self.config.stdp)
        self.refresh_engine = ModuleRefreshEngine(
            self.model, self.config, hippocampus_module=hippo, stdp_module=stdp
        )
        
        self._initialized = True
        logger.info("引擎初始化完成")
        return True
    
    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        return "".join(list(self.generate_stream(prompt, max_new_tokens)))
    
    def generate_stream(self, prompt: str, max_new_tokens: int = 200) -> Generator[str, None, None]:
        if not self._initialized:
            self.initialize()
        
        # 构建消息列表（包含历史）
        messages = [{"role": "system", "content": "你是AI助手，简洁准确回答问题。"}]
        
        # 添加历史对话
        for h in self._conversation_history[-3:]:
            messages.append({"role": "user", "content": h['q']})
            messages.append({"role": "assistant", "content": h['a'][:200]})
        
        # 添加当前问题
        messages.append({"role": "user", "content": prompt})
        
        # 使用 apply_chat_template 生成正确格式
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        response = ""
        generated = 0
        hidden = None
        
        with torch.inference_mode():
            past_kv = None
            while generated < max_new_tokens:
                outputs = self.model(
                    input_ids=input_ids if past_kv is None else input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True
                )
                
                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                hidden = outputs.hidden_states[-1][:, -1, :]
                
                # 采样
                logits = logits / 0.7
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                token_id = next_token.item()
                if token_id == self.tokenizer.eos_token_id:
                    break
                
                text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                
                # 乱码检测
                if len(response) > 30:
                    recent = response[-30:]
                    chinese = sum(1 for c in recent if '\u4e00' <= c <= '\u9fff')
                    if chinese / len(recent) < 0.15:
                        logger.warning("检测到乱码，终止")
                        break
                
                yield text
                response += text
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1))], dim=-1)
                generated += 1
        
        # 保存状态
        if hidden is not None:
            self._continuous_hidden = hidden.detach()
        self._conversation_history.append({'q': prompt, 'a': response})
        if len(self._conversation_history) > 10:
            self._conversation_history = self._conversation_history[-10:]
        
        # 海马体编码
        if response.strip() and hidden is not None:
            try:
                self.refresh_engine.hippocampus.encode_episode(
                    hidden, time.time() * 1000,
                    {'semantic_pointer': f"Q:{prompt} A:{response}"}
                )
            except:
                pass
    
    def get_statistics(self) -> Dict:
        return {
            'initialized': self._initialized,
            'history_count': len(self._conversation_history),
            'has_continuous_state': self._continuous_hidden is not None
        }
