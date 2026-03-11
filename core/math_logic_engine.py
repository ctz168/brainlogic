#!/usr/bin/env python3
"""
类人脑双系统AI架构 - 数学逻辑增强引擎 v4
适配 Qwen3.5-0.8B Instruct 模型
"""

import os
import sys
import logging
import time
import re
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field
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
from modules.stdp_system import STDPKernel as ModuleSTDPKernel, STDPType


@dataclass
class SessionState:
    """会话状态"""
    房租: Optional[int] = None
    押金: Optional[int] = None
    卫生费: Optional[int] = None
    天数: Optional[int] = None
    起租日: Optional[str] = None


class MathLogicEngine:
    """数学逻辑增强引擎 - 适配 Qwen3.5"""
    
    def __init__(self, model_path: str, config: BrainLikeConfig = None):
        self.model_path = model_path
        self.config = config or DEFAULT_CONFIG
        
        self.model = None
        self.tokenizer = None
        self.refresh_engine = None
        
        # 会话状态
        self.session_state = SessionState()
        self.conversation_history: List[Dict] = []
        
        # 连续隐藏状态
        self._continuous_hidden: Optional[torch.Tensor] = None
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"引擎运行设备: {self.device}")
        self._initialized = False
        self.stop_token_ids = []
    
    def initialize(self) -> bool:
        """初始化"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"初始化引擎: {self.model_path}")
        
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
        
        # 冻结权重
        self._freeze_weights()
        
        # 初始化子模块
        hippo = HippocampusSystem(self.config)
        stdp = ModuleSTDPKernel(self.config.stdp)
        self.refresh_engine = ModuleRefreshEngine(
            self.model, self.config, hippocampus_module=hippo, stdp_module=stdp
        )
        
        # 停止符
        self.stop_token_ids = [self.tokenizer.eos_token_id]
        
        self._initialized = True
        logger.info("引擎初始化完成！")
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
            self.session_state.天数 = int(m.group(1))
            self.session_state.房租 = int(m.group(2))
        
        m = re.search(r'押金[：:]*\s*(\d+)', text)
        if m:
            self.session_state.押金 = int(m.group(1))
        
        m = re.search(r'卫生费[：:]*\s*(\d+)', text)
        if m:
            self.session_state.卫生费 = int(m.group(1))
        
        m = re.search(r'(\d+)月(\d+)日', text)
        if m:
            self.session_state.起租日 = f"{m.group(1)}月{m.group(2)}日"
    
    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        """同步生成"""
        return "".join(list(self.generate_stream(prompt, max_new_tokens)))
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 400
    ) -> Generator[str, None, None]:
        """流式生成 - 使用 apply_chat_template"""
        if not self._initialized:
            if not self.initialize():
                yield "初始化失败"
                return
        
        self._extract_session_data(prompt)
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": "你是AI助手，请准确计算并详细回答问题。保持回答的连贯性。"}
        ]
        
        # 添加历史对话
        for h in self.conversation_history[-5:]:
            messages.append({"role": "user", "content": h['q']})
            messages.append({"role": "assistant", "content": h['a']})
        
        # 添加当前问题
        messages.append({"role": "user", "content": prompt})
        
        # 使用 apply_chat_template 生成正确格式
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # 使用 model.generate 进行生成（更稳定）
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # 保存历史
        self.conversation_history.append({'q': prompt, 'a': response})
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # 流式返回
        yield response
    
    def get_statistics(self) -> Dict:
        """获取统计"""
        return {
            'initialized': self._initialized,
            'device': str(self.device),
            'session_state': {
                '房租': self.session_state.房租,
                '押金': self.session_state.押金,
                '卫生费': self.session_state.卫生费,
                '天数': self.session_state.天数,
            },
            'history_count': len(self.conversation_history)
        }
    
    def clear_memory(self):
        """清空记忆"""
        self.session_state = SessionState()
        self.conversation_history = []
        self._continuous_hidden = None
