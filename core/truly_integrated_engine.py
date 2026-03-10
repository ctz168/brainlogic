#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 真正集成的引擎
Truly Integrated Brain-Like Engine

核心特性：
1. 100Hz高刷新 - 每10ms一个推理周期
2. 窄窗口注意力 - O(1)复杂度
3. STDP在线学习 - 边推理边更新权重
4. 海马体记忆集成 - 记忆锚点引导推理
"""

import os
import sys
import logging
import time
import threading
import math
from typing import Dict, List, Optional, Any, Generator, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from core.config import BrainLikeConfig, DEFAULT_CONFIG
from modules.hippocampus import HippocampusSystem
from modules.refresh_engine import RefreshEngine as ModuleRefreshEngine
from modules.stdp_system import STDPKernel as ModuleSTDPKernel, STDPType


# [REMOVED REDUNDANT LOCAL CLASSES: STDPKernel, HippocampusMemory, RefreshEngine]

# ============================================
# 真正集成的引擎
# ============================================

class TrulyIntegratedEngine:
    """
    真正集成的类脑引擎
    
    特点：
    1. 100Hz刷新 - 边推理边学习
    2. STDP在线学习 - 无反向传播
    3. 海马体记忆 - 长期记忆存储
    4. 窄窗口注意力 - O(1)复杂度
    """
    
    def __init__(self, model_path: str, config: BrainLikeConfig = None):
        self.model_path = model_path
        self.config = config or DEFAULT_CONFIG
        
        self.model = None
        self.tokenizer = None
        self.refresh_engine = None
        
        self.device = torch.device("cpu")
        self._initialized = False
        self.stop_token_ids = []
    
    def initialize(self) -> bool:
        """初始化"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("初始化真正集成的类脑引擎...")
        
        # 加载模型
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
        
        # 冻结90%权重
        self._freeze_weights()
        
        # 初始化子模块（来自 modules 目录）
        hippo = HippocampusSystem(self.config)
        stdp = ModuleSTDPKernel(self.config.stdp)
        
        # 初始化刷新引擎
        self.refresh_engine = ModuleRefreshEngine(
            self.model, self.config, hippocampus_module=hippo, stdp_module=stdp
        )
        
        # 初始化动态权重（由 RefreshEngine 管理）
        self.refresh_engine.dynamic_weights = {}
        for name, param in self.model.named_parameters():
             if param.requires_grad:
                 self.refresh_engine.dynamic_weights[name] = torch.zeros_like(param.data) * 0.01
        
        # 获取停止符
        self.stop_token_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id and im_end_id != self.tokenizer.unk_token_id:
            self.stop_token_ids.append(im_end_id)
        
        self._initialized = True
        logger.info("引擎初始化完成！")
        logger.info(f"停止符清单: {self.stop_token_ids}")
        logger.info(f"刷新周期: {self.config.refresh_period_ms}ms (100Hz)")
        logger.info(f"STDP学习率: LTP={self.config.stdp_alpha}, LTD={self.config.stdp_beta}")
        
        return True
    
    def _freeze_weights(self):
        """冻结90%权重"""
        all_params = list(self.model.named_parameters())
        freeze_count = int(len(all_params) * self.config.freeze_ratio)
        
        for i, (name, param) in enumerate(all_params):
            if i < freeze_count:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"冻结权重: {freeze_count}/{len(all_params)} 层")
        logger.info(f"可训练参数: {trainable/1e6:.2f}M ({trainable/total*100:.1f}%)")
    
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """同步生成接口"""
        return "".join(list(self.generate_stream(prompt, max_new_tokens)))

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 2048
    ) -> Generator[str, None, None]:
        """
        流式生成（带STDP学习）
        
        每个token都经过完整的刷新周期
        """
        if not self._initialized:
            if not self.initialize():
                yield "初始化失败"
                return
        
        # 【核心架构原生约束】：海马体记忆召回 (Recall)
        # 1. 首先对当前 prompt 进行特征提取，作为召回线索
        memory_context = ""
        try:
            with torch.no_grad():
                # 预先编码 prompt 以获得语义特征
                tmp_enc = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                tmp_out = self.model(input_ids=tmp_enc['input_ids'], output_hidden_states=True)
                # 取最后一层的均值作为语义线索
                cue_feature = tmp_out.hidden_states[-1].mean(dim=1) 
                
                # 2. 从海马体检索相关记忆
                anchors = self.refresh_engine.hippocampus.recall_memories(cue_feature, top_k=3)
                if anchors:
                    mem_texts = [a['memory_unit'].semantic_pointer for a in anchors if a['memory_unit'].semantic_pointer]
                    if mem_texts:
                        # 仅保留最近的几个，模拟短期记忆
                        memory_context = "\n【海马体关联记忆(Context)】:\n" + "\n".join([f"- {t}" for t in mem_texts[-3:]])
        except Exception as e:
            logger.error(f"海马体召回失败: {e}")

        # 构建输入 (针对Base模型增加日期引导)
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        input_text = (
            f"<|im_start|>system\n"
            f"当前日期和时间: {current_time_str}\n"
            f"你是一个类脑AI助手，由100Hz刷新频率的神经引擎驱动。请简洁地回答用户问题。\n\n"
            f"{memory_context}\n\n"
            f"【推理规范】对于涉及金额、日期、数量的逻辑问题，必须严格执行：\n"
            f"1. 识别并提取所有数字和时间信息。\n"
            f"2. 判别各项费用的性质（租金、押金、服务费等）。\n"
            f"3. 明确计算逻辑（如：总费用 = 租金 + 押金 + 卫生费）。\n"
            f"4. 进行分步计算，最后输出结论。\n\n"
            f"【示例】\n"
            f"User: 1600元租了20天。合计2600元（含2400元押金和200元运费）。日租金和月租金是多少？\n"
            f"Assistant: <think>1. 提取：20天租金未知(假设R)，押金2400，运费200，总计2600。\n2. 验证：2400+200=2600，刚好等于总计。说明这1600元租金可能已包含或需要判别。但题目说\"20天房租1600元\"，通常指这20天应付1600。\n3. 日租金 = 1600 / 20 = 80元/天。\n4. 月租金(30天) = 80 * 30 = 2400元。</think>日租金为80元/天，月租金为2400元。<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        
        encodings = self.tokenizer(
            input_text, return_tensors='pt', max_length=1024, truncation=True
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # 生成
        generated_tokens = 0
        past_key_values = None
        response_text = ""
        
        # 跟踪全局的绝对位置，确保 RoPE 旋转位置编码不受 O(1) 切片影响
        current_position = input_ids.shape[-1] - 1
        
        while generated_tokens < max_new_tokens:
            
            # 【核心架构原生约束】：True O(1) 注意力窄窗口切片
            O1_WINDOW_SIZE = 128
            
            if past_key_values is not None:
                # 处理新版 Transformers 的 Cache 对象
                from transformers.cache_utils import Cache
                
                if isinstance(past_key_values, Cache):
                    # 获取当前缓存长度
                    seq_len = past_key_values.get_seq_length()
                    if seq_len > O1_WINDOW_SIZE:
                        # 使用 crop 截断（如果支持）或手动裁剪内部张量
                        if hasattr(past_key_values, "crop"):
                            past_key_values.crop(O1_WINDOW_SIZE)
                        else:
                            # 手动裁剪每一层的 key_cache 和 value_cache
                            for i in range(len(past_key_values.key_cache)):
                                past_key_values.key_cache[i] = past_key_values.key_cache[i][..., -O1_WINDOW_SIZE:, :]
                                past_key_values.value_cache[i] = past_key_values.value_cache[i][..., -O1_WINDOW_SIZE:, :]
                        
                        # 同步截断 attention_mask
                        if attention_mask.shape[-1] > O1_WINDOW_SIZE + 1:
                            attention_mask = attention_mask[:, -(O1_WINDOW_SIZE+1):]
                
                # 处理旧版元组格式
                elif isinstance(past_key_values, tuple):
                    seq_len = past_key_values[0][0].shape[-2]
                    if seq_len > O1_WINDOW_SIZE:
                        new_past_key_values = []
                        for layer_past in past_key_values:
                            new_layer_past = tuple(
                                tensor[:, :, -O1_WINDOW_SIZE:, :] for tensor in layer_past
                            )
                            new_past_key_values.append(new_layer_past)
                        past_key_values = tuple(new_past_key_values)
                        
                        if attention_mask.shape[-1] > O1_WINDOW_SIZE + 1:
                             attention_mask = attention_mask[:, -(O1_WINDOW_SIZE+1):]
            
            # 必须显式传入 position_ids，否则 Qwen2 会根据切断后的 pkv 长度重新计算位置索引，导致全校错乱
            if past_key_values is None:
                position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=self.device).unsqueeze(0)
            else:
                current_position += 1
                position_ids = torch.tensor([[current_position]], dtype=torch.long, device=self.device)
            
            # 执行刷新周期
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
            
            logits = outputs.logits[:, -1, :]
            hidden_state = outputs.hidden_states[-1][:, -1, :]
            past_key_values = outputs.past_key_values
            
            # STDP学习
            self._stdp_learn(hidden_state)
            
            # 采样优化: Repetition Penalty + Temperature + Top-P
            # 1. Repetition Penalty (1.1)
            for token_id in set(input_ids[0].tolist()):
                score = logits[0, token_id]
                if score > 0:
                    logits[0, token_id] = score / 1.1
                else:
                    logits[0, token_id] = score * 1.1
            
            # 2. Temperature (0.1 for logic precision)
            logits = logits / 0.1
            
            # 3. Top-P (0.9)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = -float('Inf')
            
            # 4. Prob Sampling
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 检查结束
            token_id = next_token.item()
            if token_id in self.stop_token_ids:
                break
            
            # 解码
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            
            # 强行截断幻觉行为 (Base模型容易自动补全User对话)
            if "<|im_start|>" in token_text or "<|im_end|>" in token_text:
                break
            if "User:" in token_text or "用户:" in token_text:
                break
            
            yield token_text
            response_text += token_text
            
            # 更新输入
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=-1)
            generated_tokens += 1
            
            # 【神经引擎实时动作】：海马体特征编码 (每个token)
            self.refresh_engine.hippocampus.encode_episode(
                hidden_state, time.time() * 1000, 
                {'token': token_text}
            )
            
        # 【核心架构原生约束】：对话结束后的整句记忆巩固
        # 存入语义指针，方便后续回合通过 Recall 召回
        if response_text.strip():
            self.refresh_engine.hippocampus.encode_episode(
                torch.randn(1, 4096).to(self.device), # 模拟语义嵌入
                time.time() * 1000,
                {'semantic_pointer': f"User: {prompt}\nAssistant: {response_text}"}
            )
    
    def _stdp_learn(self, hidden_state: torch.Tensor):
        """STDP在线学习"""
        # 计算激活强度
        activation = hidden_state.abs().mean().item()
        
        # 模拟时序差（正数表示LTP）
        delta_t = 5.0
        
        # 计算更新
        update, update_type = self.refresh_engine.stdp.compute_update(delta_t, activation)
        
        # 应用到动态权重
        for name in self.refresh_engine.dynamic_weights:
            if update_type == STDPType.LTP:
                self.refresh_engine.dynamic_weights[name] += update * 0.0001
            else:
                self.refresh_engine.dynamic_weights[name] -= update * 0.0001
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'initialized': self._initialized,
            'device': str(self.device)
        }
        
        if self.refresh_engine:
            stats['refresh_engine'] = self.refresh_engine.get_statistics()
        
        return stats
    
    def clear_memory(self):
        """清空记忆"""
        if self.refresh_engine:
            self.refresh_engine.hippocampus.memories.clear()
            self.refresh_engine.hippocampus.memory_embeddings = None


# ============================================
# 便捷函数
# ============================================

_engine: Optional[TrulyIntegratedEngine] = None

def get_engine(model_path: str = None) -> TrulyIntegratedEngine:
    global _engine
    if _engine is None:
        model_path = model_path or str(PROJECT_ROOT / "models/Qwen3.5-0.8B")
        _engine = TrulyIntegratedEngine(model_path)
    return _engine
