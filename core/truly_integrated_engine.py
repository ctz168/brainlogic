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
        
        # CPU 性能优化：根据物理核心数设置线程
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        torch.set_num_threads(max(1, cpu_count // 2)) 
        
        # 获取停止符
        self.stop_token_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id and im_end_id != self.tokenizer.unk_token_id:
            self.stop_token_ids.append(im_end_id)
            
        # 预存静态权重备份，用于融合缓存 (Fusion Cache)
        self._static_weights_backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._static_weights_backup[name] = param.data.detach().clone()
        
        # 初始化 STDP 累积缓冲区
        self._stdp_buffer_multiplier = 0.0
        self._stdp_update_step = 0
        self._fusion_interval = 10 # 每10个token融合一次权重
        
        self._initialized = True
        logger.info("引擎初始化完成！")
        logger.info(f"停止符清单: {self.stop_token_ids}")
        logger.info(f"刷新周期: {self.config.refresh.refresh_period_ms}ms (100Hz)")
        logger.info(f"STDP学习率: LTP={self.config.stdp.alpha}, LTD={self.config.stdp.beta}")
        logger.info(f"CPU线程设置: {torch.get_num_threads()}")
        
        return True
    
    def _freeze_weights(self):
        """冻结90%权重"""
        all_params = list(self.model.named_parameters())
        freeze_count = int(len(all_params) * self.config.weight_split.static_ratio)
        
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
        max_new_tokens: int = 512
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
        cue_feature = None
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
                    mem_texts = [a['semantic_pointer'] for a in anchors if a['semantic_pointer']]
                    if mem_texts:
                        # 仅保留最近的几个，模拟短期记忆
                        memory_context = "\n【海马体关联记忆(Context)】:\n" + "\n".join([f"- {t}" for t in mem_texts[-3:]])
        except Exception as e:
            logger.error(f"海马体召回失败: {e}")

        # 构建输入 (Instruct格式，包含记忆上下文和推理引导)
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 构建系统提示
        system_prompt = f"""你是AI助手。当前时间: {current_time_str}
请准确回答问题。遇到计算时：
1. 提取所有数字
2. 列出计算步骤
3. 给出答案"""
        
        if memory_context:
            system_prompt += f"\n\n{memory_context}"
        
        # 使用 Instruct 格式
        input_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
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
            
            # 【核心架构原生约束】：True O(1) 动态聚焦窄窗口注意力
            # 设计原理：
            # 1. 固定窗口大小 = O(1) 复杂度，无论输入多长
            # 2. 海马体记忆锚点替代被截断的远端上下文
            # 3. 窗口大小平衡：太小丢失信息，太大失去O(1)优势
            
            O1_WINDOW_SIZE = 128  # 固定窗口：16 tokens (约100ms上下文)
            MEMORY_ANCHORS_COUNT = 2  # 海马体记忆锚点数量
            
            # 动态获取海马体记忆锚点（替代远端上下文）
            memory_context_hints = []
            if self.refresh_engine.hippocampus is not None and generated_tokens > 0:
                try:
                    # 使用上一周期的 hidden_state 作为查询线索
                    if hasattr(self, '_prev_hidden_state') and self._prev_hidden_state is not None:
                        anchors = self.refresh_engine.hippocampus.recall_memories(
                            self._prev_hidden_state, top_k=MEMORY_ANCHORS_COUNT
                        )
                        # 提取语义指针作为上下文提示
                        for anchor in anchors:
                            if anchor.get('semantic_pointer'):
                                memory_context_hints.append(anchor['semantic_pointer'])
                except Exception as e:
                    logger.debug(f"海马体记忆召回: {e}")
            
            # 窄窗口 KV cache 截断 (实现 O(1) 复杂度)
            if past_key_values is not None:
                from transformers.cache_utils import Cache
                
                if isinstance(past_key_values, Cache):
                    seq_len = past_key_values.get_seq_length()
                    if seq_len > O1_WINDOW_SIZE:
                        # 截断到固定窗口大小
                        if hasattr(past_key_values, "crop"):
                            past_key_values.crop(O1_WINDOW_SIZE)
                        else:
                            for i in range(len(past_key_values.key_cache)):
                                past_key_values.key_cache[i] = past_key_values.key_cache[i][..., -O1_WINDOW_SIZE:, :]
                                past_key_values.value_cache[i] = past_key_values.value_cache[i][..., -O1_WINDOW_SIZE:, :]
                        
                        if attention_mask.shape[-1] > O1_WINDOW_SIZE + 1:
                            attention_mask = attention_mask[:, -(O1_WINDOW_SIZE+1):]
                
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
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
            
            logits = outputs.logits[:, -1, :].clone()  # Clone to allow inplace operations
            hidden_state = outputs.hidden_states[-1][:, -1, :]
            past_key_values = outputs.past_key_values
            
            # 保存当前 hidden_state 供下一周期海马体检索使用
            self._prev_hidden_state = hidden_state.detach().clone()
            
            # STDP学习 (带缓冲机制)
            self._stdp_learn_buffered(hidden_state)
            
            # 采样优化: Vectorized Repetition Penalty + Temperature + Top-P
            # 1. Vectorized Repetition Penalty (1.1)
            if generated_tokens > 0:
                # 使用 scatter_ 或直接索引进行矢量化
                unique_tokens = input_ids[0].unique()
                logits[0, unique_tokens] = torch.where(
                    logits[0, unique_tokens] > 0,
                    logits[0, unique_tokens] / 1.1,
                    logits[0, unique_tokens] * 1.1
                )
            
            # 2. Temperature (0.7 for better diversity)
            logits = logits / 0.7
            
            # 3. Top-P (0.9) - 保持矢量化计算
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
            
            # 检测重复token循环 (防止无限重复) - 放宽条件
            if len(response_text) > 100:
                # 检查最后30个字符是否连续重复3次
                last_30 = response_text[-30:]
                if len(response_text) > 90:
                    if response_text[-60:-30] == last_30 and response_text[-90:-60] == last_30:
                        logger.warning(f"检测到重复输出，强制终止")
                        break
                # 检测乱码（非中文字符比例极高）
                chinese_chars = sum(1 for c in response_text[-100:] if '\u4e00' <= c <= '\u9fff')
                if len(response_text[-100:]) > 0 and chinese_chars / len(response_text[-100:]) < 0.1:
                    logger.warning(f"检测到可能的乱码输出，强制终止")
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
        if response_text.strip() and cue_feature is not None:
            self.refresh_engine.hippocampus.encode_episode(
                cue_feature, # 动态语义嵌入
                time.time() * 1000,
                {'semantic_pointer': f"User: {prompt}\nAssistant: {response_text}"}
            )
    
    def _stdp_learn_buffered(self, hidden_state: torch.Tensor):
        """STDP在线学习 (增强版 - 基于激活模式的差异化更新)"""
        # 计算激活强度和分布特征
        activation = hidden_state.abs().mean().item()
        activation_std = hidden_state.std().item()
        activation_max = hidden_state.abs().max().item()
        
        # 根据激活模式调整学习强度
        # 高激活 + 低方差 = 确定性高的信号，增强学习
        # 低激活 + 高方差 = 不确定信号，减弱学习
        confidence = activation / (activation_std + 1e-6)
        
        # 模拟时序差 (基于激活强度动态调整)
        delta_t = 5.0 * (1.0 + 0.5 * min(activation, 1.0))
        
        # 计算单步更新强度 (增强版)
        update, update_type = self.refresh_engine.stdp.compute_update(delta_t, activation)
        
        # 根据置信度调整更新方向和强度
        if update_type == STDPType.LTP:
            # 高置信度时增强LTP
            multiplier = update * (1.0 + 0.5 * min(confidence, 2.0))
        else:
            # LTD保持稳定
            multiplier = -update
        
        # 累积到缓冲区
        self._stdp_buffer_multiplier += multiplier
        self._stdp_update_step += 1
        
        # 记录学习统计
        if not hasattr(self, '_stdp_stats'):
            self._stdp_stats = {'ltp_count': 0, 'ltd_count': 0, 'total_update': 0}
        if update_type == STDPType.LTP:
            self._stdp_stats['ltp_count'] += 1
        else:
            self._stdp_stats['ltd_count'] += 1
        self._stdp_stats['total_update'] += abs(multiplier)
        
        # 达到阈值或步数时，执行融合 (Weight Fusion)
        if self._stdp_update_step >= self._fusion_interval:
            self._fuse_dynamic_weights()
            self._stdp_buffer_multiplier = 0.0
            self._stdp_update_step = 0

    def _fuse_dynamic_weights(self):
        """核心优化：双权重融合缓存 (增强版 - 分层差异化更新)"""
        if self._stdp_buffer_multiplier == 0:
            return
        
        # 基础学习率
        base_learning_rate = 0.0001
        
        # 根据累积更新方向调整学习强度
        if self._stdp_buffer_multiplier > 0:
            # LTP主导 - 增强学习
            learning_rate = base_learning_rate * 1.5
        else:
            # LTD主导 - 保守学习
            learning_rate = base_learning_rate * 0.8
        
        total_multiplier = self._stdp_buffer_multiplier * learning_rate
        
        updated_count = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.refresh_engine.dynamic_weights:
                    # 分层学习率：注意力层更强，FFN层适中
                    layer_multiplier = total_multiplier
                    if 'attention' in name.lower() or 'attn' in name.lower():
                        layer_multiplier *= 1.2  # 注意力层增强
                    elif 'ffn' in name.lower() or 'mlp' in name.lower():
                        layer_multiplier *= 0.9  # FFN层适中
                    
                    # 1. 更新动态偏置向量 (Dynamic Weights)
                    self.refresh_engine.dynamic_weights[name].add_(layer_multiplier)
                    
                    # 2. 权重裁剪 (防止过度偏离)
                    max_deviation = 0.1  # 最大偏离阈值
                    self.refresh_engine.dynamic_weights[name].clamp_(-max_deviation, max_deviation)
                    
                    # 3. 融合到模型实体权重中 (Fusion Cache)
                    param.data.copy_(self._static_weights_backup[name] + self.refresh_engine.dynamic_weights[name])
                    updated_count += 1
        
        # 记录融合统计
        if hasattr(self, '_stdp_stats'):
            self._stdp_stats['fusion_count'] = self._stdp_stats.get('fusion_count', 0) + 1
            self._stdp_stats['last_fusion_params'] = updated_count

    def _stdp_learn(self, hidden_state: torch.Tensor):
        """过时的逐步骤学习，已由 _stdp_learn_buffered 替代"""
        self._stdp_learn_buffered(hidden_state)
    
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
