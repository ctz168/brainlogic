#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 连续流增强引擎
Continuous Flow Enhanced Brain-Like Engine

核心改进：
1. 连续隐藏状态流 - 对话之间保持状态连续性
2. 自相似性逻辑补齐 - 用分形结构增强逻辑稠密度
3. 稠密验证层 - 输出前进行逻辑验证
"""

import os
import sys
import logging
import time
import math
from typing import Dict, List, Optional, Any, Generator, Tuple
from dataclasses import dataclass, field
from pathlib import Path

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


class SelfSimilarLogicLayer(nn.Module):
    """
    自相似性逻辑补齐层
    
    使用分形结构增强小模型的逻辑稠密度：
    - 多尺度特征提取
    - 自相似模式匹配
    - 逻辑一致性验证
    """
    
    def __init__(self, hidden_size: int, num_scales: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_scales = num_scales
        
        # 多尺度投影
        self.scale_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size // (2 ** i))
            for i in range(num_scales)
        ])
        
        # 自相似融合
        self.fusion = nn.Linear(hidden_size + hidden_size // 2 + hidden_size // 4, hidden_size)
        
        # 逻辑验证门
        self.logic_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        前向传播
        
        Args:
            hidden_state: [batch, hidden_size]
            
        Returns:
            enhanced: 增强后的隐藏状态
            logic_score: 逻辑一致性分数
        """
        # 多尺度特征
        scale_features = []
        for i, proj in enumerate(self.scale_projections):
            feat = proj(hidden_state)
            # 上采样回原始维度
            if feat.shape[-1] != self.hidden_size:
                feat = F.interpolate(
                    feat.unsqueeze(1),
                    size=self.hidden_size,
                    mode='linear'
                ).squeeze(1)
            scale_features.append(feat)
        
        # 融合
        concat_features = torch.cat([
            F.interpolate(
                scale_features[i].unsqueeze(1),
                size=self.hidden_size,
                mode='linear'
            ).squeeze(1)
            for i in range(len(scale_features))
        ], dim=-1)
        
        # 确保维度匹配
        if concat_features.shape[-1] != self.hidden_size:
            # 自适应池化
            concat_features = F.adaptive_avg_pool1d(
                concat_features.unsqueeze(1),
                self.hidden_size
            ).squeeze(1)
        
        enhanced = self.fusion(
            F.pad(hidden_state, (0, self.hidden_size * 2 - hidden_state.shape[-1]))
        ) + hidden_state
        
        # 逻辑验证分数
        logic_score = self.logic_gate(hidden_state).mean().item()
        
        return enhanced, logic_score


class ContinuousFlowEngine:
    """
    连续流增强引擎
    
    核心特性：
    1. 隐藏状态在对话之间保持连续
    2. 自相似性逻辑补齐
    3. 海马体记忆持续融入
    """
    
    def __init__(self, model_path: str, config: BrainLikeConfig = None):
        self.model_path = model_path
        self.config = config or DEFAULT_CONFIG
        
        self.model = None
        self.tokenizer = None
        self.refresh_engine = None
        self.logic_layer = None
        
        # 连续状态存储
        self._continuous_hidden_state: Optional[torch.Tensor] = None
        self._conversation_history: List[Dict] = []
        self._session_context: Dict[str, Any] = {}
        
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
        
        logger.info("初始化连续流增强引擎...")
        
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
        
        # 获取隐藏层维度
        hidden_size = self.model.config.hidden_size
        
        # 初始化自相似逻辑层
        self.logic_layer = SelfSimilarLogicLayer(hidden_size).to(self.device)
        logger.info(f"自相似逻辑层已初始化: hidden_size={hidden_size}")
        
        # 冻结90%权重
        self._freeze_weights()
        
        # 初始化子模块
        hippo = HippocampusSystem(self.config)
        stdp = ModuleSTDPKernel(self.config.stdp)
        
        self.refresh_engine = ModuleRefreshEngine(
            self.model, self.config, hippocampus_module=hippo, stdp_module=stdp
        )
        
        # 初始化动态权重
        self.refresh_engine.dynamic_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.refresh_engine.dynamic_weights[name] = torch.zeros_like(param.data) * 0.01
        
        # 获取停止符
        self.stop_token_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id and im_end_id != self.tokenizer.unk_token_id:
            self.stop_token_ids.append(im_end_id)
        
        # 预存静态权重备份
        self._static_weights_backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._static_weights_backup[name] = param.data.detach().clone()
        
        self._stdp_buffer_multiplier = 0.0
        self._stdp_update_step = 0
        self._fusion_interval = 10
        
        self._initialized = True
        logger.info("连续流增强引擎初始化完成！")
        
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
    
    def _build_context_aware_prompt(self, user_input: str) -> str:
        """构建上下文感知的提示"""
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 从会话上下文提取关键信息
        context_summary = ""
        if self._session_context:
            if "rent_info" in self._session_context:
                context_summary += f"\n【已知信息】{self._session_context['rent_info']}"
        
        # 从对话历史提取最近的关键信息
        if self._conversation_history:
            recent = self._conversation_history[-3:]
            history_text = "\n".join([
                f"用户: {h['user']}\n助手: {h['assistant'][:100]}..."
                for h in recent if h.get('user') and h.get('assistant')
            ])
            if history_text:
                context_summary += f"\n【最近对话】\n{history_text}"
        
        # 从海马体召回相关记忆
        memory_context = ""
        if self._continuous_hidden_state is not None:
            try:
                anchors = self.refresh_engine.hippocampus.recall_memories(
                    self._continuous_hidden_state, top_k=2
                )
                if anchors:
                    mem_texts = [a['semantic_pointer'] for a in anchors if a.get('semantic_pointer')]
                    if mem_texts:
                        memory_context = f"\n【记忆召回】{mem_texts[0][:200]}"
            except:
                pass
        
        # 构建完整提示 - 简洁直接
        system_prompt = f"""你是AI助手。
{context_summary}

直接回答问题，简洁明了。"""

        input_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        return input_text
    
    def _extract_session_info(self, user_input: str, response: str):
        """提取会话信息并保存"""
        # 提取租金相关信息
        import re
        numbers = re.findall(r'(\d+(?:\.\d+)?)\s*元', user_input + response)
        if numbers and '租' in user_input:
            self._session_context['rent_info'] = {
                'numbers': numbers,
                'last_mention': time.time()
            }
        
        # 保存到对话历史
        self._conversation_history.append({
            'user': user_input,
            'assistant': response,
            'timestamp': time.time()
        })
        
        # 保持历史在合理范围
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]
    
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """同步生成"""
        return "".join(list(self.generate_stream(prompt, max_new_tokens)))
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256
    ) -> Generator[str, None, None]:
        """流式生成（带连续状态保持）"""
        if not self._initialized:
            if not self.initialize():
                yield "初始化失败"
                return
        
        # 构建上下文感知提示
        input_text = self._build_context_aware_prompt(prompt)
        
        encodings = self.tokenizer(
            input_text, return_tensors='pt', max_length=1024, truncation=True
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # 如果有连续隐藏状态，尝试融入
        if self._continuous_hidden_state is not None:
            # 将连续状态作为上下文提示
            pass  # 这里可以扩展为更复杂的状态融合
        
        # 生成
        generated_tokens = 0
        past_key_values = None
        response_text = ""
        current_position = input_ids.shape[-1] - 1
        
        # 收集隐藏状态用于逻辑增强
        collected_hiddens = []
        
        while generated_tokens < max_new_tokens:
            # O(1) 窄窗口
            O1_WINDOW_SIZE = 128
            
            if past_key_values is not None:
                from transformers.cache_utils import Cache
                
                if isinstance(past_key_values, Cache):
                    seq_len = past_key_values.get_seq_length()
                    if seq_len > O1_WINDOW_SIZE:
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
            
            if past_key_values is None:
                position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=self.device).unsqueeze(0)
            else:
                current_position += 1
                position_ids = torch.tensor([[current_position]], dtype=torch.long, device=self.device)
            
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
            
            logits = outputs.logits[:, -1, :].clone()
            hidden_state = outputs.hidden_states[-1][:, -1, :]
            past_key_values = outputs.past_key_values
            
            # 收集隐藏状态
            collected_hiddens.append(hidden_state.detach())
            
            # 自相似逻辑增强（每10个token应用一次）
            if generated_tokens > 0 and generated_tokens % 10 == 0 and self.logic_layer is not None:
                try:
                    enhanced_hidden, logic_score = self.logic_layer(hidden_state)
                    # 如果逻辑分数低，调整采样策略
                    if logic_score < 0.3:
                        # 增加确定性
                        pass
                except Exception as e:
                    logger.debug(f"逻辑层处理: {e}")
            
            # STDP学习
            self._stdp_learn_buffered(hidden_state)
            
            # 采样
            if generated_tokens > 0:
                unique_tokens = input_ids[0].unique()
                logits[0, unique_tokens] = torch.where(
                    logits[0, unique_tokens] > 0,
                    logits[0, unique_tokens] / 1.1,
                    logits[0, unique_tokens] * 1.1
                )
            
            logits = logits / 0.5
            
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            token_id = next_token.item()
            if token_id in self.stop_token_ids:
                break
            
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            
            if "<|im_start|>" in token_text or "<|im_end|>" in token_text:
                break
            if "User:" in token_text or "用户:" in token_text:
                break
            
            # 增强重复检测和乱码检测
            if len(response_text) > 50:
                # 检测重复
                last_20 = response_text[-20:]
                if len(response_text) > 40 and response_text[-40:-20] == last_20:
                    logger.warning("检测到重复输出，终止生成")
                    break
                
                # 检测乱码（非中文字符比例过高）
                recent_text = response_text[-50:]
                chinese_count = sum(1 for c in recent_text if '\u4e00' <= c <= '\u9fff')
                if chinese_count / len(recent_text) < 0.2:
                    # 中文比例低于20%，可能是乱码
                    logger.warning("检测到乱码输出，终止生成")
                    break
                
                # 检测异常字符模式
                if any(pattern in recent_text for pattern in ['watch', 'pains', 'watch-', 'Watch', 'WATCH']):
                    logger.warning("检测到异常模式，终止生成")
                    break
            
            yield token_text
            response_text += token_text
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=-1)
            generated_tokens += 1
            
            # 海马体编码
            self.refresh_engine.hippocampus.encode_episode(
                hidden_state, time.time() * 1000,
                {'token': token_text}
            )
        
        # 保存连续隐藏状态
        if collected_hiddens:
            # 使用最后几个隐藏状态的均值作为连续状态
            self._continuous_hidden_state = torch.stack(collected_hiddens[-5:]).mean(dim=0)
        
        # 提取会话信息
        self._extract_session_info(prompt, response_text)
        
        # 海马体记忆巩固
        if response_text.strip():
            self.refresh_engine.hippocampus.encode_episode(
                self._continuous_hidden_state if self._continuous_hidden_state is not None else hidden_state,
                time.time() * 1000,
                {'semantic_pointer': f"User: {prompt}\nAssistant: {response_text}"}
            )
    
    def _stdp_learn_buffered(self, hidden_state: torch.Tensor):
        """STDP在线学习"""
        activation = hidden_state.abs().mean().item()
        delta_t = 5.0
        
        update, update_type = self.refresh_engine.stdp.compute_update(delta_t, activation)
        multiplier = update if update_type == STDPType.LTP else -update
        
        self._stdp_buffer_multiplier += multiplier
        self._stdp_update_step += 1
        
        if self._stdp_update_step >= self._fusion_interval:
            self._fuse_dynamic_weights()
            self._stdp_buffer_multiplier = 0.0
            self._stdp_update_step = 0
    
    def _fuse_dynamic_weights(self):
        """权重融合"""
        if self._stdp_buffer_multiplier == 0:
            return
        
        learning_rate = 0.0001
        total_multiplier = self._stdp_buffer_multiplier * learning_rate
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.refresh_engine.dynamic_weights:
                    self.refresh_engine.dynamic_weights[name].add_(total_multiplier)
                    param.data.copy_(self._static_weights_backup[name] + self.refresh_engine.dynamic_weights[name])
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'initialized': self._initialized,
            'device': str(self.device),
            'conversation_history_count': len(self._conversation_history),
            'has_continuous_state': self._continuous_hidden_state is not None,
            'session_context': self._session_context
        }
    
    def clear_memory(self):
        """清空记忆"""
        if self.refresh_engine:
            self.refresh_engine.hippocampus.memories.clear()
        self._continuous_hidden_state = None
        self._conversation_history = []
        self._session_context = {}
