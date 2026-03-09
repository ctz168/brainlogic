"""
类人脑双系统全闭环AI架构 - 基础模型模块
Human-Like Brain Dual-System Full-Loop AI Architecture - Base Model Module

实现基于Qwen3.5-0.8B的类脑架构核心模型
包含权重双轨拆分、原生接口适配、角色切换等核心功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math
import time

from .config import BrainLikeConfig, ModelMode, OptimizationMode


@dataclass
class TokenFeatures:
    """Token特征数据结构"""
    token_id: int
    hidden_state: torch.Tensor
    attention_weights: torch.Tensor
    timing_info: Dict[str, float]
    semantic_vector: torch.Tensor


@dataclass
class CycleOutput:
    """单周期输出数据结构"""
    token_id: int
    token_text: str
    features: TokenFeatures
    memory_anchors: List[Dict]
    stdp_updates: Dict[str, torch.Tensor]
    cycle_time_ms: float


class DynamicWeightBranch(nn.Module):
    """STDP动态增量权重分支"""
    
    def __init__(self, hidden_size: int, intermediate_size: int = None, 
                 init_std: float = 0.02):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or hidden_size * 4
        
        # 动态权重初始化（小权重随机正态分布）
        self.gate_weight = nn.Parameter(
            torch.randn(hidden_size, hidden_size) * init_std
        )
        self.gate_bias = nn.Parameter(torch.zeros(hidden_size))
        
        # FFN动态分支
        self.up_weight = nn.Parameter(
            torch.randn(hidden_size, self.intermediate_size) * init_std
        )
        self.down_weight = nn.Parameter(
            torch.randn(self.intermediate_size, hidden_size) * init_std
        )
        
        # STDP可塑性标记
        self._stdp_eligible = True
        self._last_update_time = 0.0
        
    def forward(self, x: torch.Tensor, static_output: torch.Tensor) -> torch.Tensor:
        """前向传播，将动态分支与静态分支融合"""
        # 门控机制
        gate = torch.sigmoid(F.linear(x, self.gate_weight, self.gate_bias))
        
        # 动态分支计算
        dynamic_hidden = F.linear(x, self.up_weight)
        dynamic_hidden = F.silu(dynamic_hidden)
        dynamic_output = F.linear(dynamic_hidden, self.down_weight)
        
        # 融合静态和动态输出
        output = static_output + gate * dynamic_output
        return output
    
    def get_stdp_weights(self) -> Dict[str, nn.Parameter]:
        """获取可进行STDP更新的权重"""
        return {
            'gate_weight': self.gate_weight,
            'gate_bias': self.gate_bias,
            'up_weight': self.up_weight,
            'down_weight': self.down_weight
        }


class AttentionWithDynamicBranch(nn.Module):
    """带动态分支的注意力层"""
    
    def __init__(self, config: BrainLikeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.model_hidden_size
        self.num_heads = config.model_num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # 静态基础分支（将加载预训练权重）
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # STDP动态增量分支
        self.dynamic_branch = DynamicWeightBranch(
            self.hidden_size,
            init_std=config.weight_split.dynamic_init_std
        )
        
        # 海马体门控接口
        self.hippocampus_gate = nn.Parameter(
            torch.zeros(self.num_heads, 1, 1)
        )
        
        # 特征输出缓存
        self._feature_cache: List[TokenFeatures] = []
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_anchors: Optional[List[Dict]] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[TokenFeatures]]:
        """
        前向传播
        
        Args:
            hidden_states: 输入隐藏状态 [batch, seq_len, hidden_size]
            attention_mask: 注意力掩码
            memory_anchors: 海马体记忆锚点
            position_ids: 位置编码
            
        Returns:
            output: 输出隐藏状态
            attention_weights: 注意力权重
            features: Token特征列表
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 计算Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用海马体记忆锚点门控
        if memory_anchors is not None and len(memory_anchors) > 0:
            gate_signal = self._apply_hippocampus_gate(memory_anchors)
            q = q * (1 + self.hippocampus_gate * gate_signal)
        
        # 窄窗口注意力计算（O(1)复杂度）
        # 仅计算当前token与最近2个token的注意力
        window_size = self.config.refresh.max_context_per_cycle + 1
        if seq_len > window_size:
            # 窄窗口模式
            k_window = k[:, :, -window_size:, :]
            v_window = v[:, :, -window_size:, :]
        else:
            k_window = k
            v_window = v
        
        # 注意力计算
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k_window.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用注意力到V
        attn_output = torch.matmul(attn_weights, v_window)
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # 静态分支输出
        static_output = self.o_proj(attn_output)
        
        # 动态分支融合
        output = self.dynamic_branch(hidden_states, static_output)
        
        # 提取特征用于海马体
        features = self._extract_features(hidden_states, attn_weights)
        
        return output, attn_weights, features
    
    def _apply_hippocampus_gate(self, memory_anchors: List[Dict]) -> torch.Tensor:
        """应用海马体门控信号"""
        # 将记忆锚点转换为门控信号
        gate_signal = torch.zeros(1, device=self.hippocampus_gate.device)
        for anchor in memory_anchors:
            if 'gate_vector' in anchor:
                gate_signal += anchor['gate_vector']
        return gate_signal
    
    def _extract_features(
        self, 
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> List[TokenFeatures]:
        """提取Token特征"""
        features = []
        seq_len = hidden_states.shape[1]
        
        for i in range(seq_len):
            feature = TokenFeatures(
                token_id=i,
                hidden_state=hidden_states[:, i, :].detach().clone(),
                attention_weights=attention_weights[:, :, i, :].detach().clone(),
                timing_info={'timestamp': time.time() * 1000},
                semantic_vector=F.normalize(hidden_states[:, i, :], dim=-1).detach().clone()
            )
            features.append(feature)
        
        return features
    
    def freeze_static_weights(self):
        """冻结静态基础权重"""
        for param in [self.q_proj.weight, self.k_proj.weight, 
                      self.v_proj.weight, self.o_proj.weight]:
            param.requires_grad = False
    
    def get_dynamic_weights(self) -> Dict[str, nn.Parameter]:
        """获取动态分支权重（用于STDP更新）"""
        return self.dynamic_branch.get_stdp_weights()


class FFNWithDynamicBranch(nn.Module):
    """带动态分支的前馈网络层"""
    
    def __init__(self, config: BrainLikeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.model_hidden_size
        self.intermediate_size = self.hidden_size * 4
        
        # 静态基础分支
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # STDP动态增量分支
        self.dynamic_branch = DynamicWeightBranch(
            self.hidden_size,
            self.intermediate_size,
            init_std=config.weight_split.dynamic_init_std
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 静态分支计算（SwiGLU激活）
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        static_output = self.down_proj(gate * up)
        
        # 动态分支融合
        output = self.dynamic_branch(x, static_output)
        return output
    
    def freeze_static_weights(self):
        """冻结静态基础权重"""
        for param in [self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight]:
            param.requires_grad = False
    
    def get_dynamic_weights(self) -> Dict[str, nn.Parameter]:
        """获取动态分支权重"""
        return self.dynamic_branch.get_stdp_weights()


class TransformerBlockWithDynamicBranch(nn.Module):
    """带动态分支的Transformer块"""
    
    def __init__(self, config: BrainLikeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        # 注意力层
        self.self_attn = AttentionWithDynamicBranch(config)
        
        # FFN层
        self.mlp = FFNWithDynamicBranch(config)
        
        # 层归一化
        self.input_layernorm = nn.LayerNorm(config.model_hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.model_hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_anchors: Optional[List[Dict]] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[TokenFeatures]]:
        """前向传播"""
        # 自注意力
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, attn_weights, features = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            memory_anchors=memory_anchors,
            position_ids=position_ids
        )
        hidden_states = residual + hidden_states
        
        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, attn_weights, features
    
    def freeze_static_weights(self):
        """冻结静态权重"""
        self.self_attn.freeze_static_weights()
        self.mlp.freeze_static_weights()
    
    def get_dynamic_weights(self) -> Dict[str, nn.Parameter]:
        """获取所有动态权重"""
        weights = {}
        weights.update({f'layer{self.layer_idx}.attn.{k}': v 
                       for k, v in self.self_attn.get_dynamic_weights().items()})
        weights.update({f'layer{self.layer_idx}.mlp.{k}': v 
                       for k, v in self.mlp.get_dynamic_weights().items()})
        return weights


class BrainLikeQwenModel(nn.Module):
    """
    类人脑双系统全闭环AI架构核心模型
    
    基于Qwen3.5-0.8B实现：
    - 90%静态基础权重 + 10%STDP动态增量权重双轨体系
    - 100Hz高刷新推理引擎
    - 海马体记忆系统接口
    - 自闭环优化系统接口
    """
    
    def __init__(self, config: BrainLikeConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入层
        self.embed_tokens = nn.Embedding(
            config.model_vocab_size, 
            config.model_hidden_size
        )
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlockWithDynamicBranch(config, i)
            for i in range(config.model_num_layers)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(config.model_hidden_size)
        
        # 输出层（lm_head）
        self.lm_head = nn.Linear(
            config.model_hidden_size,
            config.model_vocab_size,
            bias=False
        )
        
        # 动态输出层分支
        self.lm_head_dynamic = DynamicWeightBranch(
            config.model_hidden_size,
            config.model_vocab_size,
            init_std=config.weight_split.dynamic_init_std
        )
        
        # 角色适配提示词模板
        self.role_templates = {
            ModelMode.GENERATION: "你是一个智能助手，请根据用户输入生成准确、有帮助的回答。",
            ModelMode.VERIFICATION: "你是一个严谨的验证者，请仔细检查给定内容的逻辑正确性和事实准确性，指出任何错误或漏洞。",
            ModelMode.JUDGMENT: "你是一个公正的评判者，请从事实准确性、逻辑完整性、语义连贯性、指令遵循度四个维度对给定内容进行评分。"
        }
        
        # 当前模式
        self._current_mode = ModelMode.GENERATION
        
        # 特征缓存（用于海马体）
        self._feature_buffer: List[TokenFeatures] = []
        
        # STDP更新缓存
        self._stdp_update_buffer: Dict[str, torch.Tensor] = {}
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_anchors: Optional[List[Dict]] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[TokenFeatures], Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch, seq_len]
            attention_mask: 注意力掩码
            memory_anchors: 海马体记忆锚点
            position_ids: 位置编码
            
        Returns:
            logits: 输出logits
            features: Token特征列表
            dynamic_weights: 动态权重字典
        """
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 逐层处理
        all_features = []
        for layer in self.layers:
            hidden_states, attn_weights, features = layer(
                hidden_states,
                attention_mask=attention_mask,
                memory_anchors=memory_anchors,
                position_ids=position_ids
            )
            all_features.extend(features)
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 输出层
        static_logits = self.lm_head(hidden_states)
        logits = self.lm_head_dynamic(hidden_states, static_logits)
        
        # 收集动态权重
        dynamic_weights = self.get_all_dynamic_weights()
        
        return logits, all_features, dynamic_weights
    
    def set_mode(self, mode: ModelMode):
        """设置模型运行模式"""
        self._current_mode = mode
    
    def get_mode_prompt(self) -> str:
        """获取当前模式的提示词模板"""
        return self.role_templates[self._current_mode]
    
    def freeze_static_weights(self):
        """冻结所有静态基础权重"""
        # 冻结嵌入层
        self.embed_tokens.weight.requires_grad = False
        
        # 冻结各层静态权重
        for layer in self.layers:
            layer.freeze_static_weights()
        
        # 冻结输出层静态权重
        self.lm_head.weight.requires_grad = False
    
    def get_all_dynamic_weights(self) -> Dict[str, nn.Parameter]:
        """获取所有STDP动态权重"""
        weights = {}
        
        # 各层动态权重
        for layer in self.layers:
            weights.update(layer.get_dynamic_weights())
        
        # 输出层动态权重
        weights.update({
            f'lm_head.{k}': v 
            for k, v in self.lm_head_dynamic.get_stdp_weights().items()
        })
        
        return weights
    
    def get_static_weight_ratio(self) -> float:
        """计算静态权重占比"""
        total_params = sum(p.numel() for p in self.parameters())
        static_params = sum(
            p.numel() for p in self.parameters() 
            if not p.requires_grad
        )
        return static_params / total_params
    
    def get_dynamic_weight_ratio(self) -> float:
        """计算动态权重占比"""
        return 1.0 - self.get_static_weight_ratio()
    
    def estimate_memory_mb(self, quantized: bool = True) -> float:
        """估算模型内存占用"""
        total_params = sum(p.numel() for p in self.parameters())
        
        if quantized:
            # INT4量化：每个参数0.5字节
            bytes_per_param = 0.5
        else:
            # FP16：每个参数2字节
            bytes_per_param = 2
        
        memory_bytes = total_params * bytes_per_param
        memory_mb = memory_bytes / (1024 * 1024)
        
        return memory_mb
    
    def load_pretrained_weights(self, pretrained_path: str):
        """
        加载预训练权重到静态分支
        
        Args:
            pretrained_path: 预训练权重路径
        """
        from .weight_loader import load_qwen_weights_to_model
        missing_keys, unexpected_keys = load_qwen_weights_to_model(
            self, pretrained_path, freeze_static=True
        )
        return missing_keys, unexpected_keys
    
    def save_weights(self, save_path: str, save_static: bool = False):
        """
        保存模型权重
        
        Args:
            save_path: 保存路径
            save_static: 是否保存静态权重（默认只保存动态权重）
        """
        if save_static:
            torch.save(self.state_dict(), save_path)
        else:
            # 只保存动态权重
            dynamic_state = {
                k: v for k, v in self.state_dict().items()
                if 'dynamic' in k or not v.requires_grad
            }
            torch.save(dynamic_state, save_path)


class ModelInterfaces:
    """模型接口管理类"""
    
    def __init__(self, model: BrainLikeQwenModel):
        self.model = model
    
    def get_attention_features(self) -> List[TokenFeatures]:
        """获取注意力层特征输出接口"""
        return self.model._feature_buffer
    
    def set_hippocampus_gate(self, memory_anchors: List[Dict]):
        """设置海马体注意力门控接口"""
        # 将记忆锚点传递给各注意力层
        for layer in self.model.layers:
            layer.self_attn._memory_anchors = memory_anchors
    
    def switch_role(self, mode: ModelMode) -> str:
        """角色适配接口"""
        self.model.set_mode(mode)
        return self.model.get_mode_prompt()
    
    def get_stdp_weights(self) -> Dict[str, nn.Parameter]:
        """获取STDP可更新权重接口"""
        return self.model.get_all_dynamic_weights()
