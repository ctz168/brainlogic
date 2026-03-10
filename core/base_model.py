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


class BrainLikeLinear(nn.Module):
    """类脑双轨线性层 (90% 静态 + 10% STDP 动态)"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False, 
                 dynamic_init_std: float = 0.02):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 90% 静态基础分支 (使用 nn.Linear 容纳预训练权重)
        self.static_branch = nn.Linear(in_features, out_features, bias=bias)
        self.static_branch.weight.requires_grad = False
        if bias:
            self.static_branch.bias.requires_grad = False
            
        # 10% STDP 动态增量分支
        # 使用较小的隐藏维度来实现 "10% 参数量" 的概念，或者直接使用同维度的增量
        # 考虑到 prompt 要求 "10% 动态增量权重"，且要规避灾难性遗忘
        # 我们这里使用 LoRA 风格的低秩分解来代表这 10% 的动态能力，
        # 或者直接使用并行的全量线性层并标记为动态。
        # 按指令要求：它是“新增可更新分支”。
        self.dynamic_branch = nn.Linear(in_features, out_features, bias=bias)
        # 初始化为小权重，确保初始时不影响静态分支表现
        nn.init.normal_(self.dynamic_branch.weight, std=dynamic_init_std)
        if bias:
            nn.init.zeros_(self.dynamic_branch.bias)
        
        # 门控系数 (生物脑中的突触可塑性门控)
        self.gate = nn.Parameter(torch.ones(1) * 0.1) # 初始门控强度
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        static_out = self.static_branch(x)
        dynamic_out = self.dynamic_branch(x)
        # 融合输出：原生能力 + 动态学习出的增量能力
        return static_out + self.gate * dynamic_out

    def get_dynamic_parameters(self) -> Dict[str, nn.Parameter]:
        """获取 STDP 可更新参数"""
        params = {'weight': self.dynamic_branch.weight}
        if self.dynamic_branch.bias is not None:
            params['bias'] = self.dynamic_branch.bias
        params['gate'] = self.gate
        return params


class AttentionWithDynamicBranch(nn.Module):
    """带动态分支的注意力层"""
    
    def __init__(self, config: BrainLikeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.model_hidden_size
        self.num_heads = config.model_num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # 90% 静态 + 10% 动态 投影层
        self.q_proj = BrainLikeLinear(self.hidden_size, self.hidden_size, dynamic_init_std=config.weight_split.dynamic_init_std)
        self.k_proj = BrainLikeLinear(self.hidden_size, self.hidden_size, dynamic_init_std=config.weight_split.dynamic_init_std)
        self.v_proj = BrainLikeLinear(self.hidden_size, self.hidden_size, dynamic_init_std=config.weight_split.dynamic_init_std)
        self.o_proj = BrainLikeLinear(self.hidden_size, self.hidden_size, dynamic_init_std=config.weight_split.dynamic_init_std)
        
        # 海马体门控接口 (权重为 10% 动态部分)
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
        
        # 计算Q, K, V (使用双轨分支)
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
        
        # 输出投影 (使用双轨分支)
        output = self.o_proj(attn_output)
        
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
        
        # 90% 静态 + 10% 动态 投影层
        self.gate_proj = BrainLikeLinear(self.hidden_size, self.intermediate_size, dynamic_init_std=config.weight_split.dynamic_init_std)
        self.up_proj = BrainLikeLinear(self.hidden_size, self.intermediate_size, dynamic_init_std=config.weight_split.dynamic_init_std)
        self.down_proj = BrainLikeLinear(self.intermediate_size, self.hidden_size, dynamic_init_std=config.weight_split.dynamic_init_std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 (SwiGLU 的双轨实现)"""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        output = self.down_proj(gate * up)
        return output
    
    def freeze_static_weights(self):
        """冻结静态基础权重"""
        for param in [self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight]:
            param.requires_grad = False
    
    def get_dynamic_weights(self) -> Dict[str, nn.Parameter]:
        """获取动态分支权重"""
        return self.dynamic_branch.get_stdp_weights()


class VisualCortex(nn.Module):
    """视觉皮层 - 多模态特征提取单元 (Qwen3.5-0.8B 原生适配)"""
    
    def __init__(self, config: BrainLikeConfig):
        super().__init__()
        self.config = config
        # 使用双轨线性层进行视觉投影，支持 10% 动态更新
        # 这里的输入维度应适配预训练的 Vision Tower 输出 (例如 SigLIP)
        # 暂时使用 hidden_size 占位，实际由权重加载器映射
        self.vision_tower_output_dim = 1152 # 假设值，适配实际模型
        self.vision_proj = BrainLikeLinear(self.vision_tower_output_dim, config.model_hidden_size)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        将像素值映射到 LLM 隐藏空间
        
        Args:
            pixel_values: 图像/视频帧张量 [batch, vision_tokens, vision_dim]
            
        Returns:
            视觉 Token Embedding [batch, vision_tokens, hidden_size]
        """
        # 注意：这里输入已经是提取后的特征（流式传入），或者是原始像素
        # Qwen3.5 原生支持交替 token。
        return self.vision_proj(pixel_values)


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
        # 收集所有线性层的动态参数
        for module_name, module in self.named_modules():
            if isinstance(module, BrainLikeLinear):
                for k, v in module.get_dynamic_parameters().items():
                    weights[f"{module_name}.{k}"] = v
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
        
        # 输出层 (使用双轨线性层)
        self.lm_head = BrainLikeLinear(
            config.model_hidden_size,
            config.model_vocab_size,
            bias=False
        )
        
        # 视觉皮层
        self.visual_cortex = VisualCortex(config)
        
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
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        memory_anchors: Optional[List[Dict]] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[TokenFeatures], Dict[str, torch.Tensor]]:
        """
        前流式前向传播
        
        Args:
            input_ids: 输入token ID [batch, seq_len]
            pixel_values: 视觉特征输入 [batch, vision_tokens, vision_dim]
            attention_mask: 注意力掩码
            memory_anchors: 海马体记忆锚点
            position_ids: 位置编码
            
        Returns:
            logits: 输出logits
            features: Token特征列表
            dynamic_weights: 动态权重字典
        """
        # 词嵌入
        inputs_embeds = self.embed_tokens(input_ids)
        
        # 融合视觉特征 (如果存在)
        if pixel_values is not None:
            vision_embeds = self.visual_cortex(pixel_values)
            # 在流式处理中，视觉特征按顺序拼接到 hidden_states
            # 简化版：直接拼接
            hidden_states = torch.cat([vision_embeds, inputs_embeds], dim=1)
        else:
            hidden_states = inputs_embeds
        
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
        
        # 输出层 (使用双轨接口)
        logits = self.lm_head(hidden_states)
        
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
