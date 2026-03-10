"""
类人脑双系统全闭环AI架构 - STDP时序可塑性权重刷新系统
Human-Like Brain Dual-System Full-Loop AI Architecture - STDP Weight Update System

实现Transformer原生适配的STDP机制：
- LTP（长期增强）：时序正确的权重增强
- LTD（长期减弱）：时序错误或无效的权重减弱
- 全节点覆盖：注意力层、FFN层、自评判、海马体门控
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import math

from core.config import BrainLikeConfig, STDPConfig


class STDPType(Enum):
    """STDP更新类型"""
    LTP = "ltp"  # 长期增强
    LTD = "ltd"  # 长期减弱


@dataclass
class STDPEvent:
    """STDP事件记录"""
    pre_activation_time: float  # 前序激活时间
    post_activation_time: float  # 后序激活时间
    weight_name: str  # 权重名称
    contribution_score: float  # 贡献度分数
    stdp_type: STDPType  # 更新类型
    update_magnitude: float  # 更新幅度


@dataclass
class STDPStatistics:
    """STDP统计信息"""
    total_updates: int = 0
    ltp_count: int = 0
    ltd_count: int = 0
    total_ltp_magnitude: float = 0.0
    total_ltd_magnitude: float = 0.0
    average_update: float = 0.0
    weight_change_history: List[float] = field(default_factory=list)


class STDPKernel:
    """
    STDP核函数
    
    实现基于时序差值的权重更新计算
    """
    
    def __init__(self, config: STDPConfig):
        self.config = config
        self.timing_window = config.timing_window_ms
        self.alpha = config.alpha  # LTP学习率
        self.beta = config.beta  # LTD学习率
    
    def compute_update(
        self,
        delta_t: float,  # 时间差（后序-前序），毫秒
        contribution: float = 1.0  # 贡献度分数
    ) -> Tuple[float, STDPType]:
        """
        计算STDP更新量
        
        Args:
            delta_t: 时间差（后序激活时间 - 前序激活时间）
            contribution: 对当前输出的贡献度分数（0-1）
            
        Returns:
            update: 权重更新量
            stdp_type: 更新类型（LTP或LTD）
        """
        # 在时序窗口内才更新
        if abs(delta_t) > self.timing_window:
            return 0.0, STDPType.LTP
        
        if delta_t > 0:
            # 前序先激活，后序后激活 -> LTP（增强）
            # 使用指数衰减函数
            update = self.alpha * contribution * math.exp(-delta_t / self.timing_window)
            return update, STDPType.LTP
        else:
            # 后序先激活，前序后激活 -> LTD（减弱）
            update = -self.beta * contribution * math.exp(delta_t / self.timing_window)
            return update, STDPType.LTD
    
    def compute_ltp_curve(self, delta_t: float) -> float:
        """计算LTP曲线值"""
        if delta_t <= 0 or delta_t > self.timing_window:
            return 0.0
        return self.alpha * math.exp(-delta_t / self.timing_window)
    
    def compute_ltd_curve(self, delta_t: float) -> float:
        """计算LTD曲线值"""
        if delta_t >= 0 or abs(delta_t) > self.timing_window:
            return 0.0
        return -self.beta * math.exp(delta_t / self.timing_window)


class AttentionSTDP:
    """
    注意力层STDP更新器
    
    根据窄窗口内上下文与当前token的时序关联、语义贡献度，
    实时刷新动态注意力权重
    """
    
    def __init__(self, config: STDPConfig):
        self.config = config
        self.kernel = STDPKernel(config)
        self._event_history: List[STDPEvent] = []
    
    def compute_attention_stdp(
        self,
        attention_weights: torch.Tensor,
        current_time: float,
        context_times: List[float],
        contribution_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        计算注意力层的STDP更新
        
        Args:
            attention_weights: 当前注意力权重 [batch, heads, seq_len]
            current_time: 当前时间戳
            context_times: 上下文token的时间戳列表
            contribution_scores: 各上下文token的贡献度分数
            
        Returns:
            更新量张量
        """
        batch_size, num_heads, seq_len = attention_weights.shape
        updates = torch.zeros_like(attention_weights)
        
        for i, ctx_time in enumerate(context_times):
            delta_t = current_time - ctx_time
            
            for h in range(num_heads):
                for b in range(batch_size):
                    contribution = contribution_scores[b, h, i].item()
                    update, stdp_type = self.kernel.compute_update(delta_t, contribution)
                    updates[b, h, i] = update
        
        return updates
    
    def get_events(self) -> List[STDPEvent]:
        """获取事件历史"""
        return self._event_history


class FFNSTDP:
    """
    FFN层STDP更新器
    
    对当前任务、当前会话的高频特征、专属术语、用户习惯表达，
    自动增强对应FFN层的动态权重
    """
    
    def __init__(self, config: STDPConfig):
        self.config = config
        self.kernel = STDPKernel(config)
        self._feature_frequency: Dict[str, int] = {}
        self._session_patterns: Dict[str, float] = {}
    
    def compute_ffn_stdp(
        self,
        hidden_states: torch.Tensor,
        output_states: torch.Tensor,
        current_time: float,
        activation_times: List[float]
    ) -> torch.Tensor:
        """
        计算FFN层的STDP更新
        
        Args:
            hidden_states: 输入隐藏状态
            output_states: 输出隐藏状态
            current_time: 当前时间戳
            activation_times: 各神经元的激活时间
            
        Returns:
            更新量张量
        """
        # 计算神经元贡献度
        contribution = torch.sigmoid(output_states)
        
        updates = torch.zeros_like(hidden_states)
        
        for i, act_time in enumerate(activation_times):
            delta_t = current_time - act_time
            update, _ = self.kernel.compute_update(delta_t, contribution[i].item())
            updates[:, i] = update
        
        return updates
    
    def update_feature_frequency(self, feature_id: str):
        """更新特征频率统计"""
        self._feature_frequency[feature_id] = self._feature_frequency.get(feature_id, 0) + 1
    
    def get_high_frequency_features(self, threshold: int = 5) -> List[str]:
        """获取高频特征"""
        return [k for k, v in self._feature_frequency.items() if v >= threshold]


class SelfJudgmentSTDP:
    """
    自评判STDP更新器
    
    每10个刷新周期，根据模型自评判结果，
    对正确、优质的生成路径增强动态权重，
    对错误、劣质的路径减弱权重
    """
    
    def __init__(self, config: STDPConfig):
        self.config = config
        self.kernel = STDPKernel(config)
        self._judgment_interval = config.self_judgment_interval
        self._cycle_count = 0
        self._judgment_history: List[Dict] = []
    
    def should_judge(self) -> bool:
        """判断是否应该执行自评判"""
        self._cycle_count += 1
        return self._cycle_count >= self._judgment_interval
    
    def compute_judgment_stdp(
        self,
        candidate_weights: Dict[str, torch.Tensor],
        judgment_scores: Dict[str, float],
        correct_paths: List[str],
        incorrect_paths: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        计算自评判STDP更新
        
        Args:
            candidate_weights: 候选路径的权重
            judgment_scores: 各维度的评判分数
            correct_paths: 正确路径列表
            incorrect_paths: 错误路径列表
            
        Returns:
            各权重的更新量字典
        """
        updates = {}
        
        # 计算总分
        total_score = sum(judgment_scores.values())
        max_score = len(judgment_scores) * 10  # 每维度最高10分
        quality_ratio = total_score / max_score
        
        # 对正确路径增强
        for path in correct_paths:
            if path in candidate_weights:
                update = self.config.alpha * quality_ratio
                updates[path] = torch.ones_like(candidate_weights[path]) * update
        
        # 对错误路径减弱
        for path in incorrect_paths:
            if path in candidate_weights:
                update = -self.config.beta * (1 - quality_ratio)
                updates[path] = torch.ones_like(candidate_weights[path]) * update
        
        # 记录评判历史
        self._judgment_history.append({
            'cycle': self._cycle_count,
            'scores': judgment_scores,
            'quality_ratio': quality_ratio
        })
        
        # 重置计数
        self._cycle_count = 0
        
        return updates
    
    def get_judgment_history(self) -> List[Dict]:
        """获取评判历史"""
        return self._judgment_history


class HippocampusGateSTDP:
    """
    海马体门控STDP更新器
    
    对推理有正向贡献的记忆锚点，对应的连接权重自动增强，
    无效的记忆锚点权重自动减弱
    """
    
    def __init__(self, config: STDPConfig):
        self.config = config
        self.kernel = STDPKernel(config)
        self._anchor_effectiveness: Dict[str, float] = {}
    
    def compute_gate_stdp(
        self,
        gate_weights: torch.Tensor,
        anchor_contributions: Dict[str, float],
        current_time: float,
        anchor_times: Dict[str, float]
    ) -> torch.Tensor:
        """
        计算海马体门控STDP更新
        
        Args:
            gate_weights: 门控权重
            anchor_contributions: 各记忆锚点的贡献度
            current_time: 当前时间戳
            anchor_times: 各锚点的时间戳
            
        Returns:
            更新量张量
        """
        updates = torch.zeros_like(gate_weights)
        
        for anchor_id, contribution in anchor_contributions.items():
            if anchor_id in anchor_times:
                delta_t = current_time - anchor_times[anchor_id]
                update, _ = self.kernel.compute_update(delta_t, contribution)
                
                # 更新锚点有效性记录
                self._anchor_effectiveness[anchor_id] = (
                    self._anchor_effectiveness.get(anchor_id, 0.5) * 0.9 + contribution * 0.1
                )
        
        return updates
    
    def get_effective_anchors(self, threshold: float = 0.7) -> List[str]:
        """获取有效锚点列表"""
        return [k for k, v in self._anchor_effectiveness.items() if v >= threshold]


class STDPSystem:
    """
    STDP时序可塑性权重刷新系统
    
    整合所有STDP更新器，实现全链路权重更新
    """
    
    def __init__(self, config: BrainLikeConfig):
        self.config = config
        self.stdp_config = config.stdp
        
        # 初始化各节点STDP更新器
        self.attention_stdp = AttentionSTDP(self.stdp_config) if self.stdp_config.enable_attention_stdp else None
        self.ffn_stdp = FFNSTDP(self.stdp_config) if self.stdp_config.enable_ffn_stdp else None
        self.judgment_stdp = SelfJudgmentSTDP(self.stdp_config) if self.stdp_config.enable_self_judgment_stdp else None
        self.hippocampus_stdp = HippocampusGateSTDP(self.stdp_config) if self.stdp_config.enable_hippocampus_stdp else None
        
        # 统计信息
        self._statistics = STDPStatistics()
        
        # 权重缓存
        self._weight_cache: Dict[str, torch.Tensor] = {}
    
    def compute_all_updates(
        self,
        model_output: Dict[str, Any],
        current_time: float
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有节点的STDP更新
        
        Args:
            model_output: 模型输出，包含各层的中间结果
            current_time: 当前时间戳
            
        Returns:
            所有权重的更新量字典
        """
        all_updates = {}
        
        # 注意力层STDP更新
        if self.attention_stdp and 'attention_weights' in model_output:
            attn_updates = self.attention_stdp.compute_attention_stdp(
                model_output['attention_weights'],
                current_time,
                model_output.get('context_times', []),
                model_output.get('contribution_scores', torch.ones_like(model_output['attention_weights']))
            )
            all_updates['attention'] = attn_updates
        
        # FFN层STDP更新
        if self.ffn_stdp and 'hidden_states' in model_output:
            ffn_updates = self.ffn_stdp.compute_ffn_stdp(
                model_output['hidden_states'],
                model_output.get('output_states', model_output['hidden_states']),
                current_time,
                model_output.get('activation_times', [current_time - 10])
            )
            all_updates['ffn'] = ffn_updates
        
        # 自评判STDP更新
        if self.judgment_stdp and self.judgment_stdp.should_judge():
            if 'judgment_scores' in model_output:
                judgment_updates = self.judgment_stdp.compute_judgment_stdp(
                    model_output.get('candidate_weights', {}),
                    model_output['judgment_scores'],
                    model_output.get('correct_paths', []),
                    model_output.get('incorrect_paths', [])
                )
                all_updates.update(judgment_updates)
        
        # 海马体门控STDP更新
        if self.hippocampus_stdp and 'gate_weights' in model_output:
            gate_updates = self.hippocampus_stdp.compute_gate_stdp(
                model_output['gate_weights'],
                model_output.get('anchor_contributions', {}),
                current_time,
                model_output.get('anchor_times', {})
            )
            all_updates['hippocampus_gate'] = gate_updates
        
        return all_updates
    
    def apply_update(
        self,
        param: nn.Parameter,
        update: torch.Tensor
    ):
        """
        应用STDP权重更新
        
        Args:
            param: 权重参数对象
            update: 更新量
        """
        if not param.requires_grad:
            return
            
        # 应用更新 (纯本地刷新，无反向传播)
        # update 可能是一个标量或者与 param 同形状的张量
        if update.dim() == 0:
            param.data += update
        else:
            # 确保形状匹配
            if update.shape == param.shape:
                param.data += update
            else:
                # 均值更新
                param.data += update.mean()
        
        # 权重裁剪 (生物脑突触强度限制)
        param.data = torch.clamp(
            param.data,
            self.stdp_config.weight_min,
            self.stdp_config.weight_max
        )
        
        # 更新统计
        self._update_statistics(update)
    
    def apply_all_updates(
        self,
        updates: Dict[str, torch.Tensor],
        dynamic_params: Dict[str, nn.Parameter]
    ):
        """
        应用所有STDP更新
        
        Args:
            updates: 更新量字典
            dynamic_params: 动态权重参数字典 (来自 model.get_all_dynamic_weights())
        """
        for name, update in updates.items():
            if name in dynamic_params:
                self.apply_update(dynamic_params[name], update)
    
    def _update_statistics(self, update: torch.Tensor):
        """更新统计信息"""
        self._statistics.total_updates += 1
        
        mean_update = update.abs().mean().item()
        self._statistics.average_update = (
            self._statistics.average_update * (self._statistics.total_updates - 1) + mean_update
        ) / self._statistics.total_updates
        
        if update.mean().item() > 0:
            self._statistics.ltp_count += 1
            self._statistics.total_ltp_magnitude += mean_update
        else:
            self._statistics.ltd_count += 1
            self._statistics.total_ltd_magnitude += mean_update
        
        self._statistics.weight_change_history.append(mean_update)
    
    def get_statistics(self) -> STDPStatistics:
        """获取统计信息"""
        return self._statistics
    
    def reset_statistics(self):
        """重置统计信息"""
        self._statistics = STDPStatistics()
    
    def get_config(self) -> Dict[str, Any]:
        """获取STDP配置"""
        return {
            'alpha': self.stdp_config.alpha,
            'beta': self.stdp_config.beta,
            'weight_min': self.stdp_config.weight_min,
            'weight_max': self.stdp_config.weight_max,
            'timing_window_ms': self.stdp_config.timing_window_ms,
            'self_judgment_interval': self.stdp_config.self_judgment_interval
        }
    
    def set_learning_rates(self, alpha: float, beta: float):
        """
        设置学习率
        
        Args:
            alpha: LTP学习率
            beta: LTD学习率
        """
        self.stdp_config.alpha = alpha
        self.stdp_config.beta = beta
        
        # 更新各更新器的学习率
        if self.attention_stdp:
            self.attention_stdp.kernel.alpha = alpha
            self.attention_stdp.kernel.beta = beta
        if self.ffn_stdp:
            self.ffn_stdp.kernel.alpha = alpha
            self.ffn_stdp.kernel.beta = beta
        if self.judgment_stdp:
            self.judgment_stdp.kernel.alpha = alpha
            self.judgment_stdp.kernel.beta = beta
        if self.hippocampus_stdp:
            self.hippocampus_stdp.kernel.alpha = alpha
            self.hippocampus_stdp.kernel.beta = beta
