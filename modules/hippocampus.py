"""
类人脑双系统全闭环AI架构 - 海马体记忆系统模块
Human-Like Brain Dual-System Full-Loop AI Architecture - Hippocampus Memory System

严格基于人脑海马体-新皮层双系统神经科学原理开发：
- EC内嗅皮层：特征编码单元
- DG齿状回：模式分离单元
- CA3区：情景记忆库+模式补全单元
- CA1区：时序编码+注意力门控单元
- SWR尖波涟漪：离线回放巩固单元
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import time
import math
import random

from core.config import BrainLikeConfig, HippocampusConfig


@dataclass
class MemoryUnit:
    """记忆单元"""
    memory_id: str
    timestamp_ms: float
    feature_vector: torch.Tensor
    semantic_pointer: str
    temporal_skeleton: List[float]  # 时序骨架
    causal_links: List[str]  # 因果关联
    access_count: int = 0
    last_access_time: float = 0.0
    consolidation_strength: float = 0.0


@dataclass
class MemoryAnchor:
    """记忆锚点"""
    anchor_id: str
    memory_unit: MemoryUnit
    relevance_score: float
    gate_signal: torch.Tensor


class EntorhinalCortex:
    """
    内嗅皮层（EC）- 特征编码单元
    
    接收模型注意力层输出的token特征，
    归一化稀疏编码为64维固定低维特征向量
    """
    
    def __init__(self, config: HippocampusConfig):
        self.config = config
        self.feature_dim = config.ec_feature_dim
        self.sparse_ratio = config.ec_sparse_ratio
        
        # 稀疏编码投影矩阵
        self.projection_matrix: Optional[torch.Tensor] = None
    
    def initialize(self, input_dim: int):
        """初始化投影矩阵"""
        # 使用随机正交矩阵进行投影
        random_matrix = torch.randn(input_dim, self.feature_dim)
        # 正交化
        self.projection_matrix, _ = torch.linalg.qr(random_matrix)
    
    def encode(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        编码特征
        
        Args:
            features: 输入特征 [batch, seq_len, input_dim]
            
        Returns:
            编码后的稀疏特征 [batch, seq_len, feature_dim]
        """
        if self.projection_matrix is None:
            self.initialize(features.shape[-1])
        
        # 投影到低维空间
        encoded = torch.matmul(features, self.projection_matrix.to(features.device))
        
        # 归一化
        encoded = F.normalize(encoded, dim=-1)
        
        # 稀疏化：只保留top-k个最大值
        k = int(self.feature_dim * self.sparse_ratio)
        if k < self.feature_dim:
            # 获取top-k的阈值
            values, _ = torch.topk(encoded.abs(), k, dim=-1)
            threshold = values[..., -1:].expand_as(encoded)
            # 应用稀疏掩码
            mask = (encoded.abs() >= threshold).float()
            encoded = encoded * mask
        
        return encoded
    
    def decode(
        self,
        encoded: torch.Tensor
    ) -> torch.Tensor:
        """
        解码特征（近似重建）
        
        Args:
            encoded: 编码特征 [batch, seq_len, feature_dim]
            
        Returns:
            重建特征 [batch, seq_len, input_dim]
        """
        if self.projection_matrix is None:
            return encoded
        
        # 使用投影矩阵的伪逆进行重建
        pseudo_inverse = torch.linalg.pinv(self.projection_matrix.to(encoded.device))
        return torch.matmul(encoded, pseudo_inverse.T)


class DentateGyrus:
    """
    齿状回（DG）- 模式分离单元
    
    对编码特征做稀疏随机投影正交化处理，
    为相似输入生成完全正交的唯一记忆ID
    """
    
    def __init__(self, config: HippocampusConfig):
        self.config = config
        self.separation_strength = config.dg_pattern_separation_strength
        self.orthogonal_dim = config.dg_orthogonal_dim
        
        # 随机投影矩阵（固定，无训练参数）
        self._projection_matrix: Optional[torch.Tensor] = None
        self._orthogonal_basis: Optional[torch.Tensor] = None
    
    def initialize(self, input_dim: int):
        """初始化随机投影"""
        # 创建随机稀疏投影矩阵
        self._projection_matrix = torch.randn(input_dim, self.orthogonal_dim)
        self._projection_matrix = F.normalize(self._projection_matrix, dim=0)
        
        # 创建正交基
        self._orthogonal_basis, _ = torch.linalg.qr(
            torch.randn(self.orthogonal_dim, self.orthogonal_dim)
        )
    
    def separate(
        self,
        encoded_features: torch.Tensor
    ) -> Tuple[torch.Tensor, str]:
        """
        模式分离
        
        Args:
            encoded_features: 编码特征 [batch, feature_dim]
            
        Returns:
            separated: 分离后的正交特征
            memory_id: 生成的唯一记忆ID
        """
        if self._projection_matrix is None:
            self.initialize(encoded_features.shape[-1])
        
        # 随机投影
        projected = torch.matmul(
            encoded_features, 
            self._projection_matrix.to(encoded_features.device)
        )
        
        # 正交化变换
        separated = torch.matmul(
            projected, 
            self._orthogonal_basis.to(encoded_features.device)
        )
        
        # 应用非线性增强分离效果
        separated = torch.sign(separated) * torch.pow(
            separated.abs(), 
            1.0 / self.separation_strength
        )
        
        # 归一化
        separated = F.normalize(separated, dim=-1)
        
        # 生成唯一记忆ID
        memory_id = self._generate_memory_id(separated)
        
        return separated, memory_id
    
    def _generate_memory_id(self, features: torch.Tensor) -> str:
        """生成唯一记忆ID"""
        # 使用特征的哈希作为ID
        feature_hash = hash(tuple(features.flatten().tolist()))
        timestamp = int(time.time() * 1000000)
        return f"mem_{timestamp}_{abs(feature_hash) % 1000000:06d}"
    
    def compute_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> float:
        """
        计算分离后的相似度
        
        模式分离后，相似输入的相似度应显著降低
        """
        return F.cosine_similarity(
            features1.flatten().unsqueeze(0),
            features2.flatten().unsqueeze(0)
        ).item()


class CA3Region:
    """
    CA3区 - 情景记忆库+模式补全单元
    
    以「记忆ID+10ms级时间戳+时序骨架+语义指针+因果关联」格式存储情景记忆，
    仅存指针不存完整文本；
    基于部分线索完成完整记忆链条召回
    """
    
    def __init__(self, config: HippocampusConfig):
        self.config = config
        self.capacity = config.ca3_memory_capacity
        self.recall_top_k = config.ca3_recall_top_k
        self.completion_threshold = config.ca3_completion_threshold
        
        # 记忆存储（循环缓存，显式限流在 2MB 以内）
        self._memory_store: Dict[str, MemoryUnit] = {}
        self._memory_order: deque = deque(maxlen=self.capacity)
        
        # 存算分离：活跃特征在 RAM，非活跃可能持久化（这里简化为 RAM 循环缓存）
        self._max_ram_bytes = 2 * 1024 * 1024 # 2MB
        
        # 特征索引（用于快速检索）
        self._feature_index: Optional[torch.Tensor] = None
        self._memory_ids: List[str] = []
    
    def store(
        self,
        memory_id: str,
        features: torch.Tensor,
        timestamp_ms: float,
        semantic_info: Dict[str, Any]
    ) -> bool:
        """
        存储情景记忆
        
        Args:
            memory_id: 记忆ID
            features: 特征向量
            timestamp_ms: 时间戳
            semantic_info: 语义信息
            
        Returns:
            是否存储成功
        """
        # 动态控制：如果接近 2MB，强制清理
        current_size = len(self._memory_store) * (self.config.ec_feature_dim * 4 + 256) # 估算每个 Unit 大小
        if current_size >= self._max_ram_bytes - 1024 or len(self._memory_store) >= self.capacity:
            # 移除最旧的记忆 (FIFO，类似人脑遗忘机制)
            if self._memory_order:
                oldest_id = self._memory_order.popleft()
                if oldest_id in self._memory_store:
                    del self._memory_store[oldest_id]
        
        # 创建记忆单元
        memory_unit = MemoryUnit(
            memory_id=memory_id,
            timestamp_ms=timestamp_ms,
            feature_vector=features.detach().clone(),
            semantic_pointer=semantic_info.get('semantic_pointer', ''),
            temporal_skeleton=semantic_info.get('temporal_skeleton', []),
            causal_links=semantic_info.get('causal_links', []),
            access_count=0,
            last_access_time=time.time()
        )
        
        # 存储
        self._memory_store[memory_id] = memory_unit
        self._memory_order.append(memory_id)
        
        # 更新索引
        self._update_index(memory_id, features)
        
        return True
    
    def recall(
        self,
        cue: torch.Tensor,
        top_k: int = None
    ) -> List[MemoryAnchor]:
        """
        召回记忆
        
        Args:
            cue: 召回线索特征
            top_k: 返回的记忆数量
            
        Returns:
            记忆锚点列表
        """
        if top_k is None:
            top_k = self.recall_top_k
        
        if not self._memory_store:
            return []
        
        # 确保cue是正确的形状
        cue_flat = cue.flatten()
        
        # 计算与所有记忆的相似度
        similarities = []
        for memory_id, memory_unit in self._memory_store.items():
            memory_flat = memory_unit.feature_vector.flatten()
            
            # 处理维度不匹配的情况
            if cue_flat.shape[0] != memory_flat.shape[0]:
                # 使用较小的维度进行匹配
                min_dim = min(cue_flat.shape[0], memory_flat.shape[0])
                cue_compare = cue_flat[:min_dim]
                memory_compare = memory_flat[:min_dim]
            else:
                cue_compare = cue_flat
                memory_compare = memory_flat
            
            similarity = F.cosine_similarity(
                cue_compare.unsqueeze(0),
                memory_compare.unsqueeze(0)
            ).item()
            similarities.append((memory_id, similarity))
        
        # 排序并选择top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_memories = similarities[:top_k]
        
        # 构建记忆锚点
        anchors = []
        for memory_id, relevance in top_memories:
            memory_unit = self._memory_store[memory_id]
            
            # 更新访问计数
            memory_unit.access_count += 1
            memory_unit.last_access_time = time.time()
            
            # 创建门控信号
            gate_signal = self._create_gate_signal(
                memory_unit.feature_vector,
                relevance
            )
            
            anchor = MemoryAnchor(
                anchor_id=f"anchor_{memory_id}",
                memory_unit=memory_unit,
                relevance_score=relevance,
                gate_signal=gate_signal
            )
            anchors.append(anchor)
        
        return anchors
    
    def complete_pattern(
        self,
        partial_cue: torch.Tensor
    ) -> Optional[MemoryUnit]:
        """
        模式补全
        
        基于部分线索补全完整记忆
        
        Args:
            partial_cue: 部分线索特征
            
        Returns:
            补全的记忆单元
        """
        anchors = self.recall(partial_cue, top_k=1)
        
        if not anchors:
            return None
        
        best_anchor = anchors[0]
        
        # 检查是否达到补全阈值
        if best_anchor.relevance_score >= self.completion_threshold:
            return best_anchor.memory_unit
        
        return None
    
    def _update_index(self, memory_id: str, features: torch.Tensor):
        """更新特征索引"""
        self._memory_ids.append(memory_id)
        
        if self._feature_index is None:
            self._feature_index = features.detach().clone().unsqueeze(0)
        else:
            self._feature_index = torch.cat([
                self._feature_index,
                features.detach().clone().unsqueeze(0)
            ], dim=0)
    
    def _create_gate_signal(
        self,
        features: torch.Tensor,
        relevance: float
    ) -> torch.Tensor:
        """创建门控信号"""
        # 门控信号 = 特征 * 相关性权重
        return features * relevance
    
    def get_memory_count(self) -> int:
        """获取记忆数量"""
        return len(self._memory_store)
    
    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryUnit]:
        """按ID获取记忆"""
        return self._memory_store.get(memory_id)


class CA1Region:
    """
    CA1区 - 时序编码+注意力门控单元
    
    为每个记忆单元打精准时间戳，绑定时序-情景-因果关系，
    形成连续记忆链条；
    每个刷新周期输出记忆锚点给模型注意力层
    """
    
    def __init__(self, config: HippocampusConfig):
        self.config = config
        self.timestamp_precision = config.ca1_timestamp_precision_ms
        self.gate_strength = config.ca1_gate_strength
        
        # 时序链条
        self._temporal_chain: List[str] = []
        self._causal_graph: Dict[str, List[str]] = {}
    
    def encode_temporal(
        self,
        memory_id: str,
        timestamp_ms: float,
        prev_memory_id: Optional[str] = None
    ):
        """
        时序编码
        
        Args:
            memory_id: 当前记忆ID
            timestamp_ms: 时间戳
            prev_memory_id: 前序记忆ID
        """
        # 添加到时序链条
        self._temporal_chain.append(memory_id)
        
        # 建立因果关联
        if prev_memory_id:
            if prev_memory_id not in self._causal_graph:
                self._causal_graph[prev_memory_id] = []
            self._causal_graph[prev_memory_id].append(memory_id)
    
    def generate_gate_signal(
        self,
        memory_anchors: List[MemoryAnchor]
    ) -> torch.Tensor:
        """
        生成注意力门控信号
        
        Args:
            memory_anchors: 记忆锚点列表
            
        Returns:
            门控信号
        """
        if not memory_anchors:
            return torch.zeros(1)
        
        # 加权聚合所有锚点的门控信号
        total_signal = torch.zeros_like(memory_anchors[0].gate_signal)
        total_weight = 0.0
        
        for anchor in memory_anchors:
            weight = anchor.relevance_score * self.gate_strength
            total_signal += anchor.gate_signal * weight
            total_weight += weight
        
        if total_weight > 0:
            total_signal /= total_weight
        
        return total_signal
    
    def get_temporal_sequence(
        self,
        start_id: str,
        length: int = 10
    ) -> List[str]:
        """
        获取时序序列
        
        Args:
            start_id: 起始记忆ID
            length: 序列长度
            
        Returns:
            记忆ID序列
        """
        sequence = []
        current_id = start_id
        
        for _ in range(length):
            if current_id not in self._temporal_chain:
                break
            
            sequence.append(current_id)
            
            # 获取下一个记忆
            idx = self._temporal_chain.index(current_id)
            if idx + 1 < len(self._temporal_chain):
                current_id = self._temporal_chain[idx + 1]
            else:
                break
        
        return sequence
    
    def get_causal_chain(
        self,
        memory_id: str
    ) -> List[str]:
        """
        获取因果链条
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            因果关联的记忆ID列表
        """
        return self._causal_graph.get(memory_id, [])


class SharpWaveRipple:
    """
    尖波涟漪（SWR）- 离线回放巩固单元
    
    端侧空闲时，模拟人脑睡眠尖波涟漪，
    回放记忆序列与推理过程，完成记忆巩固、权重优化、记忆修剪
    """
    
    def __init__(self, config: HippocampusConfig):
        self.config = config
        self.enabled = config.swr_enabled
        self.idle_threshold = config.swr_idle_threshold_minutes
        self.replay_frequency = config.swr_replay_frequency
        self.consolidation_ratio = config.swr_consolidation_ratio
        
        # 回放状态
        self._is_replaying = False
        self._last_activity_time = time.time()
        self._replay_count = 0
    
    def check_idle(self) -> bool:
        """检查是否处于空闲状态"""
        if not self.enabled:
            return False
        
        idle_time = (time.time() - self._last_activity_time) / 60  # 分钟
        return idle_time >= self.idle_threshold
    
    def start_replay(
        self,
        ca3: CA3Region,
        ca1: CA1Region
    ) -> List[Dict[str, Any]]:
        """
        开始记忆回放
        
        Args:
            ca3: CA3区实例
            ca1: CA1区实例
            
        Returns:
            回放的记忆序列
        """
        if not self.enabled or self._is_replaying:
            return []
        
        self._is_replaying = True
        self._replay_count += 1
        
        # 选择需要巩固的记忆
        memories_to_consolidate = self._select_memories_for_consolidation(ca3)
        
        # 回放记忆序列
        replay_sequence = []
        for memory_id in memories_to_consolidate:
            memory_unit = ca3.get_memory_by_id(memory_id)
            if memory_unit:
                # 获取时序序列
                temporal_seq = ca1.get_temporal_sequence(memory_id, length=5)
                
                replay_sequence.append({
                    'memory_id': memory_id,
                    'feature': memory_unit.feature_vector,
                    'temporal_sequence': temporal_seq,
                    'access_count': memory_unit.access_count
                })
        
        return replay_sequence
    
    def end_replay(self):
        """结束回放"""
        self._is_replaying = False
        self._last_activity_time = time.time()
    
    def _select_memories_for_consolidation(
        self,
        ca3: CA3Region
    ) -> List[str]:
        """
        选择需要巩固的记忆
        
        优先选择：
        1. 访问频率高的记忆
        2. 最近形成的记忆
        3. 与其他记忆关联多的记忆
        """
        all_memories = list(ca3._memory_store.items())
        
        # 计算巩固优先级分数
        scored_memories = []
        for memory_id, memory_unit in all_memories:
            # 访问频率分数
            access_score = min(memory_unit.access_count / 10, 1.0)
            
            # 新近度分数
            age = (time.time() - memory_unit.last_access_time) / 3600  # 小时
            recency_score = max(0, 1 - age / 24)  # 24小时内
            
            # 关联度分数
            causal_score = min(len(memory_unit.causal_links) / 5, 1.0)
            
            # 综合分数
            total_score = access_score * 0.4 + recency_score * 0.3 + causal_score * 0.3
            
            scored_memories.append((memory_id, total_score))
        
        # 排序并选择
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        num_to_consolidate = int(len(scored_memories) * self.consolidation_ratio)
        
        return [m[0] for m in scored_memories[:num_to_consolidate]]
    
    def prune_memories(
        self,
        ca3: CA3Region,
        threshold_access: int = 1,
        max_age_hours: float = 48.0
    ) -> int:
        """
        修剪无效记忆
        
        Args:
            ca3: CA3区实例
            threshold_access: 访问次数阈值
            max_age_hours: 最大保留时间（小时）
            
        Returns:
            修剪的记忆数量
        """
        pruned = 0
        to_remove = []
        
        for memory_id, memory_unit in ca3._memory_store.items():
            age_hours = (time.time() - memory_unit.last_access_time) / 3600
            
            # 删除条件：访问次数少且时间久
            if memory_unit.access_count <= threshold_access and age_hours > max_age_hours:
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            del ca3._memory_store[memory_id]
            if memory_id in ca3._memory_order:
                ca3._memory_order.remove(memory_id)
            pruned += 1
        
        return pruned


class HippocampusSystem:
    """
    海马体记忆系统
    
    整合EC、DG、CA3、CA1、SWR各模块，
    实现完整的类脑海马体功能
    """
    
    def __init__(self, config: BrainLikeConfig):
        self.config = config
        self.hippo_config = config.hippocampus
        
        # 初始化各子模块
        self.ec = EntorhinalCortex(self.hippo_config)
        self.dg = DentateGyrus(self.hippo_config)
        self.ca3 = CA3Region(self.hippo_config)
        self.ca1 = CA1Region(self.hippo_config)
        self.swr = SharpWaveRipple(self.hippo_config)
        
        # 上一个记忆ID（用于建立时序关联）
        self._last_memory_id: Optional[str] = None
        
        # 统计信息
        self._encode_count = 0
        self._recall_count = 0
    
    def encode_episode(
        self,
        features: torch.Tensor,
        timestamp_ms: float,
        semantic_info: Dict[str, Any]
    ) -> str:
        """
        编码情景记忆
        
        完整流程：EC编码 -> DG模式分离 -> CA3存储 -> CA1时序编码
        
        Args:
            features: 输入特征
            timestamp_ms: 时间戳
            semantic_info: 语义信息
            
        Returns:
            记忆ID
        """
        self._encode_count += 1
        
        # EC编码
        encoded = self.ec.encode(features)
        
        # DG模式分离
        separated, memory_id = self.dg.separate(encoded)
        
        # CA3存储
        self.ca3.store(
            memory_id,
            separated,
            timestamp_ms,
            semantic_info
        )
        
        # CA1时序编码
        self.ca1.encode_temporal(
            memory_id,
            timestamp_ms,
            self._last_memory_id
        )
        
        # 更新上一个记忆ID
        self._last_memory_id = memory_id
        
        return memory_id
    
    def recall_memories(
        self,
        cue: torch.Tensor,
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """
        召回记忆
        
        完整流程：EC编码 -> CA3召回 -> CA1门控
        
        Args:
            cue: 召回线索
            top_k: 返回数量
            
        Returns:
            记忆锚点列表
        """
        self._recall_count += 1
        
        # EC编码线索
        encoded_cue = self.ec.encode(cue)
        
        # CA3召回
        anchors = self.ca3.recall(encoded_cue, top_k)
        
        # 转换为字典格式
        result = []
        for anchor in anchors:
            result.append({
                'anchor_id': anchor.anchor_id,
                'memory_id': anchor.memory_unit.memory_id,
                'relevance_score': anchor.relevance_score,
                'gate_signal': anchor.gate_signal,
                'semantic_pointer': anchor.memory_unit.semantic_pointer,
                'timestamp_ms': anchor.memory_unit.timestamp_ms
            })
        
        return result
    
    def get_attention_gate(
        self,
        memory_anchors: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """
        获取注意力门控信号
        
        Args:
            memory_anchors: 记忆锚点列表
            
        Returns:
            门控信号
        """
        # 转换为MemoryAnchor对象
        anchors = []
        for anchor_dict in memory_anchors:
            memory_unit = MemoryUnit(
                memory_id=anchor_dict['memory_id'],
                timestamp_ms=anchor_dict['timestamp_ms'],
                feature_vector=anchor_dict['gate_signal'],
                semantic_pointer=anchor_dict.get('semantic_pointer', ''),
                temporal_skeleton=[],
                causal_links=[]
            )
            anchors.append(MemoryAnchor(
                anchor_id=anchor_dict['anchor_id'],
                memory_unit=memory_unit,
                relevance_score=anchor_dict['relevance_score'],
                gate_signal=anchor_dict['gate_signal']
            ))
        
        return self.ca1.generate_gate_signal(anchors)
    
    def offline_consolidation(self) -> Dict[str, Any]:
        """
        离线记忆巩固
        
        Returns:
            巩固结果
        """
        if not self.swr.check_idle():
            return {'status': 'not_idle'}
        
        # 开始回放
        replay_sequence = self.swr.start_replay(self.ca3, self.ca1)
        
        # 修剪记忆
        pruned_count = self.swr.prune_memories(self.ca3)
        
        # 结束回放
        self.swr.end_replay()
        
        return {
            'status': 'completed',
            'replay_count': len(replay_sequence),
            'pruned_count': pruned_count
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'encode_count': self._encode_count,
            'recall_count': self._recall_count,
            'memory_count': self.ca3.get_memory_count(),
            'is_replaying': self.swr._is_replaying,
            'replay_count': self.swr._replay_count
        }
    
    def clear(self):
        """清空所有记忆"""
        self.ca3._memory_store.clear()
        self.ca3._memory_order.clear()
        self.ca3._memory_ids.clear()
        self.ca3._feature_index = None
        self.ca1._temporal_chain.clear()
        self.ca1._causal_graph.clear()
        self._last_memory_id = None
        self._encode_count = 0
        self._recall_count = 0
