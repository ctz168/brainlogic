"""
类人脑双系统全闭环AI架构 - 100Hz高刷新推理引擎模块
Human-Like Brain Dual-System Full-Loop AI Architecture - 100Hz Refresh Engine Module

实现人脑gamma高频认知节律对齐的100Hz刷新推理引擎
- 10ms刷新周期
- 窄窗口O(1)注意力复杂度
- 单周期固定执行流
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from queue import Queue
from collections import deque

from core.config import BrainLikeConfig, RefreshConfig


class CyclePhase(Enum):
    """刷新周期阶段枚举"""
    INPUT_RECEIVE = "input_receive"  # 输入接收
    MEMORY_RECALL = "memory_recall"  # 记忆召回
    ATTENTION_GATE = "attention_gate"  # 注意力门控
    FORWARD_INFERENCE = "forward_inference"  # 前向推理
    OUTPUT_GENERATE = "output_generate"  # 输出生成
    STDP_UPDATE = "stdp_update"  # STDP更新
    MEMORY_ENCODE = "memory_encode"  # 记忆编码
    CYCLE_COMPLETE = "cycle_complete"  # 周期完成


@dataclass
class CycleContext:
    """单周期上下文"""
    cycle_id: int
    start_time_ms: float
    current_token_id: int
    input_token: Optional[torch.Tensor] = None
    context_tokens: List[torch.Tensor] = field(default_factory=list)
    memory_anchors: List[Dict] = field(default_factory=list)
    pixel_values: Optional[torch.Tensor] = None
    hidden_state: Optional[torch.Tensor] = None
    output_token: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    stdp_updates: Dict[str, torch.Tensor] = field(default_factory=dict)
    phase_timings: Dict[str, float] = field(default_factory=dict)
    end_time_ms: Optional[float] = None


@dataclass
class CycleResult:
    """单周期执行结果"""
    cycle_id: int
    output_token_id: int
    output_text: str
    cycle_time_ms: float
    phase_timings: Dict[str, float]
    memory_used: List[Dict]
    stddp_updates_count: int
    compute_ratio: float  # 相对原生模型的算力比例


class NarrowWindowAttention:
    """
    窄窗口注意力机制
    
    实现O(1)复杂度的注意力计算：
    - 每周期仅处理1-2个token
    - 仅从海马体调取1-2个相关记忆锚点
    - 其余上下文不进入当前周期计算流
    """
    
    def __init__(self, config: RefreshConfig):
        self.config = config
        self.window_size = config.max_context_per_cycle + 1
        self.memory_anchors_per_cycle = config.max_context_per_cycle
    
    def compute_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        memory_anchors: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算窄窗口注意力
        
        Args:
            query: 当前token的query [batch, heads, 1, head_dim]
            key_cache: K缓存 [batch, heads, seq_len, head_dim]
            value_cache: V缓存 [batch, heads, seq_len, head_dim]
            memory_anchors: 海马体记忆锚点
            
        Returns:
            output: 注意力输出
            attention_weights: 注意力权重
        """
        batch_size, num_heads, _, head_dim = query.shape
        seq_len = key_cache.shape[2]
        
        # 窄窗口选择：仅取最近的window_size个token
        if seq_len > self.window_size:
            window_start = seq_len - self.window_size
            key_window = key_cache[:, :, window_start:, :]
            value_window = value_cache[:, :, window_start:, :]
        else:
            key_window = key_cache
            value_window = value_cache
        
        # 添加记忆锚点到K, V
        if memory_anchors:
            anchor_keys = []
            anchor_values = []
            for anchor in memory_anchors[:self.memory_anchors_per_cycle]:
                if 'key' in anchor:
                    anchor_keys.append(anchor['key'])
                if 'value' in anchor:
                    anchor_values.append(anchor['value'])
            
            if anchor_keys:
                anchor_keys = torch.stack(anchor_keys, dim=2)  # [batch, heads, num_anchors, head_dim]
                anchor_values = torch.stack(anchor_values, dim=2)
                key_window = torch.cat([anchor_keys, key_window], dim=2)
                value_window = torch.cat([anchor_values, value_window], dim=2)
        
        # 计算注意力分数
        scale = 1.0 / (head_dim ** 0.5)
        attention_scores = torch.matmul(query, key_window.transpose(-2, -1)) * scale
        
        # Softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 应用注意力
        output = torch.matmul(attention_weights, value_window)
        
        return output, attention_weights
    
    def get_complexity(self, seq_len: int) -> int:
        """
        获取注意力计算复杂度
        
        窄窗口注意力复杂度固定为O(1)，与序列长度无关
        """
        return self.window_size * self.window_size


class RefreshEngine:
    """
    100Hz高刷新推理引擎
    
    实现人脑gamma节律对齐的10ms刷新周期：
    1. 输入token接收与特征提取
    2. 海马体记忆锚点调取与注意力门控加载
    3. 窄窗口上下文+当前token的模型前向推理
    4. 单周期输出结果生成
    5. 全链路STDP权重本地刷新
    6. 海马体情景记忆编码与更新
    7. 全局工作记忆压缩更新
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: BrainLikeConfig,
        hippocampus_module=None,
        stdp_module=None
    ):
        self.model = model
        self.config = config
        self.refresh_config = config.refresh
        
        # 子模块
        self.hippocampus = hippocampus_module
        self.stdp = stdp_module
        
        # 窄窗口注意力
        self.narrow_attention = NarrowWindowAttention(self.refresh_config)
        
        # 周期状态
        self._cycle_id = 0
        self._current_context: Optional[CycleContext] = None
        self._is_running = False
        
        # KV缓存
        self._key_cache: Optional[torch.Tensor] = None
        self._value_cache: Optional[torch.Tensor] = None
        
        # 统计信息
        self._cycle_times: deque = deque(maxlen=1000)
        self._phase_timings: Dict[str, List[float]] = {
            phase.value: [] for phase in CyclePhase
        }
        
        # 回调函数
        self._on_cycle_complete: Optional[Callable] = None
    
    def start(self):
        """启动刷新引擎"""
        self._is_running = True
        self._cycle_id = 0
    
    def stop(self):
        """停止刷新引擎"""
        self._is_running = False
    
    def run_cycle(
        self,
        input_token: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        context_tokens: Optional[List[torch.Tensor]] = None
    ) -> CycleResult:
        """
        执行单个刷新周期
        
        Args:
            input_token: 输入token ID
            context_tokens: 上下文token列表
            
        Returns:
            CycleResult: 周期执行结果
        """
        cycle_start = time.time() * 1000
        
        # 初始化周期上下文
        self._current_context = CycleContext(
            cycle_id=self._cycle_id,
            start_time_ms=cycle_start,
            current_token_id=input_token.item() if isinstance(input_token, torch.Tensor) else input_token,
            input_token=input_token,
            pixel_values=pixel_values,
            context_tokens=context_tokens or []
        )
        
        try:
            # 阶段1：输入token接收与特征提取
            self._phase_input_receive()
            
            # 阶段2：海马体记忆锚点调取与注意力门控加载
            self._phase_memory_recall()
            
            # 阶段3：窄窗口上下文+当前token的模型前向推理
            self._phase_forward_inference()
            
            # 阶段4：单周期输出结果生成
            self._phase_output_generate()
            
            # 阶段5：全链路STDP权重本地刷新
            self._phase_stdp_update()
            
            # 阶段6：海马体情景记忆编码与更新
            self._phase_memory_encode()
            
            # 阶段7：全局工作记忆压缩更新
            self._phase_working_memory_update()
            
        except Exception as e:
            print(f"Cycle {self._cycle_id} error: {e}")
        
        # 记录周期结束时间
        cycle_end = time.time() * 1000
        self._current_context.end_time_ms = cycle_end
        cycle_time = cycle_end - cycle_start
        
        # 更新统计
        self._cycle_times.append(cycle_time)
        for phase, timing in self._current_context.phase_timings.items():
            self._phase_timings[phase].append(timing)
        
        # 构建结果
        result = CycleResult(
            cycle_id=self._cycle_id,
            output_token_id=self._current_context.output_token.item() if self._current_context.output_token is not None else 0,
            output_text="",  # 需要tokenizer解码
            cycle_time_ms=cycle_time,
            phase_timings=self._current_context.phase_timings.copy(),
            memory_used=self._current_context.memory_anchors,
            stddp_updates_count=len(self._current_context.stdp_updates),
            compute_ratio=self._calculate_compute_ratio()
        )
        
        # 回调
        if self._on_cycle_complete:
            self._on_cycle_complete(result)
        
        self._cycle_id += 1
        
        return result
    
    def _phase_input_receive(self):
        """阶段1：输入token接收与特征提取"""
        phase_start = time.time() * 1000
        
        # 特征提取（复用模型嵌入层）
        if hasattr(self.model, 'embed_tokens'):
            self._current_context.hidden_state = self.model.embed_tokens(
                self._current_context.input_token
            )
        
        self._current_context.phase_timings['input_receive'] = time.time() * 1000 - phase_start
    
    def _phase_memory_recall(self):
        """阶段2：海马体记忆锚点调取与注意力门控加载"""
        phase_start = time.time() * 1000
        
        if self.hippocampus is not None:
            # 从海马体召回相关记忆锚点
            current_feature = self._current_context.hidden_state
            if current_feature is not None:
                memory_anchors = self.hippocampus.recall_memories(
                    current_feature,
                    top_k=self.refresh_config.max_context_per_cycle
                )
                self._current_context.memory_anchors = memory_anchors
        
        self._current_context.phase_timings['memory_recall'] = time.time() * 1000 - phase_start
    
    def _phase_forward_inference(self):
        """阶段3：窄窗口上下文+当前token的模型前向推理"""
        phase_start = time.time() * 1000
        
        # 准备输入
        input_ids = self._current_context.input_token.unsqueeze(0)
        
        # 执行前向推理
        with torch.no_grad():
            logits, features, dynamic_weights = self.model(
                input_ids,
                pixel_values=self._current_context.pixel_values,
                memory_anchors=self._current_context.memory_anchors
            )
        
        self._current_context.hidden_state = logits
        self._current_context.stdp_updates = dynamic_weights
        
        self._current_context.phase_timings['forward_inference'] = time.time() * 1000 - phase_start
    
    def _phase_output_generate(self):
        """阶段4：单周期输出结果生成"""
        phase_start = time.time() * 1000
        
        if self._current_context.hidden_state is not None:
            # 获取logits
            logits = self._current_context.hidden_state[:, -1, :]
            
            # 采样输出token
            probs = torch.softmax(logits, dim=-1)
            output_token = torch.argmax(probs, dim=-1)
            
            self._current_context.output_token = output_token
        
        self._current_context.phase_timings['output_generate'] = time.time() * 1000 - phase_start
    
    def _phase_stdp_update(self):
        """阶段5：全链路STDP权重本地刷新"""
        phase_start = time.time() * 1000
        
        if self.stdp is not None and self._current_context.stdp_updates:
            # 应用STDP权重更新
            for name, update in self._current_context.stdp_updates.items():
                self.stdp.apply_update(name, update)
        
        self._current_context.phase_timings['stdp_update'] = time.time() * 1000 - phase_start
    
    def _phase_memory_encode(self):
        """阶段6：海马体情景记忆编码与更新"""
        phase_start = time.time() * 1000
        
        if self.hippocampus is not None:
            # 编码当前周期信息到海马体
            self.hippocampus.encode_episode(
                self._current_context.hidden_state,
                self._current_context.start_time_ms,
                {
                    'cycle_id': self._cycle_id,
                    'input_token': self._current_context.current_token_id,
                    'output_token': self._current_context.output_token.item() if self._current_context.output_token is not None else 0
                }
            )
        
        self._current_context.phase_timings['memory_encode'] = time.time() * 1000 - phase_start
    
    def _phase_working_memory_update(self):
        """阶段7：全局工作记忆压缩更新"""
        phase_start = time.time() * 1000
        
        # 更新KV缓存
        # 这里简化处理，实际需要更复杂的缓存管理
        
        self._current_context.phase_timings['working_memory_update'] = time.time() * 1000 - phase_start
    
    def _calculate_compute_ratio(self) -> float:
        """计算相对原生模型的算力比例"""
        # 窄窗口注意力复杂度固定
        window_ops = self.narrow_attention.get_complexity(0)
        
        # 原生Transformer复杂度（假设序列长度为512）
        native_ops = 512 * 512
        
        return window_ops / native_ops
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        avg_cycle_time = sum(self._cycle_times) / len(self._cycle_times) if self._cycle_times else 0
        
        phase_avg = {}
        for phase, timings in self._phase_timings.items():
            if timings:
                phase_avg[phase] = sum(timings) / len(timings)
        
        return {
            'total_cycles': self._cycle_id,
            'is_running': self._is_running,
            'average_cycle_time_ms': avg_cycle_time,
            'target_cycle_time_ms': self.refresh_config.refresh_period_ms,
            'cycle_time_compliance': avg_cycle_time <= self.refresh_config.refresh_period_ms,
            'phase_average_times_ms': phase_avg,
            'compute_ratio': self._calculate_compute_ratio()
        }
    
    def set_cycle_callback(self, callback: Callable):
        """设置周期完成回调"""
        self._on_cycle_complete = callback


class ContinuousRefreshEngine(RefreshEngine):
    """
    连续刷新引擎
    
    支持连续输入流的实时处理
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_queue: Queue = Queue()
        self._output_queue: Queue = Queue()
        self._worker_thread: Optional[threading.Thread] = None
    
    def start_continuous(self):
        """启动连续处理模式"""
        self.start()
        self._worker_thread = threading.Thread(target=self._continuous_loop)
        self._worker_thread.daemon = True
        self._worker_thread.start()
    
    def stop_continuous(self):
        """停止连续处理模式"""
        self.stop()
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
    
    def submit_input(self, token: torch.Tensor):
        """提交输入token"""
        self._input_queue.put(token)
    
    def get_output(self, timeout: float = 1.0) -> Optional[CycleResult]:
        """获取输出结果"""
        try:
            return self._output_queue.get(timeout=timeout)
        except:
            return None
    
    def _continuous_loop(self):
        """连续处理循环"""
        while self._is_running:
            try:
                # 从队列获取输入
                input_token = self._input_queue.get(timeout=0.01)
                
                # 执行刷新周期
                result = self.run_cycle(input_token)
                
                # 输出结果
                self._output_queue.put(result)
                
            except:
                # 队列为空，继续等待
                continue


class BatchRefreshEngine(RefreshEngine):
    """
    批量刷新引擎
    
    支持批量token的高效处理
    """
    
    def run_batch(
        self,
        input_tokens: List[torch.Tensor]
    ) -> List[CycleResult]:
        """
        批量执行刷新周期
        
        Args:
            input_tokens: 输入token列表
            
        Returns:
            周期结果列表
        """
        results = []
        
        for token in input_tokens:
            result = self.run_cycle(token)
            results.append(result)
        
        return results
    
    def run_sequence(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100
    ) -> List[CycleResult]:
        """
        执行序列生成
        
        Args:
            input_ids: 输入序列
            max_new_tokens: 最大生成token数
            
        Returns:
            所有周期的结果列表
        """
        results = []
        
        # 处理输入序列
        for i in range(input_ids.shape[1]):
            token = input_ids[:, i]
            result = self.run_cycle(token)
            results.append(result)
        
        # 生成新token
        for _ in range(max_new_tokens):
            if self._current_context and self._current_context.output_token is not None:
                result = self.run_cycle(self._current_context.output_token)
                results.append(result)
                
                # 检查是否生成结束符
                if result.output_token_id == 2:  # 假设2是EOS token
                    break
        
        return results
