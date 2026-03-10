"""
类人脑双系统全闭环AI架构 - 全局配置模块
Human-Like Brain Dual-System Full-Loop AI Architecture - Global Configuration

定义所有模块的全局配置参数，包括：
- 模型基础配置
- 刷新周期配置
- STDP学习参数
- 海马体记忆参数
- 端侧部署约束
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class ModelMode(Enum):
    """模型运行模式枚举"""
    GENERATION = "generation"  # 生成模式
    VERIFICATION = "verification"  # 验证模式
    JUDGMENT = "judgment"  # 评判模式


class OptimizationMode(Enum):
    """优化模式枚举"""
    SELF_GENERATION = "self_generation"  # 自生成组合输出
    SELF_PLAY = "self_play"  # 自博弈竞争优化
    SELF_JUDGMENT = "self_judgment"  # 自双输出+自评判


@dataclass
class RefreshConfig:
    """100Hz高刷新引擎配置"""
    # 刷新周期配置
    refresh_rate: int = 100  # Hz，刷新频率
    refresh_period_ms: float = 10.0  # ms，刷新周期
    
    # 窄窗口配置
    tokens_per_cycle: int = 1  # 每周期处理token数
    max_context_per_cycle: int = 2  # 每周期最大上下文token数
    
    # 算力约束
    max_compute_ratio: float = 0.1  # 单周期算力不超过原生模型10%
    attention_complexity: str = "O(1)"  # 注意力复杂度必须固定为O(1)
    
    # 执行流配置
    enable_feature_extraction: bool = True
    enable_hippocampus_gate: bool = True
    enable_stdp_update: bool = True
    enable_memory_encoding: bool = True


@dataclass
class STDPConfig:
    """STDP时序可塑性配置"""
    # STDP核心参数
    alpha: float = 0.01  # LTP学习率（权重增强）
    beta: float = 0.008  # LTD学习率（权重减弱）
    
    # 权重约束
    weight_min: float = -1.0  # 权重下限
    weight_max: float = 1.0  # 权重上限
    update_threshold: float = 0.001  # 更新阈值
    
    # 时序窗口
    timing_window_ms: float = 40.0  # STDP时序窗口（毫秒）
    pre_post_delay_ms: float = 10.0  # 前后神经元激活延迟
    
    # 更新节点开关
    enable_attention_stdp: bool = True  # 注意力层STDP
    enable_ffn_stdp: bool = True  # FFN层STDP
    enable_self_judgment_stdp: bool = True  # 自评判STDP
    enable_hippocampus_stdp: bool = True  # 海马体门控STDP
    
    # 自评判周期
    self_judgment_interval: int = 10  # 每10个刷新周期执行一次自评判


@dataclass
class HippocampusConfig:
    """海马体记忆系统配置"""
    # EC内嗅皮层配置
    ec_feature_dim: int = 64  # 特征编码维度
    ec_sparse_ratio: float = 0.3  # 稀疏编码比例
    
    # DG齿状回配置
    dg_pattern_separation_strength: float = 0.8  # 模式分离强度
    dg_orthogonal_dim: int = 128  # 正交化维度
    
    # CA3区配置
    ca3_memory_capacity: int = 10000  # 情景记忆容量
    ca3_recall_top_k: int = 2  # 每周期召回记忆数
    ca3_completion_threshold: float = 0.7  # 模式补全阈值
    
    # CA1区配置
    ca1_timestamp_precision_ms: float = 10.0  # 时间戳精度
    ca1_gate_strength: float = 0.6  # 注意力门控强度
    
    # SWR尖波涟漪配置
    swr_enabled: bool = True  # 是否启用离线回放
    swr_idle_threshold_minutes: float = 5.0  # 空闲触发阈值
    swr_replay_frequency: int = 10  # 回放频率（次/分钟）
    swr_consolidation_ratio: float = 0.3  # 记忆巩固比例
    
    # 内存约束
    max_memory_mb: float = 2.0  # 最大内存占用（MB）
    circular_buffer_enabled: bool = True  # 启用循环缓存


@dataclass
class WeightSplitConfig:
    """权重双轨拆分配置"""
    # 拆分比例
    static_ratio: float = 0.9  # 静态基础权重比例（90%）
    dynamic_ratio: float = 0.1  # STDP动态增量权重比例（10%）
    
    # 初始化配置
    dynamic_init_mean: float = 0.0  # 动态权重初始化均值
    dynamic_init_std: float = 0.02  # 动态权重初始化标准差
    
    # 冻结配置
    freeze_static_weights: bool = True  # 永久冻结静态权重
    allow_dynamic_update: bool = True  # 允许动态权重更新
    
    # 量化配置
    quantization_bits: int = 4  # INT4量化
    max_vram_mb: float = 420.0  # 最大显存占用（MB）


@dataclass
class OptimizationConfig:
    """自闭环优化配置"""
    # 模式1：自生成组合输出
    self_gen_temperature_range: tuple = (0.7, 0.9)  # 温度范围
    self_gen_candidates: int = 2  # 候选数量
    self_gen_voting_method: str = "weighted_consistency"  # 投票方法
    
    # 模式2：自博弈竞争优化
    self_play_max_iterations: int = 5  # 最大迭代次数
    self_play_convergence_threshold: float = 0.95  # 收敛阈值
    
    # 模式3：自双输出+自评判
    self_judgment_dimensions: List[str] = field(default_factory=lambda: [
        "factual_accuracy",  # 事实准确性
        "logical_completeness",  # 逻辑完整性
        "semantic_coherence",  # 语义连贯性
        "instruction_following"  # 指令遵循度
    ])
    self_judgment_max_score: int = 10  # 每维度最高分
    self_judgment_total_score: int = 40  # 总分
    
    # 自动模式切换关键词
    self_play_keywords: List[str] = field(default_factory=lambda: [
        "计算", "推理", "代码", "逻辑", "证明", "数学"
    ])
    self_judgment_keywords: List[str] = field(default_factory=lambda: [
        "方案", "建议", "决策", "最优", "选择", "评估"
    ])


@dataclass
class TrainingConfig:
    """训练配置"""
    # 预适配微调配置
    pre_adapt_lr: float = 1e-5  # 学习率
    pre_adapt_batch_size: int = 8  # 批大小
    pre_adapt_epochs: int = 3  # 训练轮数
    pre_adapt_optimizer: str = "AdamW"  # 优化器
    
    # 在线学习配置
    online_learning_enabled: bool = True  # 启用在线学习
    online_compute_ratio: float = 0.02  # 算力开销不超过2%
    
    # 离线巩固配置
    offline_consolidation_enabled: bool = True  # 启用离线巩固
    offline_idle_threshold_minutes: float = 5.0  # 空闲触发阈值


@dataclass
class EvaluationConfig:
    """测评配置"""
    # 海马体记忆能力测评（权重40%）
    hippocampus_weight: float = 0.4
    recall_accuracy_threshold: float = 0.95  # 召回准确率阈值
    confusion_rate_threshold: float = 0.03  # 混淆率阈值
    long_sequence_tokens: int = 100000  # 长序列测试token数
    
    # 基础能力对标测评（权重20%）
    baseline_weight: float = 0.2
    baseline_performance_ratio: float = 0.95  # 不低于原生模型95%
    
    # 逻辑推理能力测评（权重20%）
    reasoning_weight: float = 0.2
    reasoning_improvement_ratio: float = 0.6  # 超过原生模型60%
    
    # 端侧性能测评（权重10%）
    edge_weight: float = 0.1
    
    # 自闭环优化能力测评（权重10%）
    self_correction_weight: float = 0.1
    self_correction_threshold: float = 0.9  # 自纠错准确率阈值
    hallucination_reduction: float = 0.7  # 幻觉率下降70%


@dataclass
class BrainLikeConfig:
    """类人脑双系统全闭环AI架构全局配置"""
    # 基础模型配置
    base_model_name: str = "Qwen/Qwen3.5-0.8B-Base"  # 底座模型名称
    model_hidden_size: int = 1024  # 隐藏层维度
    model_num_layers: int = 24  # 层数
    model_num_heads: int = 16  # 注意力头数
    model_vocab_size: int = 151936  # 词表大小
    
    # 子配置
    refresh: RefreshConfig = field(default_factory=RefreshConfig)
    stdp: STDPConfig = field(default_factory=STDPConfig)
    hippocampus: HippocampusConfig = field(default_factory=HippocampusConfig)
    weight_split: WeightSplitConfig = field(default_factory=WeightSplitConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # 端侧部署配置
    target_device: str = "edge"  # 目标设备：edge/desktop/server
    enable_quantization: bool = True  # 启用量化
    offline_mode: bool = True  # 离线模式
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'base_model_name': self.base_model_name,
            'model_hidden_size': self.model_hidden_size,
            'model_num_layers': self.model_num_layers,
            'model_num_heads': self.model_num_heads,
            'model_vocab_size': self.model_vocab_size,
            'refresh': {
                'refresh_rate': self.refresh.refresh_rate,
                'refresh_period_ms': self.refresh.refresh_period_ms,
                'tokens_per_cycle': self.refresh.tokens_per_cycle,
                'max_context_per_cycle': self.refresh.max_context_per_cycle,
                'max_compute_ratio': self.refresh.max_compute_ratio,
            },
            'stdp': {
                'alpha': self.stdp.alpha,
                'beta': self.stdp.beta,
                'weight_min': self.stdp.weight_min,
                'weight_max': self.stdp.weight_max,
            },
            'hippocampus': {
                'ec_feature_dim': self.hippocampus.ec_feature_dim,
                'ca3_memory_capacity': self.hippocampus.ca3_memory_capacity,
                'max_memory_mb': self.hippocampus.max_memory_mb,
            },
            'weight_split': {
                'static_ratio': self.weight_split.static_ratio,
                'dynamic_ratio': self.weight_split.dynamic_ratio,
                'max_vram_mb': self.weight_split.max_vram_mb,
            }
        }
    
    def save(self, path: str):
        """保存配置到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'BrainLikeConfig':
        """从文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


# 默认配置实例
DEFAULT_CONFIG = BrainLikeConfig()
