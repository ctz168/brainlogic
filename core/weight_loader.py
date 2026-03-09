"""
类人脑双系统全闭环AI架构 - 权重加载器
Human-Like Brain Dual-System Full-Loop AI Architecture - Weight Loader

实现从Qwen3.5-0.8B加载预训练权重到静态分支
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import json
import logging

logger = logging.getLogger(__name__)


class PretrainedWeightLoader:
    """
    预训练权重加载器
    
    从Qwen3.5-0.8B加载权重到类脑架构的静态分支
    动态分支保持随机初始化
    """
    
    # Qwen模型权重名称到类脑架构的映射
    WEIGHT_MAPPING = {
        # 嵌入层
        'model.embed_tokens.weight': 'embed_tokens.weight',
        
        # 注意力层
        'model.layers.{i}.self_attn.q_proj.weight': 'layers.{i}.self_attn.q_proj.weight',
        'model.layers.{i}.self_attn.k_proj.weight': 'layers.{i}.self_attn.k_proj.weight',
        'model.layers.{i}.self_attn.v_proj.weight': 'layers.{i}.self_attn.v_proj.weight',
        'model.layers.{i}.self_attn.o_proj.weight': 'layers.{i}.self_attn.o_proj.weight',
        
        # FFN层
        'model.layers.{i}.mlp.gate_proj.weight': 'layers.{i}.mlp.gate_proj.weight',
        'model.layers.{i}.mlp.up_proj.weight': 'layers.{i}.mlp.up_proj.weight',
        'model.layers.{i}.mlp.down_proj.weight': 'layers.{i}.mlp.down_proj.weight',
        
        # 层归一化
        'model.layers.{i}.input_layernorm.weight': 'layers.{i}.input_layernorm.weight',
        'model.layers.{i}.post_attention_layernorm.weight': 'layers.{i}.post_attention_layernorm.weight',
        
        # 最终层归一化
        'model.norm.weight': 'norm.weight',
        
        # 输出层
        'lm_head.weight': 'lm_head.weight',
    }
    
    def __init__(self, pretrained_path: str):
        """
        初始化加载器
        
        Args:
            pretrained_path: 预训练模型路径
        """
        self.pretrained_path = pretrained_path
        self._pretrained_weights: Dict[str, torch.Tensor] = {}
        self._config: Dict[str, Any] = {}
    
    def load_pretrained(self) -> Dict[str, torch.Tensor]:
        """
        加载预训练权重
        
        Returns:
            预训练权重字典
        """
        logger.info(f"加载预训练权重: {self.pretrained_path}")
        
        # 尝试加载safetensors格式
        safetensors_path = os.path.join(self.pretrained_path, 'model.safetensors')
        if os.path.exists(safetensors_path):
            return self._load_safetensors(safetensors_path)
        
        # 尝试加载pytorch格式
        pytorch_path = os.path.join(self.pretrained_path, 'pytorch_model.bin')
        if os.path.exists(pytorch_path):
            return self._load_pytorch(pytorch_path)
        
        # 尝试加载分片的pytorch格式
        index_path = os.path.join(self.pretrained_path, 'pytorch_model.bin.index.json')
        if os.path.exists(index_path):
            return self._load_sharded_pytorch(index_path)
        
        raise FileNotFoundError(f"未找到预训练权重文件: {self.pretrained_path}")
    
    def _load_safetensors(self, path: str) -> Dict[str, torch.Tensor]:
        """加载safetensors格式权重"""
        try:
            from safetensors.torch import load_file
            self._pretrained_weights = load_file(path)
            logger.info(f"成功加载safetensors权重: {len(self._pretrained_weights)} 个张量")
            return self._pretrained_weights
        except ImportError:
            logger.warning("safetensors库未安装，尝试其他格式")
            return {}
    
    def _load_pytorch(self, path: str) -> Dict[str, torch.Tensor]:
        """加载pytorch格式权重"""
        self._pretrained_weights = torch.load(path, map_location='cpu')
        logger.info(f"成功加载pytorch权重: {len(self._pretrained_weights)} 个张量")
        return self._pretrained_weights
    
    def _load_sharded_pytorch(self, index_path: str) -> Dict[str, torch.Tensor]:
        """加载分片的pytorch格式权重"""
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        weight_map = index.get('weight_map', {})
        shard_files = set(weight_map.values())
        
        for shard_file in shard_files:
            shard_path = os.path.join(self.pretrained_path, shard_file)
            shard_weights = torch.load(shard_path, map_location='cpu')
            self._pretrained_weights.update(shard_weights)
        
        logger.info(f"成功加载分片权重: {len(self._pretrained_weights)} 个张量")
        return self._pretrained_weights
    
    def load_config(self) -> Dict[str, Any]:
        """加载模型配置"""
        config_path = os.path.join(self.pretrained_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self._config = json.load(f)
        return self._config
    
    def map_weights(
        self,
        num_layers: int
    ) -> Dict[str, torch.Tensor]:
        """
        映射权重名称
        
        Args:
            num_layers: 层数
            
        Returns:
            映射后的权重字典
        """
        mapped_weights = {}
        
        for pretrained_name, weight in self._pretrained_weights.items():
            # 直接匹配
            if pretrained_name in self.WEIGHT_MAPPING:
                target_name = self.WEIGHT_MAPPING[pretrained_name]
                mapped_weights[target_name] = weight
                continue
            
            # 带层号的匹配
            for pattern, target_pattern in self.WEIGHT_MAPPING.items():
                if '{i}' in pattern:
                    for i in range(num_layers):
                        full_pattern = pattern.format(i=i)
                        if pretrained_name == full_pattern:
                            target_name = target_pattern.format(i=i)
                            mapped_weights[target_name] = weight
                            break
        
        logger.info(f"映射权重: {len(mapped_weights)} 个")
        return mapped_weights
    
    def load_to_model(
        self,
        model: nn.Module,
        strict: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        加载权重到模型
        
        Args:
            model: 目标模型
            strict: 是否严格匹配
            
        Returns:
            (missing_keys, unexpected_keys)
        """
        # 加载预训练权重
        self.load_pretrained()
        
        # 获取模型层数
        num_layers = len(model.layers)
        
        # 映射权重名称
        mapped_weights = self.map_weights(num_layers)
        
        # 获取模型当前状态
        model_state = model.state_dict()
        
        # 更新静态分支权重
        updated_keys = []
        for name, weight in mapped_weights.items():
            if name in model_state:
                if model_state[name].shape == weight.shape:
                    model_state[name] = weight
                    updated_keys.append(name)
                else:
                    logger.warning(f"形状不匹配: {name}, "
                                 f"模型: {model_state[name].shape}, "
                                 f"预训练: {weight.shape}")
        
        # 加载更新后的状态
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        
        logger.info(f"成功加载 {len(updated_keys)} 个静态权重")
        logger.info(f"缺失键: {len(missing_keys)}, 意外键: {len(unexpected_keys)}")
        
        return missing_keys, unexpected_keys


def load_qwen_weights_to_model(
    model: nn.Module,
    pretrained_path: str,
    freeze_static: bool = True
) -> Tuple[List[str], List[str]]:
    """
    从Qwen模型加载权重到类脑架构
    
    Args:
        model: 类脑架构模型
        pretrained_path: Qwen预训练模型路径
        freeze_static: 是否冻结静态权重
        
    Returns:
        (missing_keys, unexpected_keys)
    """
    loader = PretrainedWeightLoader(pretrained_path)
    missing_keys, unexpected_keys = loader.load_to_model(model)
    
    if freeze_static:
        model.freeze_static_weights()
        logger.info("已冻结静态权重")
    
    return missing_keys, unexpected_keys


class DynamicWeightInitializer:
    """
    动态权重初始化器
    
    为动态分支生成合适的初始权重
    """
    
    @staticmethod
    def initialize_dynamic_branch(
        model: nn.Module,
        init_method: str = 'small_normal',
        init_std: float = 0.02
    ):
        """
        初始化动态分支权重
        
        Args:
            model: 模型
            init_method: 初始化方法
            init_std: 初始化标准差
        """
        for name, param in model.named_parameters():
            if 'dynamic' in name.lower() and param.requires_grad:
                if init_method == 'small_normal':
                    nn.init.normal_(param, mean=0.0, std=init_std)
                elif init_method == 'xavier':
                    nn.init.xavier_normal_(param)
                elif init_method == 'kaiming':
                    nn.init.kaiming_normal_(param)
                elif init_method == 'zeros':
                    nn.init.zeros_(param)
        
        logger.info(f"动态权重初始化完成: {init_method}")


def convert_qwen_to_brainlike(
    pretrained_path: str,
    output_path: str,
    config: Any = None
):
    """
    将Qwen模型转换为类脑架构格式
    
    Args:
        pretrained_path: Qwen预训练模型路径
        output_path: 输出路径
        config: 类脑架构配置
    """
    from core.config import BrainLikeConfig
    from core.base_model import BrainLikeQwenModel
    
    if config is None:
        config = BrainLikeConfig()
    
    # 创建模型
    model = BrainLikeQwenModel(config)
    
    # 加载预训练权重
    load_qwen_weights_to_model(model, pretrained_path)
    
    # 初始化动态分支
    DynamicWeightInitializer.initialize_dynamic_branch(model)
    
    # 保存
    os.makedirs(output_path, exist_ok=True)
    model.save_weights(os.path.join(output_path, 'model_weights.pt'), save_static=True)
    config.save(os.path.join(output_path, 'config.json'))
    
    logger.info(f"模型转换完成: {output_path}")
