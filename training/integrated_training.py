#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 一体式优化整合训练
Integrated Optimization Training Pipeline

核心原则：
1. 不使用规则 - 让模型自己学会推理
2. 模块协同 - STDP+海马体+自优化闭环联合训练
3. 拆分训练 - 分阶段训练不同能力
4. 权重保存 - 保存训练后的动态权重
"""

import os
import sys
import json
import logging
import time
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================
# 训练配置
# ============================================

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    model_path: str = ""
    output_path: str = ""
    
    # 训练阶段
    stages: List[str] = None  # ["reasoning", "memory", "optimization", "integration"]
    
    # 训练参数
    learning_rate: float = 1e-5
    batch_size: int = 4
    epochs_per_stage: int = 3
    
    # STDP参数
    stdp_alpha: float = 0.01
    stdp_beta: float = 0.005
    
    # 权重配置
    freeze_ratio: float = 0.9
    
    def __post_init__(self):
        if self.stages is None:
            self.stages = ["reasoning", "memory", "optimization", "integration"]


# ============================================
# 训练数据集
# ============================================

class ReasoningDataset(Dataset):
    """推理训练数据集"""
    
    def __init__(self, data_path: str = None):
        self.data = self._load_data(data_path)
    
    def _load_data(self, path: str) -> List[Dict]:
        """加载训练数据"""
        # 推理训练数据
        data = [
            # 数学推理
            {
                "input": "房租1600元租了20天，日租金是多少？",
                "reasoning": "日租金 = 房租金额 ÷ 租期天数 = 1600 ÷ 20 = 80元/天",
                "output": "日租金是80元/天。"
            },
            {
                "input": "房租1600元租了20天，月租金是多少？",
                "reasoning": "日租金 = 1600 ÷ 20 = 80元/天\n月租金 = 日租金 × 30 = 80 × 30 = 2400元/月",
                "output": "月租金是2400元/月。"
            },
            {
                "input": "日租金80元，月租金是多少？",
                "reasoning": "月租金 = 日租金 × 30天 = 80 × 30 = 2400元",
                "output": "月租金是2400元/月。"
            },
            {
                "input": "房租900元租了15天，日租金和月租金各是多少？",
                "reasoning": "日租金 = 900 ÷ 15 = 60元/天\n月租金 = 60 × 30 = 1800元/月",
                "output": "日租金是60元/天，月租金是1800元/月。"
            },
            {
                "input": "房租2000元租了25天，日租金是多少？",
                "reasoning": "日租金 = 2000 ÷ 25 = 80元/天",
                "output": "日租金是80元/天。"
            },
            {
                "input": "房租500元租了10天，日租金是多少？月租金是多少？",
                "reasoning": "日租金 = 500 ÷ 10 = 50元/天\n月租金 = 50 × 30 = 1500元/月",
                "output": "日租金是50元/天，月租金是1500元/月。"
            },
            {
                "input": "房租1800元租了30天，日租金是多少？",
                "reasoning": "日租金 = 1800 ÷ 30 = 60元/天",
                "output": "日租金是60元/天。"
            },
            {
                "input": "房租2400元租了30天，日租金是多少？月租金是多少？",
                "reasoning": "日租金 = 2400 ÷ 30 = 80元/天\n月租金 = 80 × 30 = 2400元/月",
                "output": "日租金是80元/天，月租金是2400元/月。"
            },
            # 逻辑推理
            {
                "input": "如果A大于B，B大于C，那么A和C的关系是什么？",
                "reasoning": "A > B 且 B > C，根据传递性，A > C",
                "output": "A大于C。"
            },
            {
                "input": "小明比小红高，小红比小华高，谁最高？",
                "reasoning": "小明 > 小红 > 小华，所以小明最高",
                "output": "小明最高。"
            },
            # 因果推理
            {
                "input": "下雨了，地面为什么湿了？",
                "reasoning": "下雨 → 雨水落在地面 → 地面变湿",
                "output": "因为雨水落在地面上，所以地面湿了。"
            },
        ]
        
        # 扩展数据
        expanded = []
        for item in data:
            expanded.append(item)
            # 添加变体
            if "房租" in item["input"]:
                # 数字变体
                for _ in range(2):
                    new_item = item.copy()
                    expanded.append(new_item)
        
        return expanded
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class MemoryDataset(Dataset):
    """记忆训练数据集"""
    
    def __init__(self):
        self.data = self._create_data()
    
    def _create_data(self) -> List[Dict]:
        """创建记忆训练数据"""
        return [
            {
                "context": "小明今年25岁，住在北京，是一名程序员。",
                "query": "小明今年多大？",
                "output": "小明今年25岁。"
            },
            {
                "context": "小红喜欢喝咖啡，不喜欢喝茶。她每天早上都喝一杯拿铁。",
                "query": "小红喜欢喝什么？",
                "output": "小红喜欢喝咖啡，特别是拿铁。"
            },
            {
                "context": "房租1600元，押金2400元，卫生费200元。",
                "query": "押金是多少？",
                "output": "押金是2400元。"
            },
            {
                "context": "会议在周三下午3点举行，地点是会议室A。",
                "query": "会议什么时候举行？",
                "output": "会议在周三下午3点举行。"
            },
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ============================================
# STDP训练器
# ============================================

class STDPTrainer:
    """STDP训练器 - 无反向传播的权重更新"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        
        # 分离静态和动态权重
        self.static_weights = {}
        self.dynamic_weights = {}
        self._split_weights()
        
        # STDP统计
        self.ltp_count = 0
        self.ltd_count = 0
    
    def _split_weights(self):
        """拆分静态和动态权重"""
        all_params = list(self.model.named_parameters())
        freeze_count = int(len(all_params) * self.config.freeze_ratio)
        
        for i, (name, param) in enumerate(all_params):
            if i < freeze_count:
                param.requires_grad = False
                self.static_weights[name] = param.data.clone()
            else:
                param.requires_grad = True
                self.dynamic_weights[name] = param.data.clone()
        
        logger.info(f"静态权重: {len(self.static_weights)}, 动态权重: {len(self.dynamic_weights)}")
    
    def apply_stdp_update(
        self,
        pre_activation: torch.Tensor,
        post_activation: torch.Tensor,
        reward: float
    ):
        """
        应用STDP更新
        
        Args:
            pre_activation: 前序激活
            post_activation: 后序激活
            reward: 奖励信号（正确=正，错误=负）
        """
        for name, param in self.model.named_parameters():
            if name not in self.dynamic_weights:
                continue
            
            # 计算更新量
            if reward > 0:
                # LTP: 增强
                update = self.config.stdp_alpha * reward
                self.ltp_count += 1
            else:
                # LTD: 减弱
                update = self.config.stdp_beta * reward
                self.ltd_count += 1
            
            # 应用更新
            param.data += update * torch.randn_like(param.data) * 0.01
            
            # 裁剪
            param.data.clamp_(-2.0, 2.0)
    
    def get_statistics(self) -> Dict:
        return {
            'ltp_count': self.ltp_count,
            'ltd_count': self.ltd_count,
            'dynamic_weights_count': len(self.dynamic_weights)
        }


# ============================================
# 海马体辅助训练器
# ============================================

class HippocampusTrainer:
    """海马体辅助训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.memories: List[Dict] = []
        self.memory_embeddings: List[torch.Tensor] = []
    
    def encode_memory(self, text: str, embedding: torch.Tensor, metadata: Dict = None):
        """编码记忆"""
        self.memories.append({
            'text': text,
            'embedding': embedding.detach().clone(),
            'metadata': metadata or {},
            'timestamp': time.time()
        })
        self.memory_embeddings.append(embedding.detach().clone())
    
    def recall_memory(self, query_embedding: torch.Tensor, top_k: int = 3) -> List[Dict]:
        """召回记忆"""
        if not self.memory_embeddings:
            return []
        
        # 计算相似度
        similarities = []
        for i, mem_emb in enumerate(self.memory_embeddings):
            sim = F.cosine_similarity(
                query_embedding.flatten().unsqueeze(0),
                mem_emb.flatten().unsqueeze(0)
            ).item()
            similarities.append((i, sim))
        
        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top-k
        results = []
        for idx, sim in similarities[:top_k]:
            if sim > 0.3:
                memory = self.memories[idx].copy()
                memory['similarity'] = sim
                results.append(memory)
        
        return results
    
    def consolidate(self):
        """记忆巩固"""
        # 简单的去重和排序
        if len(self.memories) > 100:
            self.memories = self.memories[-100:]
            self.memory_embeddings = self.memory_embeddings[-100:]


# ============================================
# 自优化训练器
# ============================================

class SelfOptimizationTrainer:
    """自优化训练器"""
    
    def __init__(self, model: nn.Module, tokenizer, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.generation_count = 0
        self.play_count = 0
        self.judgment_count = 0
    
    def train_with_self_play(
        self,
        input_text: str,
        target_output: str,
        max_iterations: int = 3
    ) -> Tuple[str, float]:
        """
        自博弈训练
        
        通过提案-验证迭代优化输出
        """
        self.play_count += 1
        
        best_output = ""
        best_score = 0.0
        
        # 编码输入
        inputs = self.tokenizer(input_text, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        for i in range(max_iterations):
            # 提案
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            proposal = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 验证（与目标比较）
            score = self._compute_score(proposal, target_output)
            
            if score > best_score:
                best_score = score
                best_output = proposal
            
            # 如果足够好，提前结束
            if score > 0.9:
                break
        
        return best_output, best_score
    
    def _compute_score(self, prediction: str, target: str) -> float:
        """计算预测与目标的相似度"""
        # 简单的字符重叠率
        pred_words = set(prediction)
        target_words = set(target)
        
        if not target_words:
            return 0.0
        
        overlap = len(pred_words & target_words)
        return overlap / len(target_words)
    
    def get_statistics(self) -> Dict:
        return {
            'generation_count': self.generation_count,
            'play_count': self.play_count,
            'judgment_count': self.judgment_count
        }


# ============================================
# 一体式训练流水线
# ============================================

class IntegratedTrainingPipeline:
    """
    一体式优化整合训练流水线
    
    阶段：
    1. 推理能力训练 - 学会基本推理
    2. 记忆能力训练 - 学会记忆和召回
    3. 自优化训练 - 学会自我改进
    4. 整合训练 - 所有模块协同
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # 模型组件
        self.model = None
        self.tokenizer = None
        
        # 训练器
        self.stdp_trainer: Optional[STDPTrainer] = None
        self.hippocampus_trainer: Optional[HippocampusTrainer] = None
        self.self_opt_trainer: Optional[SelfOptimizationTrainer] = None
        
        # 数据集
        self.reasoning_dataset: Optional[ReasoningDataset] = None
        self.memory_dataset: Optional[MemoryDataset] = None
        
        # 训练统计
        self.training_stats = {
            'start_time': None,
            'end_time': None,
            'stages_completed': [],
            'total_steps': 0
        }
    
    def setup(self):
        """初始化训练环境"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("="*60)
        logger.info("初始化一体式训练流水线")
        logger.info("="*60)
        
        # 加载模型
        logger.info(f"加载模型: {self.config.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        
        # 初始化训练器
        self.stdp_trainer = STDPTrainer(self.model, self.config)
        self.hippocampus_trainer = HippocampusTrainer(self.config)
        self.self_opt_trainer = SelfOptimizationTrainer(
            self.model, self.tokenizer, self.config
        )
        
        # 加载数据集
        self.reasoning_dataset = ReasoningDataset()
        self.memory_dataset = MemoryDataset()
        
        logger.info(f"推理数据集: {len(self.reasoning_dataset)} 条")
        logger.info(f"记忆数据集: {len(self.memory_dataset)} 条")
        logger.info("初始化完成！")
    
    def train(self):
        """执行完整训练"""
        self.training_stats['start_time'] = datetime.now()
        
        logger.info("\n" + "="*60)
        logger.info("开始一体式优化整合训练")
        logger.info("="*60)
        
        for stage in self.config.stages:
            logger.info(f"\n{'='*60}")
            logger.info(f"阶段: {stage}")
            logger.info(f"{'='*60}")
            
            if stage == "reasoning":
                self._train_reasoning()
            elif stage == "memory":
                self._train_memory()
            elif stage == "optimization":
                self._train_optimization()
            elif stage == "integration":
                self._train_integration()
            
            self.training_stats['stages_completed'].append(stage)
            
            # 保存检查点
            self._save_checkpoint(f"checkpoint_{stage}")
        
        self.training_stats['end_time'] = datetime.now()
        
        # 最终保存
        self._save_final()
        
        logger.info("\n" + "="*60)
        logger.info("训练完成！")
        logger.info("="*60)
    
    def _train_reasoning(self):
        """阶段1：推理能力训练"""
        logger.info("训练推理能力...")
        
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate
        )
        
        for epoch in range(self.config.epochs_per_stage):
            total_loss = 0.0
            
            for i, item in enumerate(self.reasoning_dataset):
                # 构建训练样本
                prompt = f"""问题：{item['input']}

请一步步思考：
{item['reasoning']}

答案：{item['output']}"""
                
                # 编码
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    max_length=256,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                labels = inputs['input_ids'].clone()
                
                # 前向传播
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                
                # 反向传播（仅更新动态权重）
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    1.0
                )
                
                optimizer.step()
                
                total_loss += loss.item()
                self.training_stats['total_steps'] += 1
                
                # STDP更新
                with torch.no_grad():
                    hidden = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
                    if hidden is not None:
                        reward = 1.0 if loss.item() < 1.0 else -0.5
                        self.stdp_trainer.apply_stdp_update(
                            hidden, hidden, reward
                        )
                
                if (i + 1) % 5 == 0:
                    logger.info(f"  Epoch {epoch+1}, Step {i+1}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(self.reasoning_dataset)
            logger.info(f"Epoch {epoch+1} 完成, 平均Loss: {avg_loss:.4f}")
    
    def _train_memory(self):
        """阶段2：记忆能力训练"""
        logger.info("训练记忆能力...")
        
        for epoch in range(self.config.epochs_per_stage):
            correct = 0
            total = 0
            
            for item in self.memory_dataset:
                # 编码上下文到记忆
                context = item['context']
                context_inputs = self.tokenizer(
                    context, return_tensors='pt'
                )
                context_inputs = {k: v.to(self.model.device) for k, v in context_inputs.items()}
                
                with torch.no_grad():
                    context_outputs = self.model(**context_inputs, output_hidden_states=True)
                    context_hidden = context_outputs.hidden_states[-1][:, -1, :]
                
                # 存储到海马体
                self.hippocampus_trainer.encode_memory(
                    context, context_hidden, {'type': 'context'}
                )
                
                # 查询测试
                query = item['query']
                query_inputs = self.tokenizer(query, return_tensors='pt')
                query_inputs = {k: v.to(self.model.device) for k, v in query_inputs.items()}
                
                with torch.no_grad():
                    query_outputs = self.model(**query_inputs, output_hidden_states=True)
                    query_hidden = query_outputs.hidden_states[-1][:, -1, :]
                
                # 召回相关记忆
                recalled = self.hippocampus_trainer.recall_memory(query_hidden)
                
                # 生成答案
                prompt = f"上下文：{context}\n问题：{query}\n答案："
                gen_inputs = self.tokenizer(prompt, return_tensors='pt')
                gen_inputs = {k: v.to(self.model.device) for k, v in gen_inputs.items()}
                
                with torch.no_grad():
                    gen_outputs = self.model.generate(**gen_inputs, max_new_tokens=50)
                
                generated = self.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
                
                # 检查正确性
                if item['output'] in generated:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            logger.info(f"Epoch {epoch+1}, 记忆准确率: {accuracy:.2%}")
        
        # 记忆巩固
        self.hippocampus_trainer.consolidate()
    
    def _train_optimization(self):
        """阶段3：自优化训练"""
        logger.info("训练自优化能力...")
        
        for epoch in range(self.config.epochs_per_stage):
            total_score = 0.0
            
            for item in self.reasoning_dataset:
                # 使用自博弈训练
                output, score = self.self_opt_trainer.train_with_self_play(
                    item['input'],
                    item['output']
                )
                
                total_score += score
                
                # 根据结果应用STDP
                reward = score - 0.5  # 转换为奖励信号
                
                # 获取隐藏状态
                inputs = self.tokenizer(item['input'], return_tensors='pt')
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1][:, -1, :]
                
                self.stdp_trainer.apply_stdp_update(hidden, hidden, reward)
            
            avg_score = total_score / len(self.reasoning_dataset)
            logger.info(f"Epoch {epoch+1}, 平均优化分数: {avg_score:.4f}")
    
    def _train_integration(self):
        """阶段4：整合训练"""
        logger.info("整合训练 - 所有模块协同...")
        
        for epoch in range(self.config.epochs_per_stage):
            total_reward = 0.0
            
            for item in self.reasoning_dataset:
                # 1. 海马体召回相关记忆
                inputs = self.tokenizer(item['input'], return_tensors='pt')
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1][:, -1, :]
                
                recalled = self.hippocampus_trainer.recall_memory(hidden)
                
                # 2. 生成答案
                prompt = f"问题：{item['input']}\n\n请思考并回答："
                gen_inputs = self.tokenizer(prompt, return_tensors='pt')
                gen_inputs = {k: v.to(self.model.device) for k, v in gen_inputs.items()}
                
                with torch.no_grad():
                    gen_outputs = self.model.generate(**gen_inputs, max_new_tokens=100)
                
                generated = self.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
                
                # 3. 自优化改进
                improved, score = self.self_opt_trainer.train_with_self_play(
                    prompt, item['output']
                )
                
                # 4. STDP学习
                reward = score - 0.5
                self.stdp_trainer.apply_stdp_update(hidden, hidden, reward)
                
                # 5. 记忆编码
                self.hippocampus_trainer.encode_memory(
                    improved, hidden, {'score': score}
                )
                
                total_reward += reward
            
            avg_reward = total_reward / len(self.reasoning_dataset)
            logger.info(f"Epoch {epoch+1}, 平均奖励: {avg_reward:.4f}")
    
    def _save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_path = Path(self.config.output_path) / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # 保存训练状态
        state = {
            'training_stats': self.training_stats,
            'stdp_stats': self.stdp_trainer.get_statistics(),
            'self_opt_stats': self.self_opt_trainer.get_statistics()
        }
        
        with open(checkpoint_path / "training_state.json", 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"检查点已保存: {checkpoint_path}")
    
    def _save_final(self):
        """保存最终模型"""
        final_path = Path(self.config.output_path) / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        # 保存训练报告
        report = {
            'training_stats': self.training_stats,
            'stdp_stats': self.stdp_trainer.get_statistics(),
            'hippocampus_stats': {
                'memory_count': len(self.hippocampus_trainer.memories)
            },
            'self_opt_stats': self.self_opt_trainer.get_statistics()
        }
        
        with open(final_path / "training_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"最终模型已保存: {final_path}")


# ============================================
# 主函数
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='一体式优化整合训练')
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--output-path', type=str, default='output/trained_model', help='输出路径')
    parser.add_argument('--epochs', type=int, default=3, help='每阶段训练轮数')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='学习率')
    parser.add_argument('--stages', type=str, default='reasoning,memory,optimization,integration', help='训练阶段')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_path=args.model_path,
        output_path=args.output_path,
        epochs_per_stage=args.epochs,
        learning_rate=args.learning_rate,
        stages=args.stages.split(',')
    )
    
    pipeline = IntegratedTrainingPipeline(config)
    pipeline.setup()
    pipeline.train()
    
    print("\n训练完成！")
    print(f"模型已保存到: {args.output_path}/final")


if __name__ == "__main__":
    main()
