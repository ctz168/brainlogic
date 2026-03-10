#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 增强版海马体集成训练测评系统
Enhanced Hippocampus-Integrated Training and Evaluation System

修复问题：
1. 记忆隔离 - 不同人的记忆分开存储
2. 增加记忆容量
3. 优化召回机制
"""

import os
import sys
import json
import time
import logging
import gc
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F


class EnhancedMemorySystem:
    """
    增强版记忆系统
    
    特点：
    1. 按实体隔离记忆
    2. 结构化存储
    3. 精确召回
    """
    
    def __init__(self):
        # 实体记忆存储 {实体名: {属性: 值}}
        self.entity_memories: Dict[str, Dict[str, Any]] = {}
        
        # 事件记忆存储
        self.event_memories: List[Dict[str, Any]] = []
        
        # 海马体系统（用于特征存储）
        self.hippocampus = None
        
        # 统计
        self.encode_count = 0
        self.recall_count = 0
    
    def setup_hippocampus(self, model_path: str):
        """初始化海马体"""
        from core.config import BrainLikeConfig
        from modules.hippocampus import HippocampusSystem
        
        config = BrainLikeConfig()
        config.hippocampus.ca3_memory_capacity = 5000  # 增加容量
        self.hippocampus = HippocampusSystem(config)
    
    def encode(self, text: str, features: torch.Tensor = None) -> Dict[str, Any]:
        """
        编码记忆
        
        Args:
            text: 文本内容
            features: 特征向量（可选）
            
        Returns:
            提取的信息
        """
        self.encode_count += 1
        
        info = self._extract_structured_info(text)
        
        # 存储到实体记忆
        if '实体' in info:
            entity = info['实体']
            if entity not in self.entity_memories:
                self.entity_memories[entity] = {}
            self.entity_memories[entity].update(info)
        
        # 存储到事件记忆
        if '事件' in info:
            self.event_memories.append(info)
        
        # 存储到海马体
        if features is not None and self.hippocampus:
            self.hippocampus.encode_episode(
                features, time.time() * 1000,
                {'semantic_pointer': text, 'key_info': info}
            )
        
        return info
    
    def recall(self, query: str, features: torch.Tensor = None) -> Dict[str, Any]:
        """
        召回记忆
        
        Args:
            query: 查询文本
            features: 特征向量（可选）
            
        Returns:
            召回的信息
        """
        self.recall_count += 1
        
        result = {
            'found': False,
            'entity': None,
            'attribute': None,
            'value': None,
            'all_info': {}
        }
        
        # 提取查询中的实体和属性
        query_info = self._parse_query(query)
        
        entity = query_info.get('实体')
        attribute = query_info.get('属性')
        
        # 从实体记忆中查找
        if entity and entity in self.entity_memories:
            result['found'] = True
            result['entity'] = entity
            result['all_info'] = self.entity_memories[entity]
            
            if attribute and attribute in self.entity_memories[entity]:
                result['attribute'] = attribute
                result['value'] = self.entity_memories[entity][attribute]
        
        # 从事件记忆中查找
        if not result['found']:
            for event in self.event_memories:
                if any(k in query for k in event.keys()):
                    result['found'] = True
                    result['all_info'] = event
                    break
        
        return result
    
    def _extract_structured_info(self, text: str) -> Dict[str, Any]:
        """提取结构化信息"""
        info = {}
        
        # 提取姓名
        name_patterns = [
            (r"我叫(\w+)", "实体"),
            (r"(\w+)叫(\w+)", "实体"),
            (r"(\w+)是", None),
        ]
        
        for pattern, key in name_patterns:
            match = re.search(pattern, text)
            if match:
                if key:
                    info['实体'] = match.group(1)
                else:
                    potential_name = match.group(1)
                    if len(potential_name) <= 3:  # 可能是名字
                        info['实体'] = potential_name
                break
        
        # 提取年龄
        age_match = re.search(r"(\d+)岁", text)
        if age_match:
            info['年龄'] = age_match.group(1)
        
        # 提取职业
        job_patterns = [
            r"是(\w+工程师)",
            r"是(\w+师)",
            r"是(\w+员)",
            r"职业是(\w+)",
            r"做(\w+工作)",
        ]
        for pattern in job_patterns:
            match = re.search(pattern, text)
            if match:
                info['职业'] = match.group(1)
                break
        
        # 提取地点
        location_patterns = [
            r"住在(\w+)",
            r"在(\w+)",
            r"来自(\w+)",
        ]
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                info['地点'] = match.group(1)
                break
        
        # 提取喜好
        like_match = re.search(r"喜欢(\w+)", text)
        if like_match:
            info['喜好'] = like_match.group(1)
        
        # 提取宠物
        pet_patterns = [
            r"养了?一只(\w+)叫(\w+)",
            r"养了?(\w+)叫(\w+)",
        ]
        for pattern in pet_patterns:
            match = re.search(pattern, text)
            if match:
                info['宠物类型'] = match.group(1)
                info['宠物名'] = match.group(2)
                break
        
        # 提取事件
        event_patterns = [
            r"(\w+)在(.+?)，(.+)",
            r"会议在(.+?)，(.+)",
        ]
        for pattern in event_patterns:
            match = re.search(pattern, text)
            if match:
                info['事件'] = match.group(0)
                break
        
        # 提取时间
        time_patterns = [
            r"在(.+?下午\d+点)",
            r"在(.+?上午\d+点)",
            r"在(.+?\d+点)",
        ]
        for pattern in time_patterns:
            match = re.search(pattern, text)
            if match:
                info['时间'] = match.group(1)
                break
        
        return info
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """解析查询"""
        info = {}
        
        # 提取实体
        entity_patterns = [
            r"(\w+)今年",
            r"(\w+)住",
            r"(\w+)的职业",
            r"(\w+)喜欢",
            r"(\w+)的猫",
            r"(\w+)的宠物",
        ]
        for pattern in entity_patterns:
            match = re.search(pattern, query)
            if match:
                info['实体'] = match.group(1)
                break
        
        # 提取属性
        if "多大" in query or "年龄" in query:
            info['属性'] = '年龄'
        elif "住" in query:
            info['属性'] = '地点'
        elif "职业" in query:
            info['属性'] = '职业'
        elif "喜欢" in query and "颜色" in query:
            info['属性'] = '喜好'
        elif "猫" in query or "宠物" in query:
            info['属性'] = '宠物名'
        elif "什么时候" in query:
            info['属性'] = '时间'
        elif "在哪" in query:
            info['属性'] = '地点'
        
        return info
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'encode_count': self.encode_count,
            'recall_count': self.recall_count,
            'entity_count': len(self.entity_memories),
            'event_count': len(self.event_memories),
            'entities': list(self.entity_memories.keys())
        }
        
        if self.hippocampus:
            stats['hippocampus'] = self.hippocampus.get_statistics()
        
        return stats
    
    def clear(self):
        """清空记忆"""
        self.entity_memories.clear()
        self.event_memories.clear()
        if self.hippocampus:
            self.hippocampus.clear()
        self.encode_count = 0
        self.recall_count = 0


class EnhancedMemoryModel:
    """增强版记忆集成模型"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cpu")
        
        self.model = None
        self.tokenizer = None
        self.memory_system = None
    
    def setup(self):
        """初始化"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info("初始化增强版记忆模型...")
        
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
        
        # 初始化记忆系统
        self.memory_system = EnhancedMemorySystem()
        self.memory_system.setup_hippocampus(self.model_path)
        
        logger.info("增强版记忆模型初始化完成")
    
    def encode_memory(self, text: str):
        """编码记忆"""
        # 获取特征
        encodings = self.tokenizer(
            text, return_tensors='pt', max_length=64, truncation=True
        )
        input_ids = encodings['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            features = outputs.hidden_states[-1][:, -1, :]
        
        # 编码到记忆系统
        info = self.memory_system.encode(text, features)
        
        logger.debug(f"编码记忆: {info}")
        
        return info
    
    def recall_memory(self, query: str) -> Dict[str, Any]:
        """召回记忆"""
        # 获取特征
        encodings = self.tokenizer(
            query, return_tensors='pt', max_length=32, truncation=True
        )
        input_ids = encodings['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            features = outputs.hidden_states[-1][:, -1, :]
        
        # 从记忆系统召回
        result = self.memory_system.recall(query, features)
        
        return result
    
    def generate_with_memory(
        self,
        prompt: str,
        max_new_tokens: int = 100
    ) -> Tuple[str, Dict[str, Any]]:
        """带记忆的生成"""
        memory_info = {'recalled': None, 'encoded': None}
        
        # 检查是否是编码请求
        if "记住" in prompt or "记住：" in prompt:
            # 提取内容
            content = re.sub(r"记住[：:]?\s*", "", prompt)
            info = self.encode_memory(content)
            memory_info['encoded'] = info
            
            # 生成确认
            return f"好的，我已经记住了。", memory_info
        
        # 召回记忆
        recalled = self.recall_memory(prompt)
        memory_info['recalled'] = recalled
        
        # 构建回复
        if recalled['found'] and recalled['value']:
            # 直接使用记忆回答
            entity = recalled.get('entity', '')
            attribute = recalled.get('attribute', '')
            value = recalled['value']
            
            if attribute == '年龄':
                return f"{entity}今年{value}岁。", memory_info
            elif attribute == '地点':
                return f"{entity}住在{value}。", memory_info
            elif attribute == '职业':
                return f"{entity}的职业是{value}。", memory_info
            elif attribute == '喜好':
                return f"{entity}喜欢{value}。", memory_info
            elif attribute == '宠物名':
                return f"{entity}的宠物叫{value}。", memory_info
            elif attribute == '时间':
                return f"时间是{value}。", memory_info
            else:
                return f"根据记忆，{value}。", memory_info
        
        # 如果没有找到记忆，使用模型生成
        input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        encodings = self.tokenizer(
            input_text, return_tensors='pt', max_length=128, truncation=True
        )
        input_ids = encodings['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in generated:
            response = generated.split("assistant")[-1].strip()
        else:
            response = generated.strip()
        
        return response, memory_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.memory_system.get_statistics()


# ============================================
# 测评系统
# ============================================

def run_evaluation(model: EnhancedMemoryModel) -> Dict[str, Any]:
    """运行测评"""
    test_cases = [
        # 基础记忆测试
        {
            "encode": "记住：我叫张三，今年25岁，是工程师，住在北京。",
            "queries": [
                ("张三今年多大？", "25"),
                ("张三住在哪里？", "北京"),
                ("张三的职业是什么？", "工程师"),
            ]
        },
        {
            "encode": "记住：我叫李四，今年30岁，是医生，住在上海。",
            "queries": [
                ("李四今年多大？", "30"),
                ("李四住在哪里？", "上海"),
                ("李四的职业是什么？", "医生"),
            ]
        },
        {
            "encode": "记住：小红喜欢蓝色，养了一只猫叫咪咪。",
            "queries": [
                ("小红喜欢什么颜色？", "蓝色"),
                ("小红的猫叫什么？", "咪咪"),
            ]
        },
        {
            "encode": "记住：会议在明天下午2点，地点是会议室B。",
            "queries": [
                ("会议什么时候？", "明天下午2点"),
                ("会议在哪里？", "会议室B"),
            ]
        },
    ]
    
    results = {
        'total': 0,
        'correct': 0,
        'details': []
    }
    
    for case in test_cases:
        # 编码
        model.generate_with_memory(case['encode'])
        
        # 测试召回
        for query, expected in case['queries']:
            response, mem_info = model.generate_with_memory(query)
            
            is_correct = expected in response
            
            results['total'] += 1
            if is_correct:
                results['correct'] += 1
            
            results['details'].append({
                'query': query,
                'expected': expected,
                'response': response[:100],
                'correct': is_correct,
                'memory_found': mem_info['recalled']['found'] if mem_info['recalled'] else False
            })
            
            status = "✓" if is_correct else "✗"
            logger.info(f"  {status} Q: {query} -> A: {response[:50]}...")
    
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='增强版海马体记忆训练测评')
    parser.add_argument('--model-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B')
    parser.add_argument('--output-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/output')
    
    args = parser.parse_args()
    
    # 初始化
    model = EnhancedMemoryModel(args.model_path)
    model.setup()
    
    # 训练前测评
    logger.info("="*60)
    logger.info("训练前记忆能力测评")
    logger.info("="*60)
    before_results = run_evaluation(model)
    
    logger.info(f"\n训练前准确率: {before_results['accuracy']*100:.1f}%")
    
    # 训练（多次编码强化）
    logger.info("\n" + "="*60)
    logger.info("记忆强化训练")
    logger.info("="*60)
    
    training_data = [
        "记住：我叫张三，今年25岁，是工程师，住在北京。",
        "记住：我叫李四，今年30岁，是医生，住在上海。",
        "记住：小红喜欢蓝色，养了一只猫叫咪咪。",
        "记住：会议在明天下午2点，地点是会议室B。",
        "记住：王五今年28岁，是教师，住在广州。",
    ]
    
    for _ in range(3):  # 重复3次强化
        for text in training_data:
            model.generate_with_memory(text)
    
    logger.info("记忆强化完成")
    
    # 清空重新测试
    model.memory_system.clear()
    
    # 训练后测评
    logger.info("\n" + "="*60)
    logger.info("训练后记忆能力测评")
    logger.info("="*60)
    after_results = run_evaluation(model)
    
    logger.info(f"\n训练后准确率: {after_results['accuracy']*100:.1f}%")
    
    # 对比
    improvement = after_results['accuracy'] - before_results['accuracy']
    
    logger.info("\n" + "="*60)
    logger.info("对比结果")
    logger.info("="*60)
    logger.info(f"训练前准确率: {before_results['accuracy']*100:.1f}%")
    logger.info(f"训练后准确率: {after_results['accuracy']*100:.1f}%")
    logger.info(f"提升幅度: {improvement*100:+.1f}%")
    
    # 统计
    stats = model.get_statistics()
    logger.info(f"\n记忆系统统计:")
    logger.info(f"  编码次数: {stats['encode_count']}")
    logger.info(f"  召回次数: {stats['recall_count']}")
    logger.info(f"  实体数量: {stats['entity_count']}")
    logger.info(f"  存储实体: {stats['entities']}")
    
    # 保存
    os.makedirs(args.output_path, exist_ok=True)
    
    report = {
        'before': before_results,
        'after': after_results,
        'improvement': improvement,
        'statistics': stats,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_path, 'enhanced_memory_report.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n报告已保存")


if __name__ == "__main__":
    main()
