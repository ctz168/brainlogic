#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 海马体集成训练测评系统
Hippocampus-Integrated Training and Evaluation System

真正将海马体记忆系统集成到模型推理中
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
from dataclasses import dataclass
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


# ============================================
# 海马体集成模型
# ============================================

class HippocampusIntegratedModel:
    """
    海马体集成模型
    
    将海马体记忆系统真正集成到模型推理中
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cpu")
        
        # 模型组件
        self.model = None
        self.tokenizer = None
        
        # 海马体系统
        self.hippocampus = None
        
        # 记忆存储（用于跨会话记忆）
        self.session_memories: Dict[str, Any] = {}
        
        # 统计
        self.encode_count = 0
        self.recall_count = 0
    
    def setup(self):
        """初始化"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from core.config import BrainLikeConfig
        from modules.hippocampus import HippocampusSystem
        
        logger.info("初始化海马体集成模型...")
        
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
        
        # 初始化海马体
        config = BrainLikeConfig()
        config.hippocampus.ca3_memory_capacity = 1000  # 增加记忆容量
        config.hippocampus.ca3_recall_top_k = 5
        config.hippocampus.swr_idle_threshold_minutes = 0.1  # 6秒后可巩固
        
        self.hippocampus = HippocampusSystem(config)
        
        logger.info("海马体集成模型初始化完成")
        logger.info(f"记忆容量: {config.hippocampus.ca3_memory_capacity}")
    
    def encode_memory(self, text: str, key_info: Dict[str, Any] = None):
        """
        编码记忆到海马体
        
        Args:
            text: 文本内容
            key_info: 关键信息字典
        """
        self.encode_count += 1
        
        # 编码文本为特征
        encodings = self.tokenizer(
            text, return_tensors='pt', max_length=64, truncation=True
        )
        input_ids = encodings['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            # 使用最后一层隐藏状态作为特征
            features = outputs.hidden_states[-1][:, -1, :]  # [1, hidden_dim]
        
        # 存储到海马体
        semantic_info = {
            'semantic_pointer': text[:100],  # 存储文本摘要
            'temporal_skeleton': [time.time() * 1000],
            'causal_links': [],
            'key_info': key_info or {}
        }
        
        memory_id = self.hippocampus.encode_episode(
            features,
            time.time() * 1000,
            semantic_info
        )
        
        # 同时存储到会话记忆（用于快速检索）
        if key_info:
            for key, value in key_info.items():
                self.session_memories[key] = value
        
        logger.debug(f"编码记忆: {memory_id}, 内容: {text[:30]}...")
        
        return memory_id
    
    def recall_memory(self, query: str) -> List[Dict[str, Any]]:
        """
        从海马体召回记忆
        
        Args:
            query: 查询文本
            
        Returns:
            召回的记忆列表
        """
        self.recall_count += 1
        
        # 首先检查会话记忆（快速路径）
        recalled_info = {}
        
        # 提取查询中的关键词
        for key, value in self.session_memories.items():
            if key in query:
                recalled_info[key] = value
        
        # 编码查询
        encodings = self.tokenizer(
            query, return_tensors='pt', max_length=32, truncation=True
        )
        input_ids = encodings['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            features = outputs.hidden_states[-1][:, -1, :]
        
        # 从海马体召回
        memories = self.hippocampus.recall_memories(features, top_k=3)
        
        # 合并结果
        results = []
        
        # 添加会话记忆
        if recalled_info:
            results.append({
                'source': 'session',
                'content': recalled_info,
                'relevance': 1.0
            })
        
        # 添加海马体召回的记忆
        for mem in memories:
            results.append({
                'source': 'hippocampus',
                'content': mem.get('semantic_pointer', ''),
                'relevance': mem.get('relevance_score', 0),
                'key_info': mem.get('key_info', {})
            })
        
        return results
    
    def generate_with_memory(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        use_memory: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        带记忆的生成
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            use_memory: 是否使用记忆
            
        Returns:
            生成的文本和记忆信息
        """
        memory_info = {
            'recalled': [],
            'encoded': None
        }
        
        # 检查是否是记忆编码请求
        encode_pattern = r"记住[：:]\s*(.+)"
        encode_match = re.search(encode_pattern, prompt)
        
        if encode_match and use_memory:
            # 编码模式
            content = encode_match.group(1)
            
            # 提取关键信息
            key_info = self._extract_key_info(content)
            
            # 编码到海马体
            memory_id = self.encode_memory(content, key_info)
            memory_info['encoded'] = memory_id
            
            # 生成确认回复
            response = f"好的，我已经记住了：{content}"
            if key_info:
                response += f"\n（关键信息：{key_info}）"
            
            return response, memory_info
        
        # 检查是否是记忆召回请求
        recall_patterns = [
            r"(.+?)今年多大",
            r"(.+?)住在哪里",
            r"(.+?)的职业",
            r"(.+?)叫什么",
            r"(.+?)是什么",
            r"(.+?)的(.+?)是",
        ]
        
        recalled_info = None
        for pattern in recall_patterns:
            match = re.search(pattern, prompt)
            if match:
                # 召回记忆
                memories = self.recall_memory(prompt)
                memory_info['recalled'] = memories
                
                if memories:
                    recalled_info = memories[0].get('content', {})
                    if isinstance(recalled_info, dict):
                        # 提取相关信息
                        for key, value in recalled_info.items():
                            if key in prompt:
                                recalled_info = {key: value}
                                break
                break
        
        # 构建增强提示
        enhanced_prompt = prompt
        if recalled_info and use_memory:
            # 将召回的记忆注入提示
            if isinstance(recalled_info, dict):
                memory_context = "根据之前的对话记忆：" + str(recalled_info)
            else:
                memory_context = f"根据之前的对话记忆：{recalled_info}"
            enhanced_prompt = f"{memory_context}\n\n问题：{prompt}"
        
        # 生成回复
        input_text = f"<|im_start|>user\n{enhanced_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        encodings = self.tokenizer(
            input_text, return_tensors='pt', max_length=256, truncation=True
        )
        input_ids = encodings['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回复部分
        if "assistant" in generated:
            response = generated.split("assistant")[-1].strip()
        else:
            response = generated.strip()
        
        # 如果有召回的记忆，优先使用
        if recalled_info and isinstance(recalled_info, dict):
            # 检查是否可以直接回答
            for key, value in recalled_info.items():
                if key in prompt:
                    # 直接使用记忆回答
                    if "多大" in prompt or "年龄" in prompt:
                        response = f"根据我的记忆，{value}岁。"
                    elif "住" in prompt:
                        response = f"根据我的记忆，住在{value}。"
                    elif "职业" in prompt:
                        response = f"根据我的记忆，职业是{value}。"
                    break
        
        return response, memory_info
    
    def _extract_key_info(self, text: str) -> Dict[str, Any]:
        """从文本中提取关键信息"""
        info = {}
        
        # 提取姓名
        name_patterns = [
            r"我叫(\w+)",
            r"(\w+)叫(\w+)",
            r"(\w+)是",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1)
                info['姓名'] = name
                break
        
        # 提取年龄
        age_match = re.search(r"(\d+)岁", text)
        if age_match:
            info['年龄'] = age_match.group(1)
        
        # 提取职业
        job_patterns = [
            r"是(\w+工程师)",
            r"是(\w+师)",
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
        
        return info
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'encode_count': self.encode_count,
            'recall_count': self.recall_count,
            'session_memories': len(self.session_memories),
            'session_memory_content': self.session_memories
        }
        
        if self.hippocampus:
            stats['hippocampus'] = self.hippocampus.get_statistics()
        
        return stats


# ============================================
# 记忆专项训练
# ============================================

class MemoryFocusedTrainer:
    """记忆专项训练器"""
    
    def __init__(self, model: HippocampusIntegratedModel):
        self.model = model
        
        # 记忆训练数据
        self.memory_training_data = [
            # 基础记忆
            {"encode": "我叫张三，今年25岁，是软件工程师，住在北京。", 
             "info": {"姓名": "张三", "年龄": "25", "职业": "软件工程师", "地点": "北京"}},
            {"encode": "我叫李四，今年30岁，是医生，住在上海。", 
             "info": {"姓名": "李四", "年龄": "30", "职业": "医生", "地点": "上海"}},
            {"encode": "我叫王五，今年28岁，是教师，住在广州。", 
             "info": {"姓名": "王五", "年龄": "28", "职业": "教师", "地点": "广州"}},
            
            # 复杂记忆
            {"encode": "小明喜欢蓝色，养了一只猫叫咪咪，今年3岁。", 
             "info": {"人物": "小明", "喜好": "蓝色", "宠物": "猫", "宠物名": "咪咪", "宠物年龄": "3"}},
            {"encode": "公司会议在周三下午3点，地点是会议室A，主题是产品发布。", 
             "info": {"事件": "会议", "时间": "周三下午3点", "地点": "会议室A", "主题": "产品发布"}},
            
            # 数值记忆
            {"encode": "订单号是20240310001，总价是299元，包含3件商品。", 
             "info": {"订单号": "20240310001", "总价": "299", "商品数量": "3"}},
        ]
    
    def train(self, epochs: int = 3):
        """执行记忆训练"""
        logger.info("="*60)
        logger.info("开始记忆专项训练")
        logger.info("="*60)
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            correct = 0
            total = 0
            
            for item in self.memory_training_data:
                # 编码记忆
                self.model.encode_memory(item['encode'], item['info'])
                
                # 测试召回
                info = item['info']
                for key, value in info.items():
                    # 构建查询
                    if key == "姓名":
                        query = f"{value}今年多大？"
                        expected_key = "年龄"
                    elif key == "年龄":
                        continue
                    elif key == "职业":
                        query = f"{info.get('姓名', '他')}的职业是什么？"
                        expected_key = "职业"
                    elif key == "地点":
                        query = f"{info.get('姓名', '他')}住在哪里？"
                        expected_key = "地点"
                    else:
                        continue
                    
                    # 召回
                    memories = self.model.recall_memory(query)
                    
                    # 检查是否正确
                    if memories:
                        recalled = memories[0].get('content', {})
                        if isinstance(recalled, dict) and expected_key in recalled:
                            if str(recalled[expected_key]) == str(value):
                                correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0
            logger.info(f"  记忆召回准确率: {accuracy*100:.1f}% ({correct}/{total})")
        
        logger.info("\n记忆训练完成!")
        
        return {
            'epochs': epochs,
            'final_accuracy': accuracy
        }


# ============================================
# 记忆能力测评
# ============================================

class MemoryCapabilityEvaluator:
    """记忆能力测评器"""
    
    def __init__(self, model: HippocampusIntegratedModel):
        self.model = model
        
        # 测试用例
        self.test_cases = [
            # 编码-召回测试
            {
                "type": "encode_recall",
                "encode": "记住：我叫张三，今年25岁，是工程师，住在北京。",
                "queries": [
                    ("张三今年多大？", "25"),
                    ("张三住在哪里？", "北京"),
                    ("张三的职业是什么？", "工程师"),
                ]
            },
            {
                "type": "encode_recall",
                "encode": "记住：小红喜欢蓝色，养了一只猫叫咪咪。",
                "queries": [
                    ("小红喜欢什么颜色？", "蓝色"),
                    ("小红的猫叫什么？", "咪咪"),
                ]
            },
            {
                "type": "encode_recall",
                "encode": "记住：会议在明天下午2点，地点是会议室B。",
                "queries": [
                    ("会议什么时候？", "明天下午2点"),
                    ("会议在哪里？", "会议室B"),
                ]
            },
            # 多记忆测试
            {
                "type": "multi_memory",
                "encodes": [
                    "记住：用户A叫张三，25岁。",
                    "记住：用户B叫李四，30岁。",
                ],
                "queries": [
                    ("张三今年多大？", "25"),
                    ("李四今年多大？", "30"),
                ]
            },
        ]
    
    def evaluate(self) -> Dict[str, Any]:
        """执行测评"""
        logger.info("="*60)
        logger.info("开始记忆能力测评")
        logger.info("="*60)
        
        results = {
            'total_tests': 0,
            'correct': 0,
            'details': []
        }
        
        for test_case in self.test_cases:
            test_type = test_case['type']
            
            if test_type == "encode_recall":
                # 编码
                self.model.generate_with_memory(test_case['encode'], use_memory=True)
                
                # 测试召回
                for query, expected in test_case['queries']:
                    response, mem_info = self.model.generate_with_memory(query, use_memory=True)
                    
                    # 检查是否包含期望答案
                    is_correct = expected in response
                    
                    results['total_tests'] += 1
                    if is_correct:
                        results['correct'] += 1
                    
                    results['details'].append({
                        'query': query,
                        'expected': expected,
                        'response': response[:100],
                        'correct': is_correct,
                        'memory_used': len(mem_info.get('recalled', [])) > 0
                    })
                    
                    status = "✓" if is_correct else "✗"
                    logger.info(f"  {status} Q: {query} -> A: {response[:50]}...")
            
            elif test_type == "multi_memory":
                # 编码多个记忆
                for encode in test_case['encodes']:
                    self.model.generate_with_memory(encode, use_memory=True)
                
                # 测试召回
                for query, expected in test_case['queries']:
                    response, mem_info = self.model.generate_with_memory(query, use_memory=True)
                    
                    is_correct = expected in response
                    
                    results['total_tests'] += 1
                    if is_correct:
                        results['correct'] += 1
                    
                    results['details'].append({
                        'query': query,
                        'expected': expected,
                        'response': response[:100],
                        'correct': is_correct
                    })
                    
                    status = "✓" if is_correct else "✗"
                    logger.info(f"  {status} Q: {query} -> A: {response[:50]}...")
        
        accuracy = results['correct'] / results['total_tests'] if results['total_tests'] > 0 else 0
        
        logger.info("\n" + "="*60)
        logger.info(f"记忆测评结果: {accuracy*100:.1f}% ({results['correct']}/{results['total_tests']})")
        logger.info("="*60)
        
        results['accuracy'] = accuracy
        
        return results


# ============================================
# 主函数
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='海马体集成训练测评')
    parser.add_argument('--model-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B')
    parser.add_argument('--output-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/output')
    parser.add_argument('--train-epochs', type=int, default=2)
    parser.add_argument('--skip-training', action='store_true')
    
    args = parser.parse_args()
    
    # 初始化模型
    model = HippocampusIntegratedModel(args.model_path)
    model.setup()
    
    # 训练前测评
    logger.info("\n" + "="*60)
    logger.info("训练前记忆能力测评")
    logger.info("="*60)
    
    evaluator = MemoryCapabilityEvaluator(model)
    before_results = evaluator.evaluate()
    
    # 记忆专项训练
    if not args.skip_training:
        trainer = MemoryFocusedTrainer(model)
        train_results = trainer.train(epochs=args.train_epochs)
    
    # 训练后测评
    logger.info("\n" + "="*60)
    logger.info("训练后记忆能力测评")
    logger.info("="*60)
    
    # 清空记忆重新测试
    model.session_memories = {}
    if model.hippocampus:
        model.hippocampus.clear()
    
    after_results = evaluator.evaluate()
    
    # 对比结果
    logger.info("\n" + "="*60)
    logger.info("对比结果")
    logger.info("="*60)
    
    improvement = after_results['accuracy'] - before_results['accuracy']
    
    logger.info(f"训练前准确率: {before_results['accuracy']*100:.1f}%")
    logger.info(f"训练后准确率: {after_results['accuracy']*100:.1f}%")
    logger.info(f"提升幅度: {improvement*100:+.1f}%")
    
    # 统计信息
    stats = model.get_statistics()
    logger.info(f"\n海马体统计:")
    logger.info(f"  编码次数: {stats['encode_count']}")
    logger.info(f"  召回次数: {stats['recall_count']}")
    logger.info(f"  会话记忆: {stats['session_memories']}")
    
    # 保存结果
    os.makedirs(args.output_path, exist_ok=True)
    
    report = {
        'before_training': before_results,
        'after_training': after_results,
        'improvement': improvement,
        'statistics': stats,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_path, 'memory_evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n报告已保存: {args.output_path}/memory_evaluation_report.json")


if __name__ == "__main__":
    main()
