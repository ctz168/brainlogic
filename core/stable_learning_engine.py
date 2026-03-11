#!/usr/bin/env python3
"""
稳定学习引擎
Stable Learning Engine

优化：
1. 低学习率 (0.0001)
2. 渐进式解冻
3. 梯度裁剪
4. 正确的问答对构建
"""

import os
import sys
import logging
import time
import re
import math
from typing import Dict, List, Optional, Any, Generator, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from core.config import BrainLikeConfig, DEFAULT_CONFIG
from modules.hippocampus import HippocampusSystem
from modules.stdp_system import STDPKernel


class StableLearningEngine:
    """
    稳定学习引擎
    
    核心优化：
    1. 低学习率 - 防止权重崩溃
    2. 梯度裁剪 - 防止梯度爆炸
    3. 正确的问答对 - 有效的学习信号
    4. 渐进式解冻 - 稳定的学习过程
    """
    
    def __init__(self, model_path: str, config: BrainLikeConfig = None):
        self.model_path = model_path
        self.config = config or DEFAULT_CONFIG
        
        self.model = None
        self.tokenizer = None
        self.hippocampus = None
        self.stdp = None
        
        # 学习参数 - 更稳定
        self.learning_rate = 0.0001  # 低学习率
        self.max_grad_norm = 1.0  # 梯度裁剪
        self.unfrozen_layers: List[str] = []
        
        # 会话状态
        self.session: Dict[str, Any] = {}
        self.history: List[Dict] = []
        self.learning_history: List[Dict] = []
        self.correct_patterns: List[Dict] = []  # 存储正确的推理模式
        
        # 统计
        self.total_updates = 0
        self.total_memories = 0
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"稳定学习引擎运行设备: {self.device}")
        self._initialized = False
    
    def initialize(self) -> bool:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"初始化稳定学习引擎: {self.model_path}")
        
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
        
        self.hippocampus = HippocampusSystem(self.config)
        self.stdp = STDPKernel(self.config.stdp)
        
        # 解冻最后几层
        self._unfreeze_last_layers(n_layers=1)  # 只解冻最后1层
        
        self.model.eval()
        self._initialized = True
        
        logger.info("稳定学习引擎初始化完成！")
        return True
    
    def _unfreeze_last_layers(self, n_layers: int = 1):
        """解冻最后几层"""
        # 先冻结所有
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 找到最后的层
        layer_names = []
        for name, param in self.model.named_parameters():
            if 'layers.' in name:
                match = re.search(r'layers\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                    layer_names.append((layer_idx, name, param))
        
        if not layer_names:
            return
        
        max_layer = max(ln[0] for ln in layer_names)
        
        # 只解冻最后的层
        for layer_idx, name, param in layer_names:
            if layer_idx >= max_layer - n_layers + 1:
                param.requires_grad = True
                self.unfrozen_layers.append(name)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"解冻了 {len(self.unfrozen_layers)} 个参数")
        logger.info(f"可训练参数: {trainable/1e6:.2f}M ({trainable/total*100:.1f}%)")
    
    def _detect_correction(self, user_input: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """检测用户纠正"""
        correction_keywords = ['不对', '错误', '错了', '不是', '应该是', '正确的是', '答案是']
        
        is_correction = any(kw in user_input for kw in correction_keywords)
        
        if is_correction:
            # 提取数字
            numbers = re.findall(r'\d+', user_input)
            if numbers:
                correct_answer = numbers[-1]
                return True, correct_answer, user_input
        
        return False, None, None
    
    def _learn_from_correction(
        self,
        question: str,
        correct_answer: str,
        explanation: str
    ) -> float:
        """
        从纠正中学习 - 稳定版本
        """
        logger.info(f"从纠正中学习: 问题={question[:30]}..., 答案={correct_answer}")
        
        # 存储正确的推理模式
        self.correct_patterns.append({
            'question': question,
            'answer': correct_answer,
            'explanation': explanation,
            'timestamp': time.time()
        })
        
        # 构建正确的问答对
        qa_pair = f"问题: {question}\n答案: {correct_answer}\n解释: {explanation}"
        
        # 编码
        inputs = self.tokenizer(
            qa_pair,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 切换到训练模式
        self.model.train()
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        total_loss = 0.0
        
        # 训练几步
        for step in range(5):  # 更多步数，但低学习率
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(
                inputs['input_ids'],
                labels=inputs['input_ids']
            )
            
            loss = outputs.loss
            
            if loss is not None and loss.requires_grad:
                loss.backward()
                
                # 梯度裁剪 - 防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                
                optimizer.step()
                total_loss += loss.item()
        
        # 切换回评估模式
        self.model.eval()
        
        self.total_updates += 1
        
        # 记录学习历史
        self.learning_history.append({
            'question': question,
            'correct_answer': correct_answer,
            'loss': total_loss / 5,
            'timestamp': time.time()
        })
        
        avg_loss = total_loss / 5
        logger.info(f"学习完成, 平均损失: {avg_loss:.4f}")
        
        return avg_loss
    
    def _store_memory(self, content: str, hidden_state: torch.Tensor, importance: float = 1.0):
        """存储记忆"""
        memory_id = self.hippocampus.encode_episode(
            features=hidden_state.squeeze(0),
            timestamp_ms=time.time() * 1000,
            semantic_info={
                'content': content,
                'importance': importance
            }
        )
        self.total_memories += 1
        return memory_id
    
    def _recall_memories(self, query: str, top_k: int = 3) -> List[Dict]:
        """召回记忆"""
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            cue = outputs.hidden_states[-1][:, -1, :].squeeze(0)
        
        memories = self.hippocampus.recall_memories(cue=cue, top_k=top_k)
        return memories if memories else []
    
    def _build_enhanced_prompt(self, prompt: str) -> str:
        """构建增强提示"""
        # 添加学习到的正确模式
        if self.correct_patterns:
            patterns_text = "\n\n已学习的正确模式:\n"
            for i, pattern in enumerate(self.correct_patterns[-3:], 1):
                patterns_text += f"{i}. {pattern['question'][:30]}... → {pattern['answer']}\n"
            return prompt + patterns_text
        return prompt
    
    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        return "".join(list(self.generate_stream(prompt, max_new_tokens)))
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 400
    ) -> Generator[str, None, None]:
        if not self._initialized:
            if not self.initialize():
                yield "初始化失败"
                return
        
        # 1. 检测纠正
        is_correction, correct_answer, explanation = self._detect_correction(prompt)
        
        # 2. 如果是纠正，执行学习
        if is_correction and correct_answer:
            if self.history:
                last_question = self.history[-1]['q']
                loss = self._learn_from_correction(last_question, correct_answer, explanation)
                yield f"[已学习: 损失={loss:.4f}]\n\n"
        
        # 3. 召回记忆
        memories = self._recall_memories(prompt, top_k=2)
        
        # 4. 构建消息
        messages = [
            {"role": "system", "content": "你是AI助手，请准确计算并简洁回答。"}
        ]
        
        # 添加学习到的正确模式
        if self.correct_patterns:
            examples = "\n".join([
                f"- {p['question'][:40]}... 答案: {p['answer']}"
                for p in self.correct_patterns[-3:]
            ])
            messages[0]['content'] += f"\n\n学习示例:\n{examples}"
        
        for h in self.history[-3:]:
            messages.append({"role": "user", "content": h['q']})
            messages.append({"role": "assistant", "content": h['a'][:100]})
        
        messages.append({"role": "user", "content": prompt})
        
        # 5. 生成
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        
        generated = []
        past = None
        response = ""
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                out = self.model(
                    input_ids=input_ids if past is None else input_ids[:, -1:],
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True
                )
                
                logits = out.logits[:, -1, :].clone()
                hidden = out.hidden_states[-1][:, -1, :]
                past = out.past_key_values
                
                logits = logits / 0.6
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                
                tid = next_tok.item()
                if tid == self.tokenizer.eos_token_id:
                    break
                
                tok_text = self.tokenizer.decode([tid], skip_special_tokens=False)
                if "<|im_" in tok_text:
                    break
                
                generated.append(tid)
                text = self.tokenizer.decode([tid], skip_special_tokens=True)
                response += text
                yield text
                
                input_ids = torch.cat([input_ids, next_tok], dim=-1)
        
        # 6. 存储记忆
        self._store_memory(
            content=f"Q: {prompt} A: {response[:100]}",
            hidden_state=hidden,
            importance=2.0 if is_correction else 1.0
        )
        
        # 7. 保存历史
        self.history.append({'q': prompt, 'a': response})
        if len(self.history) > 20:
            self.history = self.history[-20:]
    
    def get_statistics(self) -> Dict:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        
        return {
            'initialized': self._initialized,
            'device': str(self.device),
            'trainable_params': f"{trainable/1e6:.2f}M",
            'trainable_ratio': f"{trainable/total*100:.1f}%",
            'total_updates': self.total_updates,
            'total_memories': self.total_memories,
            'correct_patterns': len(self.correct_patterns),
            'learning_history_count': len(self.learning_history),
            'history_count': len(self.history)
        }
    
    def clear_memory(self):
        self.history = []
        self.learning_history = []
        self.correct_patterns = []
        self.hippocampus.clear()
        self.total_memories = 0
