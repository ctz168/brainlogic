#!/usr/bin/env python3
"""
真正的梯度学习引擎
Gradient Learning Engine

核心创新：
1. 用户纠正时使用反向传播
2. 正确答案作为目标信号
3. 真正改变模型的推理模式
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


class GradientLearningEngine:
    """
    梯度学习引擎
    
    核心创新：
    1. 用户纠正 → 反向传播
    2. 正确答案 → 目标信号
    3. 真正改变推理模式
    """
    
    def __init__(self, model_path: str, config: BrainLikeConfig = None):
        self.model_path = model_path
        self.config = config or DEFAULT_CONFIG
        
        self.model = None
        self.tokenizer = None
        self.hippocampus = None
        self.stdp = None
        
        # 学习参数
        self.learning_rate = 0.1  # 较高的学习率
        self.unfrozen_layers: List[str] = []
        
        # 会话状态
        self.session: Dict[str, Any] = {}
        self.history: List[Dict] = []
        self.learning_history: List[Dict] = []
        
        # 统计
        self.total_updates = 0
        self.total_memories = 0
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"梯度学习引擎运行设备: {self.device}")
        self._initialized = False
    
    def initialize(self) -> bool:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"初始化梯度学习引擎: {self.model_path}")
        
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
        self._unfreeze_last_layers(n_layers=2)
        
        self.model.eval()
        self._initialized = True
        
        logger.info("梯度学习引擎初始化完成！")
        return True
    
    def _unfreeze_last_layers(self, n_layers: int = 2):
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
        
        # 解冻
        for layer_idx, name, param in layer_names:
            if layer_idx >= max_layer - n_layers + 1:
                param.requires_grad = True
                self.unfrozen_layers.append(name)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"解冻了 {len(self.unfrozen_layers)} 个参数")
        logger.info(f"可训练参数: {trainable/1e6:.2f}M ({trainable/total*100:.1f}%)")
    
    def _detect_correction(self, user_input: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """检测用户纠正并提取正确答案"""
        correction_keywords = ['不对', '错误', '错了', '不是', '应该是', '正确的是', '答案是']
        
        is_correction = any(kw in user_input for kw in correction_keywords)
        
        if is_correction:
            # 提取数字作为正确答案
            numbers = re.findall(r'\d+', user_input)
            if numbers:
                correct_answer = numbers[-1]  # 最后一个数字通常是答案
                return True, correct_answer, user_input
        
        return False, None, None
    
    def _learn_from_correction(
        self,
        question: str,
        correct_answer: str,
        explanation: str
    ) -> float:
        """
        从用户纠正中学习
        
        使用反向传播更新权重
        """
        logger.info(f"从纠正中学习: 正确答案={correct_answer}")
        
        # 构建训练数据
        # 输入: 问题
        # 目标: 正确答案
        
        # 编码
        input_text = f"问题: {question}\n答案: {correct_answer}"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 创建目标（正确答案的token）
        target_text = correct_answer
        target_ids = self.tokenizer.encode(target_text, return_tensors='pt').to(self.device)
        
        # 切换到训练模式
        self.model.train()
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate
        )
        
        total_loss = 0.0
        
        # 训练几步
        for step in range(3):
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(
                inputs['input_ids'],
                labels=inputs['input_ids']
            )
            
            loss = outputs.loss
            
            if loss is not None and loss.requires_grad:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # 切换回评估模式
        self.model.eval()
        
        self.total_updates += 1
        
        # 记录学习历史
        self.learning_history.append({
            'question': question,
            'correct_answer': correct_answer,
            'loss': total_loss / 3,
            'timestamp': time.time()
        })
        
        logger.info(f"学习完成, 平均损失: {total_loss/3:.4f}")
        
        return total_loss / 3
    
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
            # 获取上一个问题
            if self.history:
                last_question = self.history[-1]['q']
                loss = self._learn_from_correction(last_question, correct_answer, explanation)
                yield f"[已学习: 损失={loss:.4f}]\n\n"
        
        # 3. 召回记忆
        memories = self._recall_memories(prompt, top_k=2)
        
        # 4. 构建消息
        messages = [
            {"role": "system", "content": "你是AI助手，请准确计算。"}
        ]
        
        # 添加学习历史作为示例
        if self.learning_history:
            examples = []
            for lh in self.learning_history[-2:]:
                examples.append(f"Q: {lh['question']} A: {lh['correct_answer']}")
            if examples:
                messages[0]['content'] += f"\n\n学习示例:\n" + "\n".join(examples)
        
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
            'learning_history_count': len(self.learning_history),
            'history_count': len(self.history)
        }
    
    def clear_memory(self):
        self.history = []
        self.learning_history = []
        self.hippocampus.clear()
        self.total_memories = 0
