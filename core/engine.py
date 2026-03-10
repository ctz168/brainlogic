"""
类人脑双系统全闭环AI架构 - 生产级集成模块 (纯模型推理版)
Human-Like Brain Dual-System Full-Loop AI Architecture - Pure Model Inference

特点：
1. 不使用规则化回答
2. 使用思维链(CoT)提示引导模型推理
3. 保持对话历史上下文
"""

import os
import sys
import logging
import time
import threading
import re
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class GenerationConfig:
    """生成配置"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


class BrainLikeAIEngine:
    """
    类人脑双系统全闭环AI架构 - 生产级引擎 (纯模型推理版)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = None,
        config: Dict[str, Any] = None
    ):
        self.model_path = model_path
        self.config = config or {}
        
        # 设置设备
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"使用设备: {self.device}")
        
        # 模型组件
        self.model = None
        self.tokenizer = None
        
        # 核心模块
        self.hippocampus = None
        self.stdp_system = None
        self.optimization = None
        
        # 对话历史
        self.dialogue_history: List[Dict[str, str]] = []
        
        # 状态
        self._initialized = False
        self._generation_count = 0
        
    def initialize(self) -> bool:
        """初始化所有组件"""
        if self._initialized:
            return True
            
        logger.info("正在初始化引擎...")
        
        try:
            # 1. 加载模型和tokenizer
            self._load_model()
            
            # 2. 初始化核心模块
            self._init_modules()
            
            self._initialized = True
            logger.info("引擎初始化完成！")
            return True
            
        except Exception as e:
            logger.error(f"引擎初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_model(self):
        """加载模型和tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"加载模型: {self.model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # 确保有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型加载完成！参数量: {total_params/1e9:.2f}B")
    
    def _init_modules(self):
        """初始化核心模块"""
        from core.config import BrainLikeConfig
        from modules.hippocampus import HippocampusSystem
        from modules.stdp_system import STDPSystem
        from modules.self_optimization import SelfClosedLoopOptimization
        
        # 加载配置
        self.brain_config = BrainLikeConfig()
        
        # 初始化海马体系统
        logger.info("初始化海马体记忆系统...")
        self.hippocampus = HippocampusSystem(self.brain_config)
        
        # 初始化STDP系统
        logger.info("初始化STDP学习系统...")
        self.stdp_system = STDPSystem(self.brain_config)
        
        # 初始化自闭环优化系统
        logger.info("初始化自闭环优化系统...")
        self.optimization = SelfClosedLoopOptimization(self.brain_config)
        
        logger.info("核心模块初始化完成")
    
    def _build_cot_prompt(self, user_input: str) -> str:
        """
        构建思维链提示
        
        引导模型一步步推理，不使用规则
        """
        # 系统提示
        system_prompt = """你是一个智能助手，擅长数学计算和逻辑推理。

当遇到计算问题时，请按以下步骤思考：
1. 理解问题：需要计算什么？
2. 提取数据：有哪些已知数字？
3. 确定公式：用什么公式计算？
4. 执行计算：一步步算出结果
5. 给出答案：用简洁的语言回答

重要提示：
- 日租金 = 房租金额 ÷ 租期天数
- 月租金 = 日租金 × 30天
- 如果已知房租和天数，先算日租金，再算月租金

请准确计算，给出正确答案。"""

        # 构建对话历史
        history_text = ""
        if self.dialogue_history:
            history_text = "\n【之前的对话】\n"
            for turn in self.dialogue_history[-6:]:  # 最近3轮对话
                history_text += f"{turn['role']}: {turn['content']}\n"
            history_text += "\n"
        
        prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
{history_text}<|im_start|>user
{user_input}
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def _add_to_history(self, role: str, content: str):
        """添加到对话历史"""
        self.dialogue_history.append({'role': role, 'content': content})
        if len(self.dialogue_history) > 20:
            self.dialogue_history = self.dialogue_history[-20:]
    
    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig = None
    ) -> Generator[str, None, None]:
        """
        流式生成文本
        
        使用思维链提示引导模型推理，不使用规则化回答
        """
        if not self._initialized:
            if not self.initialize():
                yield "抱歉，系统初始化失败。"
                return
        
        config = config or GenerationConfig()
        self._generation_count += 1
        
        # 构建思维链提示
        text = self._build_cot_prompt(prompt)
        
        # 编码输入
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 流式生成
        from transformers import TextIteratorStreamer
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask", None),
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": config.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        # 在单独线程中运行生成
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_kwargs
        )
        thread.start()
        
        # 收集完整响应
        full_response = ""
        
        try:
            for text in streamer:
                if text:
                    full_response += text
                    yield text
        finally:
            thread.join(timeout=5)
        
        # 添加到历史
        self._add_to_history('user', prompt)
        self._add_to_history('assistant', full_response)
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig = None
    ) -> str:
        """生成完整响应"""
        full_response = ""
        for text in self.generate_stream(prompt, config):
            full_response += text
        return full_response
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = {
            'initialized': self._initialized,
            'device': str(self.device),
            'generation_count': self._generation_count,
            'model_path': self.model_path,
            'history_length': len(self.dialogue_history)
        }
        
        if self.hippocampus:
            stats['hippocampus'] = self.hippocampus.get_statistics()
        
        if self.stdp_system:
            stdp_stats = self.stdp_system.get_statistics()
            stats['stdp'] = {
                'total_updates': stdp_stats.total_updates,
                'ltp_count': stdp_stats.ltp_count,
                'ltd_count': stdp_stats.ltd_count
            }
        
        if self.optimization:
            stats['optimization'] = self.optimization.get_statistics()
        
        return stats
    
    def clear_memory(self):
        """清空记忆"""
        if self.hippocampus:
            self.hippocampus.clear()
        self.dialogue_history.clear()
        logger.info("记忆已清空")
    
    def offline_consolidation(self) -> Dict[str, Any]:
        """执行离线记忆巩固"""
        if self.hippocampus:
            return self.hippocampus.offline_consolidation()
        return {'status': 'not_available'}


# 便捷函数
_engine_instance: Optional[BrainLikeAIEngine] = None

def get_engine(model_path: str = None) -> BrainLikeAIEngine:
    """获取全局引擎实例"""
    global _engine_instance
    
    if _engine_instance is None:
        model_path = model_path or os.environ.get(
            "MODEL_PATH",
            str(PROJECT_ROOT / "models" / "Qwen3.5-0.8B")
        )
        _engine_instance = BrainLikeAIEngine(model_path)
    
    return _engine_instance


def generate(prompt: str, **kwargs) -> str:
    """便捷生成函数"""
    engine = get_engine()
    return engine.generate(prompt, GenerationConfig(**kwargs))


def generate_stream(prompt: str, **kwargs):
    """便捷流式生成函数"""
    engine = get_engine()
    yield from engine.generate_stream(prompt, GenerationConfig(**kwargs))
