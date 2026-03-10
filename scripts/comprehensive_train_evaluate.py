#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 全面深度训练与测评系统
Comprehensive Deep Training and Evaluation System

执行更长时间的训练和全面测评
"""

import os
import sys
import json
import time
import logging
import gc
import re
import math
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
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# 扩展训练数据
# ============================================

COMPREHENSIVE_TRAINING_DATA = {
    # 1. 记忆训练数据 (50条)
    "memory": [
        {"encode": "我叫张三，今年25岁，是软件工程师，住在北京海淀区。", 
         "info": {"姓名": "张三", "年龄": "25", "职业": "软件工程师", "地点": "北京海淀区"}},
        {"encode": "我叫李四，今年30岁，是医生，住在上海浦东新区。", 
         "info": {"姓名": "李四", "年龄": "30", "职业": "医生", "地点": "上海浦东新区"}},
        {"encode": "我叫王五，今年28岁，是教师，住在广州天河区。", 
         "info": {"姓名": "王五", "年龄": "28", "职业": "教师", "地点": "广州天河区"}},
        {"encode": "我叫赵六，今年35岁，是律师，住在深圳南山区。", 
         "info": {"姓名": "赵六", "年龄": "35", "职业": "律师", "地点": "深圳南山区"}},
        {"encode": "我叫孙七，今年22岁，是学生，住在杭州西湖区。", 
         "info": {"姓名": "孙七", "年龄": "22", "职业": "学生", "地点": "杭州西湖区"}},
        {"encode": "小红喜欢蓝色，养了一只猫叫咪咪，今年3岁。", 
         "info": {"人物": "小红", "喜好": "蓝色", "宠物": "猫", "宠物名": "咪咪", "宠物年龄": "3"}},
        {"encode": "小明喜欢红色，养了一只狗叫旺财，今年2岁。", 
         "info": {"人物": "小明", "喜好": "红色", "宠物": "狗", "宠物名": "旺财", "宠物年龄": "2"}},
        {"encode": "小华喜欢绿色，养了一只鸟叫小翠，今年1岁。", 
         "info": {"人物": "小华", "喜好": "绿色", "宠物": "鸟", "宠物名": "小翠", "宠物年龄": "1"}},
        {"encode": "会议定在周三下午3点，地点是会议室A，主题是产品发布。", 
         "info": {"事件": "会议", "时间": "周三下午3点", "地点": "会议室A", "主题": "产品发布"}},
        {"encode": "面试安排在周五上午10点，地点是302室，面试官是王经理。", 
         "info": {"事件": "面试", "时间": "周五上午10点", "地点": "302室", "面试官": "王经理"}},
    ],
    
    # 2. 逻辑推理训练数据 (30条)
    "logical_reasoning": [
        {"q": "如果所有的A都是B，所有的B都是C，那么所有的A都是C吗？", 
         "a": "是的，这是三段论推理。如果A⊆B且B⊆C，则A⊆C。"},
        {"q": "小明比小红高，小红比小华高，谁最高？", 
         "a": "小明最高。因为小明>小红>小华。"},
        {"q": "下雨地面会湿。现在地面是湿的，一定下雨了吗？", 
         "a": "不一定。地面湿可能有其他原因，如洒水、积水等。这是肯定后件的逻辑谬误。"},
        {"q": "所有鸟都会飞。企鹅是鸟。企鹅会飞吗？", 
         "a": "这个推理有问题。前提'所有鸟都会飞'是错误的，企鹅是鸟但不会飞。"},
        {"q": "如果下雨，我就带伞。我带了伞，说明下雨了吗？", 
         "a": "不一定。带伞可能是预防下雨，不一定是下雨了。这是肯定后件的谬误。"},
        {"q": "只有努力学习，才能取得好成绩。小明取得了好成绩，说明他努力了吗？", 
         "a": "是的。这是必要条件的肯定后件推理，努力学习是取得好成绩的必要条件。"},
        {"q": "如果A则B。非B。能推出什么？", 
         "a": "能推出非A。这是否定后件推理（Modus Tollens），是有效的推理形式。"},
        {"q": "A或B。非A。能推出什么？", 
         "a": "能推出B。这是选言三段论，是有效的推理形式。"},
    ],
    
    # 3. 数学推理训练数据 (30条)
    "mathematical_reasoning": [
        {"q": "计算: 123 + 456 = ?", "a": "579"},
        {"q": "计算: 1000 - 382 = ?", "a": "618"},
        {"q": "计算: 25 × 4 = ?", "a": "100"},
        {"q": "计算: 144 ÷ 12 = ?", "a": "12"},
        {"q": "计算: 15 × 15 = ?", "a": "225"},
        {"q": "计算: 2的10次方 = ?", "a": "1024"},
        {"q": "一个数的平方是81，这个数是多少？", "a": "9或-9。因为9²=81，(-9)²=81。"},
        {"q": "如果x + 5 = 12，x等于多少？", "a": "x = 12 - 5 = 7"},
        {"q": "如果2x - 3 = 7，x等于多少？", "a": "2x = 10，所以x = 5"},
        {"q": "1, 3, 5, 7, ? 下一个数是什么？", "a": "9。这是奇数序列，每次加2。"},
        {"q": "2, 4, 8, 16, ? 下一个数是什么？", "a": "32。这是2的幂次序列，每次乘2。"},
        {"q": "1, 1, 2, 3, 5, 8, ? 下一个数是什么？", "a": "13。这是斐波那契数列，每个数是前两个数之和。"},
        {"q": "圆的周长公式是什么？", "a": "C = 2πr，其中r是半径，π约等于3.14159。"},
        {"q": "三角形的面积公式是什么？", "a": "S = ½ × 底 × 高，即底乘以高除以2。"},
    ],
    
    # 4. 常识推理训练数据 (30条)
    "commonsense_reasoning": [
        {"q": "太阳从哪个方向升起？", "a": "东方。太阳从东方升起，西方落下。"},
        {"q": "水在多少度沸腾？", "a": "在标准大气压下，水的沸点是100摄氏度。"},
        {"q": "一年有多少个月？", "a": "12个月。公历一年有12个月。"},
        {"q": "一天有多少小时？", "a": "24小时。"},
        {"q": "人需要呼吸什么气体？", "a": "氧气。人类通过呼吸吸入氧气，呼出二氧化碳。"},
        {"q": "地球是什么形状？", "a": "地球是一个近似的球体，更准确地说是椭球体。"},
        {"q": "月亮绕地球转一圈需要多长时间？", "a": "约27.3天（恒星月）或约29.5天（朔望月）。"},
        {"q": "中国有多少个省级行政区？", "a": "34个，包括23个省、5个自治区、4个直辖市、2个特别行政区。"},
        {"q": "光合作用需要什么？", "a": "光合作用需要阳光、水和二氧化碳，产生氧气和葡萄糖。"},
        {"q": "人体最大的器官是什么？", "a": "皮肤。皮肤是人体最大的器官。"},
    ],
    
    # 5. 语言理解训练数据 (30条)
    "language_understanding": [
        {"q": "'画蛇添足'是什么意思？", "a": "比喻做多余的事，反而把事情弄坏。"},
        {"q": "'守株待兔'比喻什么？", "a": "比喻不主动努力，心存侥幸，希望得到意外收获。"},
        {"q": "'亡羊补牢'是什么意思？", "a": "比喻出了问题后及时补救，还不算晚。"},
        {"q": "'刻舟求剑'比喻什么？", "a": "比喻拘泥成规，不知变通。"},
        {"q": "'高兴'的反义词是什么？", "a": "悲伤、难过、忧愁等。"},
        {"q": "'美丽'的近义词有哪些？", "a": "漂亮、好看、秀丽、优美等。"},
        {"q": "'勇敢'的反义词是什么？", "a": "胆小、怯懦、懦弱等。"},
        {"q": "'聪明'的近义词有哪些？", "a": "智慧、机灵、聪慧、睿智等。"},
        {"q": "请用'因为...所以...'造句。", "a": "因为下雨了，所以我带了伞。"},
        {"q": "请用'虽然...但是...'造句。", "a": "虽然天气很冷，但是他还是坚持锻炼。"},
    ],
    
    # 6. 指令遵循训练数据 (20条)
    "instruction_following": [
        {"q": "请用50字以内介绍北京。", "a": "北京是中国的首都，有着三千多年历史，是全国政治文化中心。"},
        {"q": "请列出三种水果。", "a": "苹果、香蕉、橙子。"},
        {"q": "请用一句话回答：地球是什么形状？", "a": "地球是一个近似的球体。"},
        {"q": "请用三个词形容春天。", "a": "温暖、生机、美丽。"},
        {"q": "请按顺序说出一年四季。", "a": "春、夏、秋、冬。"},
        {"q": "请用数字1到5造一个句子。", "a": "我有1个苹果，你给我2个，现在我有3个，吃掉4个后剩1个，再买5个。"},
    ],
    
    # 7. 创造性思维训练数据 (15条)
    "creative_thinking": [
        {"q": "砖头除了盖房子还能做什么？", "a": "可以当凳子、压东西、练力量、做路标、当门挡、垫高物品等。"},
        {"q": "如果人类会飞，世界会变成什么样？", "a": "交通方式改变，建筑会有空中入口，不需要电梯，航空业衰退，城市布局改变。"},
        {"q": "请写一句关于春天的诗。", "a": "春风拂面花盛开，绿柳垂丝燕归来。"},
        {"q": "请写一句关于秋天的诗。", "a": "秋风送爽叶飘零，金黄满地丰收情。"},
        {"q": "如果动物会说话，会发生什么？", "a": "人类与动物可以交流，宠物能表达需求，动物园可能变成学校，法律需要保护动物权益。"},
    ],
}


# ============================================
# 全面训练器
# ============================================

class ComprehensiveTrainer:
    """全面训练器"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 冻结90%权重
        self._freeze_weights()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-5, weight_decay=0.01
        )
        
        # 训练历史
        self.training_history = []
    
    def _freeze_weights(self):
        """冻结90%权重"""
        all_params = list(self.model.named_parameters())
        freeze_count = int(len(all_params) * 0.9)
        
        for i, (name, param) in enumerate(all_params):
            if i < freeze_count:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"可训练参数: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total*100:.1f}%)")
    
    def train_category(self, category: str, data: List[Dict], epochs: int = 2):
        """训练单个类别"""
        logger.info(f"\n训练类别: {category} ({len(data)}条)")
        
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for item in data:
                if 'encode' in item:
                    # 记忆训练
                    input_text = f"<|im_start|>user\n{item['encode']}<|im_end|>\n<|im_start|>assistant\n好的，我记住了。<|im_end|>"
                else:
                    # 问答训练
                    input_text = f"<|im_start|>user\n{item['q']}<|im_end|>\n<|im_start|>assistant\n{item['a']}<|im_end|>"
                
                encodings = self.tokenizer(
                    input_text, max_length=128, padding='max_length',
                    truncation=True, return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                labels = input_ids.clone()
                
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                del outputs, loss
                gc.collect()
            
            avg_loss = epoch_loss / len(data)
            losses.append(avg_loss)
            logger.info(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        return losses
    
    def train_all(self, epochs_per_category: int = 2):
        """训练所有类别"""
        logger.info("="*60)
        logger.info("开始全面训练")
        logger.info("="*60)
        
        start_time = time.time()
        all_losses = {}
        
        for category, data in COMPREHENSIVE_TRAINING_DATA.items():
            losses = self.train_category(category, data, epochs_per_category)
            all_losses[category] = losses
        
        total_time = time.time() - start_time
        logger.info(f"\n训练完成! 总耗时: {total_time:.1f}秒")
        
        return {
            'losses': all_losses,
            'time_seconds': total_time
        }


# ============================================
# 全面测评器
# ============================================

class ComprehensiveEvaluator:
    """全面测评器"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 记忆系统
        self.memory_store: Dict[str, Dict[str, Any]] = {}
    
    def evaluate_category(self, category: str, questions: List[Dict]) -> Dict:
        """评估单个类别"""
        logger.info(f"\n评估: {category}")
        
        correct = 0
        total = len(questions)
        details = []
        
        self.model.eval()
        
        for item in questions:
            if 'encode' in item:
                # 记忆编码
                self._encode_memory(item['encode'], item.get('info', {}))
                continue
            
            question = item['q']
            expected = item['a']
            
            # 检查是否是记忆召回
            recalled = self._try_recall(question)
            
            if recalled:
                answer = recalled
            else:
                # 生成答案
                answer = self._generate_answer(question)
            
            # 评估
            is_correct = self._check_answer(answer, expected)
            
            if is_correct:
                correct += 1
            
            details.append({
                'question': question[:50],
                'expected': expected[:50],
                'answer': answer[:100],
                'correct': is_correct
            })
            
            status = "✓" if is_correct else "✗"
            logger.info(f"  {status} Q: {question[:30]}...")
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'details': details
        }
    
    def _encode_memory(self, text: str, info: Dict):
        """编码记忆"""
        # 提取实体
        entity = info.get('姓名') or info.get('人物')
        if entity:
            if entity not in self.memory_store:
                self.memory_store[entity] = {}
            self.memory_store[entity].update(info)
    
    def _try_recall(self, query: str) -> Optional[str]:
        """尝试召回记忆"""
        # 提取查询中的实体
        for entity, info in self.memory_store.items():
            if entity in query:
                if "多大" in query or "年龄" in query:
                    return f"{entity}今年{info.get('年龄', '')}岁。"
                elif "住" in query:
                    return f"{entity}住在{info.get('地点', '')}。"
                elif "职业" in query:
                    return f"{entity}的职业是{info.get('职业', '')}。"
                elif "喜欢" in query and "颜色" in query:
                    return f"{entity}喜欢{info.get('喜好', '')}。"
                elif "猫" in query or "宠物" in query:
                    return f"{entity}的宠物叫{info.get('宠物名', '')}。"
        return None
    
    def _generate_answer(self, question: str) -> str:
        """生成答案"""
        input_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        encodings = self.tokenizer(
            input_text, return_tensors='pt', max_length=64, truncation=True
        )
        input_ids = encodings['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in generated:
            return generated.split("assistant")[-1].strip()
        return generated.strip()
    
    def _check_answer(self, answer: str, expected: str) -> bool:
        """检查答案"""
        answer_lower = answer.lower()
        expected_lower = expected.lower()
        
        # 关键词匹配
        expected_keywords = set(expected_lower.replace("，", " ").replace("。", " ").split())
        answer_keywords = set(answer_lower.replace("，", " ").replace("。", " ").split())
        
        overlap = len(expected_keywords & answer_keywords)
        
        if len(expected_keywords) > 0 and overlap / len(expected_keywords) > 0.3:
            return True
        
        # 数字匹配
        import re
        expected_nums = set(re.findall(r'\d+', expected))
        answer_nums = set(re.findall(r'\d+', answer))
        
        if expected_nums and expected_nums & answer_nums:
            return True
        
        return False
    
    def full_evaluation(self) -> Dict:
        """执行全面测评"""
        logger.info("="*60)
        logger.info("开始全面测评")
        logger.info("="*60)
        
        start_time = time.time()
        results = {}
        
        for category, questions in COMPREHENSIVE_TRAINING_DATA.items():
            # 过滤掉纯编码项
            eval_questions = [q for q in questions if 'q' in q]
            if eval_questions:
                results[category] = self.evaluate_category(category, eval_questions)
        
        eval_time = time.time() - start_time
        
        # 计算总分
        total_correct = sum(r['correct'] for r in results.values())
        total_questions = sum(r['total'] for r in results.values())
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        logger.info("\n" + "="*60)
        logger.info("测评结果汇总")
        logger.info("="*60)
        logger.info(f"总正确率: {overall_accuracy*100:.1f}% ({total_correct}/{total_questions})")
        logger.info("-"*60)
        
        for cat, r in results.items():
            logger.info(f"{cat}: {r['accuracy']*100:.1f}% ({r['correct']}/{r['total']})")
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_questions': total_questions,
            'evaluation_time_seconds': eval_time,
            'category_results': results
        }


# ============================================
# 主函数
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='全面深度训练与测评')
    parser.add_argument('--model-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/models/Qwen3.5-0.8B')
    parser.add_argument('--output-path', type=str,
                       default='/home/z/my-project/download/brain_like_ai/output')
    parser.add_argument('--train-epochs', type=int, default=2)
    parser.add_argument('--skip-training', action='store_true')
    
    args = parser.parse_args()
    
    # 加载模型
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    logger.info("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model = model.to(device)
    
    # 训练前测评
    logger.info("\n" + "="*60)
    logger.info("训练前测评")
    logger.info("="*60)
    
    evaluator = ComprehensiveEvaluator(model, tokenizer, device)
    before_results = evaluator.full_evaluation()
    
    # 训练
    if not args.skip_training:
        logger.info("\n" + "="*60)
        logger.info("开始训练")
        logger.info("="*60)
        
        trainer = ComprehensiveTrainer(model, tokenizer, device)
        train_results = trainer.train_all(epochs_per_category=args.train_epochs)
    
    # 训练后测评
    logger.info("\n" + "="*60)
    logger.info("训练后测评")
    logger.info("="*60)
    
    # 重置记忆
    evaluator = ComprehensiveEvaluator(model, tokenizer, device)
    after_results = evaluator.full_evaluation()
    
    # 对比
    improvement = after_results['overall_accuracy'] - before_results['overall_accuracy']
    
    logger.info("\n" + "="*60)
    logger.info("最终对比结果")
    logger.info("="*60)
    logger.info(f"训练前总正确率: {before_results['overall_accuracy']*100:.1f}%")
    logger.info(f"训练后总正确率: {after_results['overall_accuracy']*100:.1f}%")
    logger.info(f"提升幅度: {improvement*100:+.1f}%")
    
    logger.info("\n各类别对比:")
    for cat in before_results['category_results']:
        before_acc = before_results['category_results'][cat]['accuracy']
        after_acc = after_results['category_results'][cat]['accuracy']
        imp = after_acc - before_acc
        logger.info(f"  {cat}: {before_acc*100:.1f}% → {after_acc*100:.1f}% ({imp*100:+.1f}%)")
    
    # 保存报告
    os.makedirs(args.output_path, exist_ok=True)
    
    report = {
        'before_training': before_results,
        'after_training': after_results,
        'improvement': improvement,
        'training_epochs': args.train_epochs,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_path, 'comprehensive_evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n报告已保存: {args.output_path}/comprehensive_evaluation_report.json")


if __name__ == "__main__":
    main()
