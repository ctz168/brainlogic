#!/usr/bin/env python3
"""
智能数值推理引擎
Intelligent Numerical Reasoning Engine

核心思想：
不是用规则，而是实际执行数值计算
将计算结果注入到prompt中
让模型基于计算结果回答
"""

import re
import math
from typing import Dict, List, Optional, Any, Tuple


class NumericalReasoner:
    """智能数值推理器"""
    
    def __init__(self):
        pass
    
    def analyze_and_compute(self, text: str) -> Dict[str, Any]:
        """
        分析问题并执行计算
        
        返回：
        {
            'question_type': 问题类型,
            'parsed': 解析出的数据,
            'computation': 计算过程,
            'result': 计算结果,
            'verification': 验证过程
        }
        """
        result = {
            'question_type': None,
            'parsed': {},
            'computation': [],
            'result': None,
            'verification': []
        }
        
        # 1. 检测范围+奇偶数问题
        range_odd_even = self._parse_range_odd_even(text)
        if range_odd_even:
            result['question_type'] = 'range_odd_even'
            result['parsed'] = range_odd_even
            result['computation'], result['result'], result['verification'] =                 self._compute_range_odd_even(range_odd_even)
            return result
        
        # 2. 检测租金计算问题
        rent_calc = self._parse_rent_calc(text)
        if rent_calc:
            result['question_type'] = 'rent_calc'
            result['parsed'] = rent_calc
            result['computation'], result['result'], result['verification'] =                 self._compute_rent(rent_calc)
            return result
        
        # 3. 检测简单数学运算
        math_calc = self._parse_math_expr(text)
        if math_calc:
            result['question_type'] = 'math_calc'
            result['parsed'] = math_calc
            result['computation'], result['result'], result['verification'] =                 self._compute_math(math_calc)
            return result
        
        return result
    
    def _parse_range_odd_even(self, text: str) -> Optional[Dict]:
        """解析范围+奇偶数问题"""
        # 模式：a=X, b=Y, ... 最大/最小 奇数/偶数
        patterns = [
            # a=100, b=235, ... 最大/最小 奇数/偶数
            r'[a-zA-Z]\s*[=:]\s*(\d+)\s*,\s*[a-zA-Z]\s*[=:]\s*(\d+).*?(最大|最小).*(奇数|偶数)',
            # 在X和Y之间...最大/最小奇数/偶数
            r'(\d+).*?(\d+).*?之间.*?(最大|最小).*(奇数|偶数)',
            # X到Y...最大/最小奇数/偶数
            r'(\d+).*?(\d+).*?(最大|最小).*(奇数|偶数)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                
                # 提取数字
                nums = re.findall(r'\d+', text)
                if len(nums) >= 2:
                    start = int(nums[0])
                    end = int(nums[1])
                    
                    # 确定最大/最小
                    is_max = '最大' in text or 'max' in text.lower()
                    is_min = '最小' in text or 'min' in text.lower()
                    
                    # 确定奇数/偶数
                    is_odd = '奇数' in text or 'odd' in text.lower()
                    is_even = '偶数' in text or 'even' in text.lower()
                    
                    return {
                        'start': min(start, end),
                        'end': max(start, end),
                        'is_max': is_max and not is_min,
                        'is_min': is_min and not is_max,
                        'is_odd': is_odd and not is_even,
                        'is_even': is_even and not is_odd,
                        'original': text
                    }
        
        return None
    
    def _compute_range_odd_even(self, parsed: Dict) -> Tuple[List[str], Any, List[str]]:
        """计算范围奇偶数问题"""
        start = parsed['start']
        end = parsed['end']
        is_max = parsed['is_max']
        is_min = parsed['is_min']
        is_odd = parsed['is_odd']
        is_even = parsed['is_even']
        
        computation = []
        verification = []
        
        computation.append(f"步骤1: 确定范围 [{start}, {end}]")
        
        # 找出范围内所有符合条件的数
        candidates = []
        
        for n in range(start, end + 1):
            if is_odd and n % 2 == 1:
                candidates.append(n)
            elif is_even and n % 2 == 0:
                candidates.append(n)
        
        if is_odd:
            computation.append(f"步骤2: 找出范围内的奇数")
            computation.append(f"  奇数有: {candidates[:10]}{'...' if len(candidates) > 10 else ''}")
        elif is_even:
            computation.append(f"步骤2: 找出范围内的偶数")
            computation.append(f"  偶数有: {candidates[:10]}{'...' if len(candidates) > 10 else ''}")
        
        computation.append(f"  共 {len(candidates)} 个")
        
        # 找最大或最小
        if candidates:
            if is_max:
                result = max(candidates)
                computation.append(f"步骤3: 找最大值")
                computation.append(f"  最大值 = {result}")
            else:
                result = min(candidates)
                computation.append(f"步骤3: 找最小值")
                computation.append(f"  最小值 = {result}")
            
            # 验证
            verification.append(f"验证: {result} 在 [{start}, {end}] 范围内: {start <= result <= end}")
            if is_odd:
                verification.append(f"验证: {result} 是奇数: {result % 2 == 1}")
            elif is_even:
                verification.append(f"验证: {result} 是偶数: {result % 2 == 0}")
        else:
            result = None
            computation.append(f"步骤3: 范围内没有符合条件的数")
        
        return computation, result, verification
    
    def _parse_rent_calc(self, text: str) -> Optional[Dict]:
        """解析租金计算问题"""
        # 模式：X天房租Y元
        match = re.search(r'(\d+)\s*天\s*房租\s*(\d+)\s*元', text)
        if match:
            return {
                'days': int(match.group(1)),
                'total_rent': int(match.group(2)),
                'original': text
            }
        return None
    
    def _compute_rent(self, parsed: Dict) -> Tuple[List[str], Dict, List[str]]:
        """计算租金"""
        days = parsed['days']
        total = parsed['total_rent']
        
        computation = []
        verification = []
        
        daily = total / days
        monthly = daily * 30
        
        computation.append(f"步骤1: 计算日租金")
        computation.append(f"  日租金 = 总租金 ÷ 天数")
        computation.append(f"  日租金 = {total} ÷ {days} = {daily:.0f} 元/天")
        
        computation.append(f"步骤2: 计算月租金（按30天）")
        computation.append(f"  月租金 = 日租金 × 30")
        computation.append(f"  月租金 = {daily:.0f} × 30 = {monthly:.0f} 元")
        
        result = {
            'daily_rent': daily,
            'monthly_rent': monthly,
            'formula': f'日租金={total}÷{days}={daily:.0f}元，月租金={daily:.0f}×30={monthly:.0f}元'
        }
        
        verification.append(f"验证: 日租金 × 天数 = {daily:.0f} × {days} = {daily*days:.0f} ≈ {total}")
        
        return computation, result, verification
    
    def _parse_math_expr(self, text: str) -> Optional[Dict]:
        """解析数学表达式"""
        # 简单的四则运算
        match = re.search(r'(\d+)\s*([+\-×÷*/])\s*(\d+)', text)
        if match:
            a = int(match.group(1))
            op = match.group(2)
            b = int(match.group(3))
            return {
                'a': a,
                'op': op,
                'b': b,
                'original': text
            }
        return None
    
    def _compute_math(self, parsed: Dict) -> Tuple[List[str], Any, List[str]]:
        """计算数学表达式"""
        a = parsed['a']
        op = parsed['op']
        b = parsed['b']
        
        computation = []
        
        if op in ['+', '加']:
            result = a + b
            computation.append(f"{a} + {b} = {result}")
        elif op in ['-', '减']:
            result = a - b
            computation.append(f"{a} - {b} = {result}")
        elif op in ['×', '*', '乘']:
            result = a * b
            computation.append(f"{a} × {b} = {result}")
        elif op in ['÷', '/', '除']:
            result = a / b
            computation.append(f"{a} ÷ {b} = {result}")
        else:
            result = None
        
        return computation, result, []
    
    def format_for_prompt(self, analysis: Dict) -> str:
        """格式化为prompt"""
        if not analysis['question_type']:
            return ""
        
        lines = []
        lines.append("【系统计算结果】")
        
        if analysis['computation']:
            lines.append("\n计算过程:")
            for step in analysis['computation']:
                lines.append(f"  {step}")
        
        if analysis['result'] is not None:
            if isinstance(analysis['result'], dict):
                lines.append(f"\n结果: {analysis['result'].get('formula', analysis['result'])}")
            else:
                lines.append(f"\n答案: {analysis['result']}")
        
        if analysis['verification']:
            lines.append("\n验证:")
            for v in analysis['verification']:
                lines.append(f"  {v}")
        
        return "\n".join(lines)


# 测试
if __name__ == "__main__":
    reasoner = NumericalReasoner()
    
    test_cases = [
        "a=100, b=235，c是它中间的一个最大偶数，是什么？",
        "a=100, b=235，d是它中间的一个最大奇数，是什么？",
        "a=100, b=235，d是它中间的一个最小奇数，是什么？",
        "房租1600元租了20天，月租是多少？",
    ]
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"问题: {test}")
        print("=" * 60)
        
        result = reasoner.analyze_and_compute(test)
        print(reasoner.format_for_prompt(result))
