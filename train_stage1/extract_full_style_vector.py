import re
import torch

from spacing import extract_full_style_vector as extract_spacing_vector
from naming import extract_naming_style_vector_v2 as extract_naming_vector
from function_score import extract_function_structure_vector

def preprocess_code(code: str) -> str:
    """
    清洗输入代码：保留注释，去除自然语言说明，补全 def/class 结构，修复括号不平衡。
    """
    lines = code.splitlines()

    cleaned_lines = []
    inside_docstring = False
    for line in lines:
        line_strip = line.strip()

        # 去除多行字符串/AI生成段落
        if inside_docstring:
            if '"""' in line_strip or "'''" in line_strip:
                inside_docstring = False
            continue
        if re.match(r'^\s*(\"\"\"|\'\'\')', line_strip):
            inside_docstring = True
            continue

        # 去除自然语言描述句子
        if re.search(r'(example|calculate|this function|returns|用于|实现|如下)', line_strip, re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    # 补全不完整 def/class 定义
    fixed_lines = []
    for line in cleaned_lines:
        if re.match(r'^\s*def\s+\w+\s*\(.*\)\s*$', line):
            fixed_lines.append(line + ':')
            fixed_lines.append('    pass')
        elif re.match(r'^\s*class\s+\w+\s*$', line):
            fixed_lines.append(line + ':')
            fixed_lines.append('    pass')
        else:
            fixed_lines.append(line)

    code = '\n'.join(fixed_lines)

    # 括号平衡修复
    open_parens = code.count('(')
    close_parens = code.count(')')
    if open_parens > close_parens:
        code += ')' * (open_parens - close_parens)
    elif close_parens > open_parens:
        code = '(' * (close_parens - open_parens) + code

    return code


def extract_full_code_style_vector(code: str) -> torch.Tensor:
    """
    主函数：提取拼接后的风格向量（spacing + naming + structure）
    """
    code_cleaned = preprocess_code(code)

    spacing_vec = extract_spacing_vector(code_cleaned)                 # 9 + N
    naming_vec = extract_naming_vector(code_cleaned)                   # 14
    structure_vec = extract_function_structure_vector(code_cleaned)   # 10

    return torch.cat([spacing_vec, naming_vec, structure_vec])
