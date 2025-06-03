import re
import torch

from .spacing import extract_full_style_vector as extract_spacing_vector
from .naming import extract_naming_style_vector_v2 as extract_naming_vector
from .function_score import extract_function_structure_vector


def preprocess_code(code: str) -> str:
    lines = code.splitlines()
    cleaned_lines = []
    inside_docstring = False

    for line in lines:
        line_strip = line.strip()

        # 跳过多行 docstring 内容
        if inside_docstring:
            if '"""' in line_strip or "'''" in line_strip:
                inside_docstring = False
            continue

        # 跳过 docstring 开始
        if re.match(r'^\s*(\"\"\"|\'\'\')', line_strip):
            if line_strip.count('"""') == 2 or line_strip.count("'''") == 2:
                continue  # 单行 docstring
            else:
                inside_docstring = True
                continue

        # ✅ 保留真正的空行
        if line_strip == "":
            cleaned_lines.append("")
            continue

        # 跳过含自然语言的注释行
        if line_strip.startswith('#') and re.search(r'(example|calculate|function|returns|用于|实现|如下)', line_strip, re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    # ✅ 修复 def/class 后未缩进的问题
    fixed_lines = []
    i = 0
    while i < len(cleaned_lines):
        line = cleaned_lines[i]
        fixed_lines.append(line)

        # 自动补全 def/class 缩进逻辑
        if re.match(r'^\s*(def|class)\s+\w+\s*\(?.*?\)?:\s*$', line):
            if i + 1 >= len(cleaned_lines) or not cleaned_lines[i + 1].startswith((' ', '\t')):
                fixed_lines.append("    pass")
        i += 1

    code = '\n'.join(fixed_lines)

    # ✅ 括号平衡修复
    open_parens = code.count('(')
    close_parens = code.count(')')
    if open_parens > close_parens:
        code += ')' * (open_parens - close_parens)
    elif close_parens > open_parens:
        code = '(' * (close_parens - open_parens) + code

    return code



def extract_full_code_style_vector(code: str) -> torch.Tensor:
    """
    主函数：提取拼接后的风格向量（spacing + naming + structure）并调试打印
    """
    print("\n[DEBUG] 原始代码:")
    print(code)

    code_cleaned = preprocess_code(code)

    print("\n[DEBUG] 清洗后的代码:")
    print(code_cleaned)

    spacing_vec = extract_spacing_vector(code_cleaned)
    naming_vec = extract_naming_vector(code_cleaned)
    structure_vec = extract_function_structure_vector(code_cleaned)

    # 打印各段统计信息
    print(f"\n[DEBUG] spacing_vec ({len(spacing_vec)}维): {spacing_vec.tolist()}")
    print(f"[DEBUG] naming_vec  ({len(naming_vec)}维): {naming_vec.tolist()}")
    print(f"[DEBUG] structure_vec ({len(structure_vec)}维): {structure_vec.tolist()}")

    # 检查是否为异常向量
    def is_all_zero(tensor): return torch.all(tensor == 0).item()

    if is_all_zero(spacing_vec):
        print("[⚠️ 警告] spacing_vec 全为 0，请检查 spacing 提取逻辑或代码是否太简洁。")
    if is_all_zero(naming_vec):
        print("[⚠️ 警告] naming_vec 全为 0，请检查命名提取器是否识别失败。")
    if is_all_zero(structure_vec):
        print("[⚠️ 警告] structure_vec 全为 0，请检查是否无函数结构或 AST 失败。")

    return torch.cat([spacing_vec, naming_vec, structure_vec])
