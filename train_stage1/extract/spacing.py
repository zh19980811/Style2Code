import numpy as np
import torch
from itertools import groupby
from scipy.stats import skew, kurtosis
import re

def is_logical_blank(line: str) -> bool:
    """
    判断是否为逻辑空行：即 strip 后为空，或只包含注释、符号等无实际语义的行。
    """
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#") and not re.search(r'\w+', stripped):
        return True
    return False

def get_space_binary_vector(line, max_len=30):
    binary = ['1' if c == ' ' else '0' for c in line[:max_len]]
    binary += ['0'] * (max_len - len(binary))
    return [int(b) for b in binary]

def extract_full_style_vector(code: str, max_len=30, max_lines=200):
    lines = code.splitlines()
    total_lines = len(lines) or 1

    # 1. 空行 + 注释行为逻辑空行（4维，新增一个是否存在逻辑空行的标志位）
    blank_mask = [1 if is_logical_blank(line) else 0 for line in lines]
    blank_indices = [i for i, v in enumerate(blank_mask) if v == 1]
    mean_pos = np.mean(blank_indices) / total_lines if blank_indices else 0.0
    std_pos = np.std(blank_indices) / total_lines if blank_indices else 0.0
    runs = [len(list(g)) for k, g in groupby(blank_mask) if k == 1]
    max_run = max(runs) / total_lines if runs else 0.0
    has_blank = torch.tensor([1.0 if any(blank_mask) else 0.0], dtype=torch.float32)
    blank_line_feat = torch.tensor([mean_pos, std_pos, max_run], dtype=torch.float32)
    spacing_blank_feat = torch.cat([blank_line_feat, has_blank])
    
    # 2. 空格风格特征（5维）
    binary_matrix = [get_space_binary_vector(line, max_len) for line in lines if line.strip()]
    if not binary_matrix:
        space_feat = torch.zeros(5, dtype=torch.float32)
    else:
        binary_matrix = np.array(binary_matrix)
        mean_space = np.mean(binary_matrix)
        std_space = np.std(binary_matrix)
        max_run_lengths = [
            max((sum(1 for _ in g) for k, g in groupby(row) if k == 1), default=0)
            for row in binary_matrix
        ]
        max_run_length = max(max_run_lengths) / max_len if max_run_lengths else 0.0
        skewness = skew(binary_matrix.flatten())
        kurt = kurtosis(binary_matrix.flatten())
        space_feat = torch.tensor([mean_space, std_space, max_run_length, skewness, kurt], dtype=torch.float32)

    # 3. 行数归一化（1维）
    line_count = torch.tensor([min(len(lines) / max_lines, 1.0)], dtype=torch.float32)

    # 拼接最终向量
    return torch.cat([blank_line_feat, space_feat, line_count])
