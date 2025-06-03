import torch
import ast
import re
import keyword
import numpy as np
from collections import defaultdict, Counter

class NameExtractor(ast.NodeVisitor):
    def __init__(self):
        self.names = defaultdict(set)

    def visit_FunctionDef(self, node):
        self.names["function"].add(node.name)
        for arg in node.args.args:
            self.names["parameter"].add(arg.arg)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.names["class"].add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        self.names["variable"].add(node.id)

    def visit_Attribute(self, node):
        self.names["attribute"].add(node.attr)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.isidentifier():
            self.names["constant"].add(node.value)

    def visit(self, node):
        try:
            super().visit(node)
        except RecursionError:
            pass  # 防止极端结构引发递归溢出


def classify_naming_style(name):
    if name.startswith('__') and name.endswith('__'):
        return 'dunder_method'
    if name.startswith('_') and not name.startswith('__'):
        return 'private'
    if re.fullmatch(r'[a-z]+(_[a-z0-9]+)*', name):
        return 'snake_case'
    if re.fullmatch(r'[A-Z][a-zA-Z0-9]+', name):
        return 'PascalCase'
    if re.fullmatch(r'[a-z]+([A-Z][a-z0-9]*)+', name):
        return 'camelCase'
    if re.fullmatch(r'[A-Z]+(_[A-Z0-9]+)*', name):
        return 'UPPER_CASE'
    return 'invalid'

def compute_style_distribution(names):
    styles = [classify_naming_style(name) for name in names]
    counter = Counter(styles)
    all_styles = ['snake_case', 'camelCase', 'PascalCase', 'UPPER_CASE', 'dunder_method', 'private']
    total = sum(counter.values())
    return [counter.get(style, 0) / total if total else 0.0 for style in all_styles]

def compute_style_stats(names):
    total_chars = sum(len(name) for name in names)
    stats = {'uppercase': 0, 'lowercase': 0, 'underscore': 0, 'digit': 0, 'symbol': 0}
    for name in names:
        for char in name:
            if char.isupper():
                stats['uppercase'] += 1
            elif char.islower():
                stats['lowercase'] += 1
            elif char == '_':
                stats['underscore'] += 1
            elif char.isdigit():
                stats['digit'] += 1
            else:
                stats['symbol'] += 1
    return [
        round(stats[k] / total_chars, 3) if total_chars else 0.0
        for k in ['uppercase', 'lowercase', 'underscore', 'digit', 'symbol']
    ]

def check_naming_rules(name: str, kind: str) -> float:
    score = 0
    total = 0
    total += 1
    score += int(name[0].isalpha())

    if kind == 'class':
        total += 1
        score += bool(re.match(r'^[A-Z][a-zA-Z0-9]+$', name))
    elif kind in ['function', 'variable', 'parameter', 'attribute']:
        total += 1
        score += bool(re.match(r'^[a-z]+(_[a-z0-9]+)*$', name))
    elif kind == 'constant':
        total += 1
        score += bool(re.match(r'^[A-Z]+(_[A-Z0-9]+)*$', name))
    elif kind == 'magic_method':
        total += 1
        score += (name.startswith('__') and name.endswith('__'))
    elif kind == 'private':
        total += 1
        score += (name.startswith('_') and not name.startswith('__'))

    total += 1
    score += int(name not in keyword.kwlist)

    return round(score / total, 2)

def extract_names_by_regex(code: str):
    return re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)

def extract_naming_style_vector_v2(code: str) -> torch.Tensor:
    try:
        tree = ast.parse(code)
        extractor = NameExtractor()
        extractor.visit(tree)

        all_names = []
        for kind, names in extractor.names.items():
            all_names.extend(list(names))

        dist_vector = torch.tensor(compute_style_distribution(all_names), dtype=torch.float32)
        ratio_vector = torch.tensor(compute_style_stats(all_names), dtype=torch.float32)

        total_names = len(all_names)
        avg_length = np.mean([len(name) for name in all_names]) if all_names else 0.0
        avg_score = np.mean([
            check_naming_rules(name, kind)
            for kind, names in extractor.names.items()
            for name in names
        ]) if all_names else 0.0

        # 归一化 total_names 和 avg_length
        total_names_norm = min(total_names / 100, 1.0)
        avg_length_norm = min(avg_length / 30, 1.0)
        stats_vector = torch.tensor([total_names_norm, avg_length_norm, avg_score], dtype=torch.float32)

        return torch.cat([dist_vector, ratio_vector, stats_vector])  # 14维

    except SyntaxError:
        # ✅ fallback 用正则提取所有可能命名词
        names = extract_names_by_regex(code)
        dist_vector = torch.zeros(6, dtype=torch.float32)  # AST依赖部分为0
        ratio_vector = torch.tensor(compute_style_stats(names), dtype=torch.float32)

        total_names = len(names)
        avg_length = np.mean([len(name) for name in names]) if names else 0.0
        total_names_norm = min(total_names / 100, 1.0)
        avg_length_norm = min(avg_length / 30, 1.0)
        avg_score = 0.0  # 无法判断语法角色，跳过 check_naming_rules

        stats_vector = torch.tensor([total_names_norm, avg_length_norm, avg_score], dtype=torch.float32)
        return torch.cat([dist_vector, ratio_vector, stats_vector])  # 14维
