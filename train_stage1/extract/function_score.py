import ast
import torch
import re

def count_return_statements(node):
    return sum(isinstance(n, ast.Return) for n in ast.walk(node))

def has_try_except(node):
    return any(isinstance(n, ast.Try) for n in ast.walk(node))

def get_call_depth(node, depth=0):
    if isinstance(node, ast.Call):
        return max([get_call_depth(arg, depth + 1) for arg in ast.iter_child_nodes(node)] + [depth])
    else:
        return max([get_call_depth(child, depth) for child in ast.iter_child_nodes(node)] + [depth])

def get_control_structure_count(node):
    return sum(isinstance(n, (ast.If, ast.For, ast.While)) for n in ast.walk(node))

def get_comment_ratio(func_node, source_lines):
    if hasattr(func_node, 'lineno') and hasattr(func_node, 'end_lineno'):
        func_lines = source_lines[func_node.lineno - 1:func_node.end_lineno]
        total_lines = len(func_lines)
        comment_lines = sum(1 for line in func_lines if line.strip().startswith('#'))
        return comment_lines / total_lines if total_lines else 0.0
    return 0.0

def get_annotation_ratio(func_node):
    args = func_node.args.args
    if not args:
        return 1.0
    annotated = sum(1 for arg in args if arg.annotation is not None)
    return annotated / len(args)

def get_exception_specificity(func_node):
    specificity = 1.0
    for node in ast.walk(func_node):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                specificity -= 0.5
            elif isinstance(node.type, ast.Name) and node.type.id in ['Exception', 'BaseException']:
                specificity -= 0.25
    return max(0.0, specificity)

def get_repetition_ratio(func_node, source_lines):
    if hasattr(func_node, 'lineno') and hasattr(func_node, 'end_lineno'):
        func_lines = source_lines[func_node.lineno - 1:func_node.end_lineno]
        norm_lines = [line.strip() for line in func_lines if line.strip()]
        total = len(norm_lines)
        unique = len(set(norm_lines))
        return (total - unique) / total if total else 0.0
    return 0.0

def normalize(value, max_val):
    return min(value / max_val, 1.0)

def extract_features_from_function(func_node, source_lines):
    length = func_node.end_lineno - func_node.lineno + 1 if hasattr(func_node, 'end_lineno') else 0
    return [
        normalize(len(func_node.args.args), 10),
        int(ast.get_docstring(func_node) is not None),
        int(has_try_except(func_node)),
        normalize(count_return_statements(func_node), 5),
        normalize(get_call_depth(func_node), 5),
        normalize(get_control_structure_count(func_node), 5),
        get_annotation_ratio(func_node),
        get_exception_specificity(func_node),
        get_repetition_ratio(func_node, source_lines),
        get_comment_ratio(func_node, source_lines),
        normalize(length, 100)  # 新增：函数行数（最多100行归一化）
    ]
def extract_function_structure_vector(code: str) -> torch.Tensor:
    source_lines = code.splitlines()

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print("[AST PARSE FAILED]")
        print(">> 错误原因:", e)
        print(">> 出错代码:\n", code)
        
        # fallback 简化向量
        comment_ratio = sum(1 for line in source_lines if line.strip().startswith('#')) / len(source_lines) if source_lines else 0.0
        norm_lines = [line.strip() for line in source_lines if line.strip()]
        total = len(norm_lines)
        unique = len(set(norm_lines))
        repetition_ratio = (total - unique) / total if total else 0.0
        approx_length = normalize(len(source_lines), 100)
        return torch.tensor([0.0] * 8 + [repetition_ratio, comment_ratio, approx_length], dtype=torch.float32)
    feature_list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            feature_list.append(extract_features_from_function(node, source_lines))

    if not feature_list:
        # ✅ 没有函数，返回默认空特征
        return torch.tensor([0.0] * 11, dtype=torch.float32)

    # ✅ 多个函数，取均值
    features = torch.tensor(feature_list, dtype=torch.float32)
    return features.mean(dim=0)

