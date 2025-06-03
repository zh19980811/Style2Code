import ast
import pandas as pd
import matplotlib.pyplot as plt
from benchmark.function_score import get_function_score_quantified
from benchmark.spacing_analysis import (
    compute_kl_divergence_dtw,
    calculate_kl_divergence_dtw,
    extract_space_alignment   # ← 需要这个！
)
from benchmark.naming import compute_style_stats, compute_style_distribution, NameExtractor


def analyze_naming(code):
    tree = ast.parse(code)
    extractor = NameExtractor()
    extractor.visit(tree)
    names = extractor.names
    all_names = [n for group in names.values() for n in group]
    return {
        "style_stats": {"code": compute_style_stats(all_names)},
        "style_distribution": {"code": compute_style_distribution(all_names)}
    }


def extract_function_metrics(code_str, label):
    tree = ast.parse(code_str)
    lines = code_str.splitlines()
    rows = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            score, metrics = get_function_score_quantified(node, lines, verbose=True)
            row = {"function_name": node.name, "source": label, "score": score}
            row.update(metrics)
            rows.append(row)
    return rows


def extract_spacing_metrics(code1, code2):
    empty_line_kl = compute_kl_divergence_dtw(code1, code2)
    space_kl = calculate_kl_divergence_dtw(code1, code2)

    # 新增空格分布编码和空行编码
    aligned1, aligned2, empty1, empty2 = extract_space_alignment(code1, code2)
    return {
        "space_pattern_code2": ' '.join(aligned2),

    }


def extract_naming_metrics(code, label):
    result = analyze_naming(code)
    style_stats = result.get("style_stats", {}).get("code", {})
    style_dist = result.get("style_distribution", {}).get("code", {})
    flat = {}
    for k, v in style_stats.items():
        flat[f"style_stat_{k}"] = v
    for k, v in style_dist.items():
        flat[f"style_dist_{k}"] = v
    return flat


def export_code_metrics(code1: str, code2: str, output_csv: str = "function_style_metrics.csv"):
    spacing_metrics = extract_spacing_metrics(code1, code2)
    naming1 = extract_naming_metrics(code1, "code1")
    naming2 = extract_naming_metrics(code2, "code2")

    rows1 = extract_function_metrics(code1, "code1")
    rows2 = extract_function_metrics(code2, "code2")

    for row in rows1:
        row.update(spacing_metrics)
        row.update(naming1)
    for row in rows2:
        row.update(spacing_metrics)
        row.update(naming2)

    df = pd.DataFrame(rows1 + rows2)
    df.to_csv(output_csv, index=False)
    print(f"✅ 导出成功: {output_csv}")
    return df


def visualize_metrics(df):
    import matplotlib
    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['axes.unicode_minus'] = False

    import numpy as np
    from math import pi

    # 图1：雷达图（前两个函数）
    selected = df.head(2)
    features = [col for col in df.columns if col not in ["function_name", "source", "score"] and df[col].dtype != object]
    values = selected[features].values

    plt.figure(figsize=(6, 6))
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    values = np.concatenate((values, values[:, [0]]), axis=1)
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)
    for i, row in enumerate(selected.itertuples()):
        ax.plot(angles, values[i], label=row.function_name)
        ax.fill(angles, values[i], alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), features, fontsize=8)
    plt.legend(loc='upper right')
    plt.title("函数风格雷达图")
    plt.tight_layout()
    plt.show()

    # 图2：总分柱状图
    plt.figure(figsize=(8, 4))
    df_sorted = df.sort_values("score", ascending=False)
    plt.bar(df_sorted["function_name"], df_sorted["score"], color="skyblue")
    plt.ylabel("综合得分")
    plt.title("函数风格评分对比")
    plt.tight_layout()
    plt.show()

    # 图3：注释率 + 嵌套深度对比
    plt.figure(figsize=(6, 4))
    for label in df["source"].unique():
        subset = df[df["source"] == label]
        plt.scatter(subset["comment_ratio"], subset["call_depth"], label=label)
    plt.xlabel("注释率")
    plt.ylabel("最大嵌套深度")
    plt.title("结构 vs 注释可视化")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    code1 = '''
PI = 3.14
class my_class:
    def processData(self):
        pass
    def __init__(self, x, y):
        self.Value = x + y
        return self.Value

def longFunctionNameWithLotsOfArgs(a, b, c, d, e, f):
    try:
        return a + b + c + d + e + f
    except Exception as e:
        print(e)
'''

    code2 = '''
MAX_RETRIES = 5
class MyClass:
    def process_data(self):
        """Process some data."""
        pass
    def __init__(self, x, y):
        self.value = x + y
        return self.value

def fetch_data(url, retries=3):
    """Fetch data from the internet."""
    for _ in range(retries):
        try:
            response = request.get(url)
            return response.json()
        except:
            continue
    return None
'''

    df = export_code_metrics(code1, code2)
    visualize_metrics(df)