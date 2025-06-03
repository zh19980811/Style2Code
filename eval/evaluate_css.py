# === evaluate_style_diff.py ===
from extract_full_style_vector import extract_full_style_vector
from Style_Eval import compute_raw_css_score
from datasets import load_from_disk
import torch
from tqdm import tqdm
import pandas as pd
import os

def evaluate_style_diff(dataset_path, save_path="logs/style_diff_scores.csv", n=100):
    dataset = load_from_disk(dataset_path)
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset = dataset["validation"].select(range(n))

    code1_list = [ex["code1"] for ex in dataset]
    code2_list = [ex["code2"] for ex in dataset]

    css_scores = []
    for idx, (code1, code2) in enumerate(tqdm(zip(code1_list, code2_list), total=len(code1_list), desc="Computing CSS (code1 vs code2)")):
        try:
            vec1 = extract_full_style_vector(code1)
            vec2 = extract_full_style_vector(code2)
            score = compute_raw_css_score(vec1, vec2)
        except Exception as e:
            print(f"[Error] Index {idx}: {e}")
            score = 0.0
        css_scores.append(score)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame({"Index": list(range(n)), "CSS_Distance": css_scores})
    df.to_csv(save_path, index=False)
    print(f"✅ Saved style difference scores to {save_path}")

if __name__ == "__main__":
    evaluate_style_diff("/root/autodl-tmp/code_perference/datasets/dataset_cleaned", n=100)
