# === runner.py ===
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import torch
from config import load_models, load_dataset
from generators import generate_local, generate_flan_t5, qwen, deepseek
from evaluate_css import robust_css_score
from evaluate_style_diff import compute_style_diff_scores
from evaluate_text_metrics import evaluate_text_metrics
from ai_judge import ai_judge_score
from save_results_markdown import save_outputs_as_markdown

def run_evaluation():
    tokenizer, style_encoder, gen_model, flan_t5, device = load_models()
    val_dataset = load_dataset()

    code_inputs = [ex["code"] for ex in val_dataset]
    prompt_inputs = [ex["prompt"] for ex in val_dataset]
    ref_outputs = [ex["code2"] for ex in val_dataset]
    style1_list = [ex["style1"] for ex in val_dataset]
    style2_list = [ex["style2"] for ex in val_dataset]

    error_tracker = defaultdict(lambda: {"count": 0, "details": []})

    print("🔍 [1] MyModel")
    baseline_outputs = generate_local(gen_model, tokenizer, code_inputs, ref_outputs, style_encoder, device)
    baseline_css = [robust_css_score(g, torch.tensor(s2), "MyModel", i, error_tracker, device) for i, (g, s2) in enumerate(zip(baseline_outputs, style2_list))]
    baseline_text_scores = evaluate_text_metrics(ref_outputs, baseline_outputs)
    baseline_ai_scores = ai_judge_score(ref_outputs, baseline_outputs)

    print("🔍 [2] Flan-T5")
    flan_outputs = generate_flan_t5(flan_t5, tokenizer, prompt_inputs, device)
    flan_css = [robust_css_score(g, torch.tensor(s2), "Flan-T5", i, error_tracker, device) for i, (g, s2) in enumerate(zip(flan_outputs, style2_list))]
    flan_text_scores = evaluate_text_metrics(ref_outputs, flan_outputs)
    flan_ai_scores = ai_judge_score(ref_outputs, flan_outputs)

    print("🔍 [3] DeepSeek")
    deepseek_outputs = [deepseek(prompt) for prompt in prompt_inputs]
    deepseek_css = [robust_css_score(g, torch.tensor(s2), "DeepSeek", i, error_tracker, device) for i, (g, s2) in enumerate(zip(deepseek_outputs, style2_list))]
    deepseek_text_scores = evaluate_text_metrics(ref_outputs, deepseek_outputs)
    deepseek_ai_scores = ai_judge_score(ref_outputs, deepseek_outputs)

    print("🔍 [4] Qwen")
    qwen_outputs = [qwen(prompt) for prompt in prompt_inputs]
    qwen_css = [robust_css_score(g, torch.tensor(s2), "Qwen", i, error_tracker, device) for i, (g, s2) in enumerate(zip(qwen_outputs, style2_list))]
    qwen_text_scores = evaluate_text_metrics(ref_outputs, qwen_outputs)
    qwen_ai_scores = ai_judge_score(ref_outputs, qwen_outputs)

    print("🔍 [5] Flan-T5(code2→style1)")
    flan_ref_prompts = [f"Please rewrite the following code into another version with the target style:\n\n{ref}\n\nOnly output the rewritten code." for ref in ref_outputs]
    flan_from_code2_outputs = generate_flan_t5(flan_t5, tokenizer, flan_ref_prompts, device)
    flan_from_code2_css = [robust_css_score(g, torch.tensor(s1), "Flan-T5(code2→style1)", i, error_tracker, device) for i, (g, s1) in enumerate(zip(flan_from_code2_outputs, style1_list))]
    flan_from_code2_text_scores = evaluate_text_metrics(code_inputs, flan_from_code2_outputs)
    flan_from_code2_ai_scores = ai_judge_score(code_inputs, flan_from_code2_outputs)

    print("🔍 [6] DeepSeek(code2→style1)")
    deepseek_ref_prompts = [f"Please rewrite the following code into another version with the target style:\n\n{ref}\n\nOnly output the rewritten code." for ref in ref_outputs]
    deepseek_from_code2_outputs = [deepseek(p) for p in deepseek_ref_prompts]
    deepseek_from_code2_css = [robust_css_score(g, torch.tensor(s1), "DeepSeek(code2→style1)", i, error_tracker, device) for i, (g, s1) in enumerate(zip(deepseek_from_code2_outputs, style1_list))]
    deepseek_from_code2_text_scores = evaluate_text_metrics(code_inputs, deepseek_from_code2_outputs)
    deepseek_from_code2_ai_scores = ai_judge_score(code_inputs, deepseek_from_code2_outputs)

    print("🔍 [7] Qwen(code2→style1)")
    qwen_ref_prompts = [f"Please rewrite the following code into another version with the target style:\n\n{ref}\n\nOnly output the rewritten code." for ref in ref_outputs]
    qwen_from_code2_outputs = [qwen(p) for p in qwen_ref_prompts]
    qwen_from_code2_css = [robust_css_score(g, torch.tensor(s1), "Qwen(code2→style1)", i, error_tracker, device) for i, (g, s1) in enumerate(zip(qwen_from_code2_outputs, style1_list))]
    qwen_from_code2_text_scores = evaluate_text_metrics(code_inputs, qwen_from_code2_outputs)
    qwen_from_code2_ai_scores = ai_judge_score(code_inputs, qwen_from_code2_outputs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame({
        "Index": list(range(len(baseline_css))),
        "MyModel": baseline_css,
        "Flan-T5": flan_css,
        "DeepSeek": deepseek_css,
        "Qwen": qwen_css,
        "Flan-T5(code2→style1)": flan_from_code2_css,
        "DeepSeek(code2→style1)": deepseek_from_code2_css,
        "Qwen(code2→style1)": qwen_from_code2_css
    })
    df.to_csv(f"logs/css_scores_detailed_{timestamp}.csv", index=False)

    summary_df = df.drop(columns=["Index"]).describe().T.reset_index().rename(columns={"index": "Model"})
    error_summary = {
        model: {
            "ErrorCount": info["count"],
            "ErrorRate": round(info["count"] / len(df), 3)
        } for model, info in error_tracker.items()
    }
    error_df = pd.DataFrame(error_summary).T.reset_index().rename(columns={"index": "Model"})
    final_df = pd.merge(summary_df, error_df, on="Model", how="left").fillna(0)
    final_df.to_csv(f"logs/css_scores_summary_{timestamp}.csv", index=False)

    all_text_scores = {
        "MyModel": baseline_text_scores,
        "Flan-T5": flan_text_scores,
        "DeepSeek": deepseek_text_scores,
        "Qwen": qwen_text_scores,
        "Flan-T5(code2→style1)": flan_from_code2_text_scores,
        "DeepSeek(code2→style1)": deepseek_from_code2_text_scores,
        "Qwen(code2→style1)": qwen_from_code2_text_scores
    }
    with open(f"logs/text_metrics_summary_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(all_text_scores, f, indent=2, ensure_ascii=False)

    all_ai_scores = {
        "MyModel": baseline_ai_scores,
        "Flan-T5": flan_ai_scores,
        "DeepSeek": deepseek_ai_scores,
        "Qwen": qwen_ai_scores,
        "Flan-T5(code2→style1)": flan_from_code2_ai_scores,
        "DeepSeek(code2→style1)": deepseek_from_code2_ai_scores,
        "Qwen(code2→style1)": qwen_from_code2_ai_scores
    }
    with open(f"logs/ai_judge_summary_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(all_ai_scores, f, indent=2, ensure_ascii=False)

    output_map = {
        "MyModel": baseline_outputs,
        "Flan-T5": flan_outputs,
        "DeepSeek": deepseek_outputs,
        "Qwen": qwen_outputs,
        "Flan-T5(code2→style1)": flan_from_code2_outputs,
        "DeepSeek(code2→style1)": deepseek_from_code2_outputs,
        "Qwen(code2→style1)": qwen_from_code2_outputs
    }
    for name, result in output_map.items():
        with open(f"logs/generated_{name}_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump({str(i): result[i] for i in range(len(result))}, f, ensure_ascii=False, indent=2)

    save_outputs_as_markdown(prompt_inputs, ref_outputs, output_map, save_dir="logs")

    error_log_path = f"logs/error_log_{timestamp}.json"
    with open(error_log_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in error_tracker.items()}, f, ensure_ascii=False, indent=2)
    print(f"⚠️ Error log saved to {error_log_path}")

    plt.figure(figsize=(10, 6))
    df.drop(columns=["Index"]).boxplot()
    plt.title("CSS Score Distribution across Models")
    plt.ylabel("CSS Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=15)
    plt.savefig(f"logs/css_eval_compare_boxplot_{timestamp}.png")
    plt.show()
    print("📉 All results and visualizations saved to 'logs/' directory.")

if __name__ == "__main__":
    run_evaluation()