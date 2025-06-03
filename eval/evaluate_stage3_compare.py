import os
import time
import torch
import json
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model.style_encoder import StyleEncoder
from model.StyleControlledGenerator import StyleControlledGenerator
from Style_Eval import compute_raw_css_score
from extract_full_style_vector import extract_full_style_vector
from openai import OpenAI
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

# === Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
style_encoder = StyleEncoder(input_dim=25, output_dim=1024).to(device)
style_encoder.load_state_dict(torch.load("/root/autodl-tmp/code_perference/checkpoints/20250420_0321_style_encoder_epoch200.pt"))
style_encoder.eval()

local_model = StyleControlledGenerator().to(device)
local_model.load_state_dict(torch.load("/root/autodl-tmp/code_perference/checkpoints/stage3_20250423_1040_best.pt"))
local_model.eval()
flan_t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
flan_t5.eval()

# === Load dataset ===
dataset = load_from_disk("/root/autodl-tmp/code_perference/datasets/dataset_cleaned")
if "validation" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
val_dataset = dataset["validation"].select(range(100))

code_inputs = [ex["code"] for ex in val_dataset]
prompt_inputs = [ex["prompt"] for ex in val_dataset]
style1_list = [ex["style1"] for ex in val_dataset]
style2_list = [ex["style2"] for ex in val_dataset]
alpha = 1.0

# === Encode style vectors ===
target_styles = []
for s1, s2 in zip(style1_list, style2_list):
    raw_style = alpha * torch.tensor(s2) + (1 - alpha) * torch.tensor(s1)
    target_styles.append(raw_style)  # 直接使用25维向量


# === APIs ===
client_qwen = OpenAI(api_key="sk-883fe9faf7544bf29c13a3a7c976168b", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
def qwen(prompt):
    messages = [{"role": "user", "content": prompt}]
    completion = client_qwen.chat.completions.create(model="qwen-plus", messages=messages)
    return completion.choices[0].message.content

client_deepseek = OpenAI(api_key="sk-7f3d001e0952475dbf9e48d88395f7d3", base_url="https://api.deepseek.com")
def deepseek(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = client_deepseek.chat.completions.create(model="deepseek-chat", messages=messages, stream=False)
    return response.choices[0].message.content

# === CSS with error handling ===
def robust_css_score(code: str, target_vec, model_name, index, error_tracker, max_retries=3):
    for attempt in range(max_retries):
        try:
            pred_vec = extract_full_style_vector(code).to(device)  # 👈 25维原始风格特征
            return compute_raw_css_score(pred_vec, target_vec.to(device))  # 👈 不再使用 style_encoder
        except Exception as e:
            error_tracker[model_name]["count"] += 1
            error_tracker[model_name]["details"].append((index, str(e)))
            time.sleep(0.1)
    return 0.0


# === Generation functions ===
def generate_local(model, input_texts, style_vecs):
    outputs = []
    for i, (text, vec) in enumerate(tqdm(zip(input_texts, style_vecs), total=len(input_texts), desc="Generating MyModel")):
        input_ids = tokenizer(text, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
        style_vec = style_encoder(torch.tensor(vec).unsqueeze(0).to(device))
        gen_ids = model.generate(input_ids, style_vec, max_length=256)
        output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        try:
            output = qwen("Please fix the complete code according to the provided code (return only the fixed code to me): " + output)
        except:
            pass
        outputs.append(output)
    return outputs

def generate_flan_t5(texts):
    outputs = []
    for i, text in enumerate(tqdm(texts, desc="Generating Flan-T5")):
        input_ids = tokenizer(text, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
        gen_ids = flan_t5.generate(input_ids=input_ids, max_length=256)
        outputs.append(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
    return outputs

# === Error tracker ===
error_tracker = defaultdict(lambda: {"count": 0, "details": []})

# === Evaluation ===
print("🔍 [1] MyModel")
baseline_outputs = generate_local(local_model, code_inputs, style1_list)
baseline_css = [robust_css_score(g, t, "MyModel", i, error_tracker) for i, (g, t) in enumerate(zip(baseline_outputs, target_styles))]

print("🔍 [2] Flan-T5")
flan_outputs = generate_flan_t5(prompt_inputs)
flan_css = [robust_css_score(g, t, "Flan-T5", i, error_tracker) for i, (g, t) in enumerate(zip(flan_outputs, target_styles))]

print("🔍 [3] DeepSeek")
deepseek_outputs, deepseek_css = [], []
for i in tqdm(range(len(prompt_inputs)), desc="Generating DeepSeek"):
    try:
        output = deepseek(prompt_inputs[i])
    except:
        output = ""
    deepseek_outputs.append(output)
    deepseek_css.append(robust_css_score(output, target_styles[i], "DeepSeek", i, error_tracker))

print("🔍 [4] Qwen")
qwen_outputs, qwen_css = [], []
for i in tqdm(range(len(prompt_inputs)), desc="Generating Qwen"):
    try:
        output = qwen(prompt_inputs[i])
    except:
        output = ""
    qwen_outputs.append(output)
    qwen_css.append(robust_css_score(output, target_styles[i], "Qwen", i, error_tracker))

# === Save results ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df = pd.DataFrame({
    "Index": list(range(len(baseline_css))), 
    "MyModel": baseline_css,
    "Flan-T5": flan_css,
    "DeepSeek": deepseek_css,
    "Qwen": qwen_css
})
df.to_csv(f"logs/css_scores_detailed_{timestamp}.csv", index=False)

summary_df = df.drop(columns=["Index"]).describe().T.reset_index().rename(columns={"index": "Model"})
error_summary = {
    model: {
        "ErrorCount": info["count"],
        "ErrorRate": round(info["count"] / 50, 3)
    } for model, info in error_tracker.items()
}
error_df = pd.DataFrame(error_summary).T.reset_index().rename(columns={"index": "Model"})
final_df = pd.merge(summary_df, error_df, on="Model", how="left").fillna(0)
final_df.to_csv(f"logs/css_scores_summary_{timestamp}.csv", index=False)

# === Save generated code ===
output_map = {
    "MyModel": baseline_outputs,
    "Flan-T5": flan_outputs,
    "DeepSeek": deepseek_outputs,
    "Qwen": qwen_outputs
}
for name, result in output_map.items():
    with open(f"logs/generated_{name}_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump({str(i): result[i] for i in range(len(result))}, f, ensure_ascii=False, indent=2)
# === Save error logs ===
error_log_path = f"logs/error_log_{timestamp}.json"
with open(error_log_path, "w", encoding="utf-8") as f:
    json.dump({k: v for k, v in error_tracker.items()}, f, ensure_ascii=False, indent=2)
print(f"⚠️ Error log saved to {error_log_path}")

# === Visualization ===
plt.figure(figsize=(10, 6))
df.drop(columns=["Index"]).boxplot()
plt.title("CSS Score Distribution across Models")
plt.ylabel("CSS Score")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(rotation=15)
plt.savefig(f"logs/css_eval_compare_boxplot_{timestamp}.png")
plt.show()
print("📉 All results and visualizations saved to 'logs/' directory.")
