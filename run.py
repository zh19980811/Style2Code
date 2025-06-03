import os
import torch
import datetime
from transformers import T5Tokenizer
from model.StyleControlledGenerator import StyleControlledGenerator
from model.style_encoder import StyleEncoder
from extract.extract_full_style_vector import extract_full_code_style_vector
from utils.css_distance import compute_raw_css_score

# === 模型权重路径 ===
# STYLE_ENCODER_PATH = "/root/autodl-tmp/code_perference/checkpoints_1/20250427_0820_style_encoder_epoch50.pt"
STYLE_ENCODER_PATH = "stage1"
# GENERATOR_PATH = "/root/autodl-tmp/code_perference/train_stage2/checkpoints_ddp/stage3_ddp_20250504_0241/epoch20.pt"
GENERATOR_PATH = "/root/autodl-tmp/code_perference/train_stage2/checkpoints_ddp/stage3_ddp_20250511_0703/best.pt"

# === 中性输入代码 ===
code_input = """
def total_area(w1, h1, w2, h2):
    return w1 * h1 + w2 * h2
"""

# === 风格参考 A（规范 + 注释） ===
ref_text_A = """
def calculate_total_area(rect1_width, rect1_height, rect2_width, rect2_height):
    '''计算两个矩形的面积之和'''
    #1
    area1 = rect1_width * rect1_height
    area2 = rect2_width * rect2_height
    return area1 + area2
"""

# === 风格参考 B（紧凑，无注释） ===
ref_text_B = """
def f(a, b, c, d): 

    return a*b + c*d
"""

def load_models(device):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    style_encoder = StyleEncoder().to(device)
    style_encoder.load_state_dict(torch.load(STYLE_ENCODER_PATH, map_location=device))
    style_encoder.eval()

    model = StyleControlledGenerator().to(device)

    # ✅ 自动识别模型保存格式（新/旧）
    checkpoint = torch.load(GENERATOR_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("✅ 加载新格式 checkpoint")
        state_dict = checkpoint["model_state_dict"]
    else:
        print("⚠️ 加载旧格式模型参数（仅含 state_dict）")
        state_dict = checkpoint

    if any(k.startswith("module.") for k in state_dict.keys()):
        print("🛠️ Detected DDP model, stripping 'module.' prefix")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    return tokenizer, style_encoder, model


def get_style_vec(code_text, style_encoder, device):
    raw_vec = extract_full_code_style_vector(code_text).unsqueeze(0).to(device)
    with torch.no_grad():
        encoded_vec = style_encoder(raw_vec)
    return raw_vec, encoded_vec

def generate_with_style(model, tokenizer, input_code, style_emb, device):
    input_ids = tokenizer(input_code, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, style_vec=style_emb, max_length=256)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def print_tensor_stats(name, vec, log_lines):
    log_lines.append(f"\n📐 {name} 向量统计信息:")
    log_lines.append(f"- 均值: {vec.mean().item():.6f}")
    log_lines.append(f"- 方差: {vec.var().item():.6f}")
    log_lines.append(f"- 向量值: {[round(x, 4) for x in vec.squeeze().cpu().tolist()]}")

def log_decomposed_style_vectors(vec_a, vec_b, log_lines):
    vec_a = vec_a.squeeze().cpu().tolist()
    vec_b = vec_b.squeeze().cpu().tolist()

    spacing_a = [round(v, 4) for v in vec_a[0:9]]
    naming_a = [round(v, 4) for v in vec_a[9:23]]
    structure_a = [round(v, 4) for v in vec_a[23:34]]
    spacing_b = [round(v, 4) for v in vec_b[0:9]]
    naming_b = [round(v, 4) for v in vec_b[9:23]]
    structure_b = [round(v, 4) for v in vec_b[23:34]]

    log_lines.append("\n🔍 分段向量拆解 (34维总向量 = spacing[0:9] + naming[9:23] + structure[23:34])")
    log_lines.append(f"Style A spacing:   {spacing_a}")
    log_lines.append(f"Style A naming:    {naming_a}")
    log_lines.append(f"Style A structure: {structure_a}")
    log_lines.append(f"Style B spacing:   {spacing_b}")
    log_lines.append(f"Style B naming:    {naming_b}")
    log_lines.append(f"Style B structure: {structure_b}")

def save_log(content: str, folder: str = "style_transfer_logs"):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"log_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n📝 日志已保存到: {filename}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, style_encoder, model = load_models(device)

    log_lines = []
    log_lines.append("\n🔍 原始风格向量差异 (34维拼接向量):")

    vec_A_raw, vec_A_encoded = get_style_vec(ref_text_A, style_encoder, device)
    vec_B_raw, vec_B_encoded = get_style_vec(ref_text_B, style_encoder, device)

    log_lines.append(f"CSS(raw): {compute_raw_css_score(vec_A_raw, vec_B_raw, method='euclidean'):.6f}")
    print_tensor_stats("Style A - Raw", vec_A_raw, log_lines)
    print_tensor_stats("Style B - Raw", vec_B_raw, log_lines)

    log_decomposed_style_vectors(vec_A_raw, vec_B_raw, log_lines)

    log_lines.append("\n🔍 编码后风格向量差异:")
    log_lines.append(f"CSS(encoded): {compute_raw_css_score(vec_A_encoded, vec_B_encoded, method='euclidean'):.6f}")
    print_tensor_stats("Style A - Encoded", vec_A_encoded, log_lines)
    print_tensor_stats("Style B - Encoded", vec_B_encoded, log_lines)

    gen_A = generate_with_style(model, tokenizer, code_input, vec_A_encoded, device)
    gen_B = generate_with_style(model, tokenizer, code_input, vec_B_encoded, device)

    log_lines.append("\n🔧 使用 style A 生成:")
    log_lines.append("gen_A:\n" + gen_A)
    log_lines.append("\n🔧 使用 style B 生成:")
    log_lines.append("gen_B:\n" + gen_B)

    vec_gen_A = extract_full_code_style_vector(gen_A).unsqueeze(0).to(device)
    vec_gen_B = extract_full_code_style_vector(gen_B).unsqueeze(0).to(device)

    log_lines.append("\n🎯 生成代码 CSS(gen_A vs gen_B):")
    log_lines.append(f"CSS(gen_A vs gen_B): {compute_raw_css_score(vec_gen_A, vec_gen_B, method='euclidean'):.6f}")
    print_tensor_stats("gen_A 风格向量", vec_gen_A, log_lines)
    print_tensor_stats("gen_B 风格向量", vec_gen_B, log_lines)

    save_log("\n".join(log_lines))
