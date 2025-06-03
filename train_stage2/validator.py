import torch
from tqdm import tqdm
from Style_Eval import compute_css_score
from extract.extract_full_style_vector import extract_full_code_style_vector

def validate(model, val_loader, tokenizer, style_encoder, device):
    # ✅ 自动兼容 DDP（解包）
    if hasattr(model, "module"):
        model = model.module

    model.eval()
    css_scores_fwd, css_scores_bwd = [], []

    with torch.no_grad():
        for code1, code2, style1, style2 in tqdm(val_loader, desc="🔍 Validating"):
            style1, style2 = style1.to(device), style2.to(device)
            try:
                # ➤ Forward: code1 + style2 → code2
                input_ids = tokenizer(code1, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
                style2_encoded = style_encoder(style2)
                gen_ids = model.generate(input_ids, style2_encoded, max_length=256)
                gen_code2 = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                feat2 = extract_full_code_style_vector(gen_code2).unsqueeze(0).to(device)
                dist_fwd = compute_css_score(feat2, style2, method="euclidean")
                css_fwd = torch.exp(-torch.tensor(dist_fwd)).item()
                css_scores_fwd.append(css_fwd)

                # ➤ Backward: code2 + style1 → code1
                input_ids_rev = tokenizer(code2, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
                style1_encoded = style_encoder(style1)
                gen_ids_rev = model.generate(input_ids_rev, style1_encoded, max_length=256)
                gen_code1 = tokenizer.decode(gen_ids_rev[0], skip_special_tokens=True)
                feat1 = extract_full_code_style_vector(gen_code1).unsqueeze(0).to(device)
                dist_bwd = compute_css_score(feat1, style1, method="euclidean")
                css_bwd = torch.exp(-torch.tensor(dist_bwd)).item()
                css_scores_bwd.append(css_bwd)
                
            except RuntimeError as e:
                print(f"⚠️ Validation error: {e}")
                torch.cuda.empty_cache()
                continue

    # ➤ 计算平均 CSS 分数
    css_forward = sum(css_scores_fwd) / len(css_scores_fwd) if css_scores_fwd else 0.0
    css_backward = sum(css_scores_bwd) / len(css_scores_bwd) if css_scores_bwd else 0.0
    avg_css = (css_forward + css_backward) / 2


    return css_forward, css_backward, avg_css
