import torch
from extract.extract_full_style_vector import extract_full_code_style_vector  # ✅ 正确

def code_collate_fn(batch):
    code1_list = [item["code"] for item in batch]
    code2_list = [item["python"] for item in batch]
    all_code = code1_list + code2_list

    vec_seqs = [extract_full_code_style_vector(code) for code in all_code]
    stacked_vecs = torch.stack(vec_seqs)  # [2*B, 33] ✅

    return {
        "code_input": all_code,
        "style_vec": stacked_vecs  # ✅ 命名清晰
    }
