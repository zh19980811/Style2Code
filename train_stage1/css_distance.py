import torch
import torch.nn.functional as F

def normalize_to_distribution(vec: torch.Tensor, eps=1e-8):
    """
    将向量归一化为概率分布（所有元素非负，和为1）
    """
    vec = torch.clamp(vec, min=0.0)  # 防止负值
    total = torch.sum(vec, dim=-1, keepdim=True) + eps
    return vec / total

def compute_raw_css_score(vec1: torch.Tensor, vec2: torch.Tensor, method="jsd") -> float:
    vec1 = vec1.float().squeeze()
    vec2 = vec2.float().squeeze()

    if method == "cosine":
        cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
        return 1 - cos_sim  # 越小越相似

    elif method == "jsd":
        # 如果向量为全 0，跳过 softmax，直接返回最大距离 1.0
        if torch.all(vec1 == 0) or torch.all(vec2 == 0):
            return 1.0

        # 为避免数值不稳定，加一个小的 epsilon，确保 softmax 不出 nan
        epsilon = 1e-8
        p = F.softmax(vec1, dim=-1) + epsilon
        q = F.softmax(vec2, dim=-1) + epsilon
        m = 0.5 * (p + q)

        # log(m) 可能有 0 的位置，加 epsilon 防止 log(0)
        jsd = 0.5 * (F.kl_div(torch.log(m), p, reduction='batchmean') +
                     F.kl_div(torch.log(m), q, reduction='batchmean'))
        return jsd.item()
    else:
        raise ValueError(f"Unsupported similarity method: {method}")

