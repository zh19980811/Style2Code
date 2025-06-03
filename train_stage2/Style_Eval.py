import torch
import torch.nn.functional as F

def normalize_to_distribution(vec: torch.Tensor, eps=1e-8):
    """
    将向量归一化为概率分布（所有元素非负，和为1）
    """
    vec = torch.clamp(vec, min=0.0)  # 防止负值
    total = torch.sum(vec, dim=-1, keepdim=True) + eps
    return vec / total

def compute_css_score(vec1: torch.Tensor, vec2: torch.Tensor, method="euclidean") -> float:
    """
    计算两个风格向量之间的差异
    支持方法:
      - "cosine": 余弦距离 (1 - similarity)
      - "euclidean": 欧几里得距离
      - "jsd": Jensen-Shannon 散度
    返回值越小，表示风格越相近
    """
    vec1 = vec1.float().squeeze()
    vec2 = vec2.float().squeeze()

    if method == "cosine":
        if torch.all(vec1 == 0) or torch.all(vec2 == 0):
            return 1.0
        cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
        return 1 - cos_sim

    elif method == "euclidean":
        return torch.norm(vec1 - vec2, p=2).item()

    elif method == "jsd":
        if torch.all(vec1 == 0) or torch.all(vec2 == 0):
            return 1.0
        epsilon = 1e-8
        p = F.softmax(vec1, dim=-1) + epsilon
        q = F.softmax(vec2, dim=-1) + epsilon
        m = 0.5 * (p + q)
        jsd = 0.5 * (F.kl_div(torch.log(m), p, reduction='batchmean') +
                     F.kl_div(torch.log(m), q, reduction='batchmean'))
        return jsd.item()

    else:
        raise ValueError(f"Unsupported similarity method: {method}")
