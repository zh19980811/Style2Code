import torch
import torch.nn.functional as F

def compute_css_score(pred_vec: torch.Tensor, target_vec: torch.Tensor):
    eps = 1e-8
    p = F.softmax(pred_vec, dim=-1) + eps
    q = F.softmax(target_vec, dim=-1) + eps
    m = 0.5 * (p + q)

    kl_pm = F.kl_div(p.log(), m, reduction='mean')  # ✅ reduction 改成 mean
    kl_qm = F.kl_div(q.log(), m, reduction='mean')
    jsd = 0.5 * (kl_pm + kl_qm)

    jsd = torch.clamp(jsd, min=0.0).item()  # ✅ 标准安全写法
    return 1.0 - jsd
