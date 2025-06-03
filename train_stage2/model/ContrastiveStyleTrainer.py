import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType

class ContrastiveStyleTrainer(nn.Module):
    def __init__(self, base_model, style_encoder, device="cuda", use_lora=True, temperature=0.07):
        super().__init__()
        self.base_model = base_model
        self.device = device
        self.temperature = temperature

        # 冻结 base_model（T5）
        for p in self.base_model.parameters():
            p.requires_grad = False

        # 加 LoRA（可选）
        if use_lora:
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=8, lora_alpha=32, lora_dropout=0.1,
                bias="none"
            )
            self.style_encoder = get_peft_model(style_encoder, lora_cfg)
        else:
            self.style_encoder = style_encoder

    def forward(self, code1_input, code2_input, style_vec):
        """
        Args:
            code1_input: dict of input_ids, attention_mask
            code2_input: dict of input_ids, attention_mask
            style_vec: FloatTensor [B, 33]，风格特征
        """
        # 编码 code1
        code1_embed = self.base_model.encoder(
            input_ids=code1_input["input_ids"],
            attention_mask=code1_input["attention_mask"]
        ).last_hidden_state[:, 0, :]  # [B, D]

        # 编码 code2
        code2_embed = self.base_model.encoder(
            input_ids=code2_input["input_ids"],
            attention_mask=code2_input["attention_mask"]
        ).last_hidden_state[:, 0, :]  # [B, D]

        # 编码风格向量（MLP处理）
        style_embed = self.style_encoder(style_vec)  # ✅ 注意这里直接输入 style_vec，不要 lengths了！

        # 融合 style 与 code1 → 得到 anchor 表示
        fusion_embed = code1_embed + style_embed  # [B, D]

        # 标准化向量（用于余弦对比）
        fusion_embed = F.normalize(fusion_embed, dim=-1)
        code2_embed = F.normalize(code2_embed, dim=-1)

        # 对比损失：NT-Xent / InfoNCE
        logits = torch.matmul(fusion_embed, code2_embed.T)  # [B, B]
        labels = torch.arange(logits.size(0)).to(self.device)  # [0, 1, ..., B-1]

        loss = F.cross_entropy(logits / self.temperature, labels)

        return loss
