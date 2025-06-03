import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class StyleControlledGenerator(nn.Module):
    def __init__(self, base_model_name="google/flan-t5-large", style_dim=1024):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(base_model_name)
        self.style_proj = nn.Linear(style_dim, self.model.config.d_model)

    def forward(self, input_ids, attention_mask, style_vec, labels=None):
        # 拼接 style embedding
        style_emb = self.style_proj(style_vec).unsqueeze(1)  # [B, 1, D]
        input_embeds = self.model.encoder.embed_tokens(input_ids)  # [B, T, D]
        input_embeds = torch.cat([style_emb, input_embeds], dim=1)

        extended_attention = torch.cat(
            [torch.ones((attention_mask.size(0), 1), device=attention_mask.device), attention_mask], dim=1
        )

        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=extended_attention,
            labels=labels,
            return_dict=True
        )
        return outputs.loss, outputs.logits

    def generate(self, input_ids, style_vec, max_length=256, num_beams=1):
        # 拼接 style embedding
        style_emb = self.style_proj(style_vec).unsqueeze(1)  # [B, 1, D]
        input_embeds = self.model.encoder.embed_tokens(input_ids)  # [B, T, D]
        input_embeds = torch.cat([style_emb, input_embeds], dim=1)

        # 正确动态生成 attention_mask
        pad_token_id = self.model.config.pad_token_id
        attention_mask = (input_ids != pad_token_id).long()
        attention_mask = torch.cat(
            [torch.ones((input_ids.size(0), 1), device=input_ids.device), attention_mask], dim=1
        )

        return self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=False
        )
