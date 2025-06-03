import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Config

class StyleT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        style_embedding=None,  # ✅ 新增：风格向量
        **kwargs
    ):
        # 获取 encoder 输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # ✅ 注入风格向量到 encoder hidden states
        if style_embedding is not None:
            # shape: [batch, hidden_size] → [batch, 1, hidden_size] → expand to [batch, seq_len, hidden_size]
            B, S, H = encoder_outputs.last_hidden_state.shape
            style_embedding = style_embedding.unsqueeze(1).expand(-1, S, -1)
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state + style_embedding

        # decoder 和损失计算
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )
