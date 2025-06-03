import torch
from transformers import AutoTokenizer
from model.style_encoder import StyleEncoder
from model.StyleControlledGenerator import StyleControlledGenerator
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def load_models(style_encoder_path, lr, device, use_ddp=False, resume_from=None):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    # 加载 StyleEncoder
    style_encoder = StyleEncoder(input_dim=34, output_dim=1024).to(device)
    style_encoder.load_state_dict(torch.load(style_encoder_path, map_location=device))
    style_encoder.eval()

    # 加载主模型
    model = StyleControlledGenerator().to(device)

    # 如果 resume_from 存在，尝试加载 model 和 optimizer 参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if resume_from and os.path.exists(resume_from):
        print(f"🔁 Loading model + optimizer from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if use_ddp:
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    return tokenizer, style_encoder, model, optimizer
