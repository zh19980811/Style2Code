# config.py

import torch
import pandas as pd
import os
import torch.distributed as dist

# 分布式初始化
def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), local_rank, dist.get_rank() == 0

def cleanup_ddp():
    dist.destroy_process_group()

# 训练配置
batch_size = 1
num_epochs = 20
lr = 1e-5
lambda_style = 1.0
warmup_epochs = 5
timestamp = "stage3_ddp_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M")

paths = {
    "dataset": "/root/autodl-tmp/code_perference/datasets/dataset_cleaned",
    "style_encoder": "/root/autodl-tmp/code_perference/checkpoints_1/20250427_0820_style_encoder_epoch50.pt",
    "save_dir": f"checkpoints_ddp/{timestamp}",
    "log_dir": f"logs/{timestamp}"
}

# ✅ resume_from 是独立变量，不在 paths 字典中
resume_from = "checkpoints_ddp/stage3_ddp_20250509_0723/epoch10.pt"
