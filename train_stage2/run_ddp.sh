#!/bin/bash

# 使用你机器上的 GPU 数量
NUM_GPUS=3

# 设置 Python 文件路径
TRAIN_SCRIPT="main.py"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 分布式训练启动（适用于 PyTorch >=1.9 的 torchrun）
torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=29506 \
  $TRAIN_SCRIPT
