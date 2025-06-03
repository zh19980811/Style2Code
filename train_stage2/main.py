from config import *
from dataloader import get_dataset, get_dataloaders
from model_loader import load_models
from trainer import train
from config import setup_ddp, cleanup_ddp, resume_from  # ✅ 导入 resume_from
import os

def main():
    # 初始化分布式环境
    device, local_rank, is_main_process = setup_ddp()

    if is_main_process:
        os.makedirs(paths["save_dir"], exist_ok=True)
        os.makedirs(paths["log_dir"], exist_ok=True)

    # 加载数据
    dataset = get_dataset(paths["dataset"])
    train_loader, val_loader, train_sampler = get_dataloaders(dataset, batch_size, is_ddp=True)

    # 加载模型和优化器
    tokenizer, style_encoder, model, optimizer = load_models(paths["style_encoder"], lr, device, use_ddp=True)

    # 启动训练（✅ 传入 resume_from 参数）
    train(
        model=model,
        tokenizer=tokenizer,
        style_encoder=style_encoder,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        device=device,
        paths=paths,
        num_epochs=num_epochs,
        lambda_style=lambda_style,
        warmup_epochs=warmup_epochs,
        is_main=is_main_process,
        resume_from=resume_from   # ✅ 加入这行
    )

    # 清理分布式环境
    cleanup_ddp()

if __name__ == "__main__":
    main()
