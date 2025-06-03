from datasets import load_from_disk, DatasetDict
from torch.utils.data import DataLoader, DistributedSampler
import torch
from extract.extract_full_style_vector import extract_full_code_style_vector

# 加载并拆分数据集
def get_dataset(dataset_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    dataset = load_from_disk(dataset_path)
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.01, seed=42)
        dataset = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

    def filter_fn(example):
        return len(tokenizer.encode(example["code"])) <= 378

    dataset["train"] = dataset["train"].filter(filter_fn, num_proc=2)
    dataset["validation"] = dataset["validation"].filter(filter_fn, num_proc=2).select(range(200))


    return dataset


# 构造 collate 函数
def collate_fn(batch):
    code1 = [b["code"] for b in batch]
    code2 = [b["python"] for b in batch]
    style1_vecs = [extract_full_code_style_vector(c) for c in code1]
    style2_vecs = [extract_full_code_style_vector(c) for c in code2]
    style1 = torch.stack(style1_vecs)
    style2 = torch.stack(style2_vecs)
    return code1, code2, style1, style2

# 获取 DataLoader，支持 DDP

def get_dataloaders(dataset, batch_size, is_ddp=False):
    if is_ddp:
        train_sampler = DistributedSampler(dataset["train"])
        train_loader = DataLoader(dataset["train"], batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)
    else:
        train_sampler = None
        train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_loader = DataLoader(dataset["validation"].select(range(min(100, len(dataset["validation"])))), batch_size=1, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, train_sampler