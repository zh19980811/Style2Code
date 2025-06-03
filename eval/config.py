# config.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model.style_encoder import StyleEncoder
from model.StyleControlledGenerator import StyleControlledGenerator
from datasets import load_from_disk, DatasetDict

def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    style_encoder = StyleEncoder(25, 1024).to(device)
    style_encoder.load_state_dict(torch.load("...style_encoder.pt"))
    style_encoder.eval()
    gen_model = StyleControlledGenerator().to(device)
    gen_model.load_state_dict(torch.load("...stage3_best.pt"))
    gen_model.eval()
    flan_t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
    flan_t5.eval()
    return tokenizer, style_encoder, gen_model, flan_t5, device

def load_dataset():
    dataset = load_from_disk(".../dataset_cleaned")
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
    return dataset["validation"].select(range(100))
