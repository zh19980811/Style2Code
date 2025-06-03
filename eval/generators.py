# === config.py ===
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model.style_encoder import StyleEncoder
from model.StyleControlledGenerator import StyleControlledGenerator
from datasets import load_from_disk, DatasetDict
from extract_full_style_vector import extract_full_code_style_vector 

def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    style_encoder = StyleEncoder(25, 1024).to(device)
    style_encoder.load_state_dict(torch.load("/root/autodl-tmp/code_perference/checkpoints_1/20250427_0820_style_encoder_epoch50.pt"))
    style_encoder.eval()

    gen_model = StyleControlledGenerator().to(device)
    gen_model.load_state_dict(torch.load("/root/autodl-tmp/code_perference/checkpoints/stage3_20250423_1040_best.pt"))
    gen_model.eval()

    flan_t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
    flan_t5.eval()

    return tokenizer, style_encoder, gen_model, flan_t5, device

def load_dataset():
    dataset = load_from_disk("/root/autodl-tmp/code_perference/datasets/dataset_cleaned")
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
    return dataset["validation"].select(range(300))

# === generators.py ===
from openai import OpenAI
from tqdm import tqdm
import torch

client_qwen = OpenAI(api_key="sk-883fe9faf7544bf29c13a3a7c976168b", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
client_deepseek = OpenAI(api_key="sk-7f3d001e0952475dbf9e48d88395f7d3", base_url="https://api.deepseek.com")

def qwen(prompt):
    messages = [{"role": "user", "content": prompt}]
    completion = client_qwen.chat.completions.create(model="qwen-plus", messages=messages)
    return completion.choices[0].message.content

def deepseek(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = client_deepseek.chat.completions.create(model="deepseek-chat", messages=messages, stream=False)
    return response.choices[0].message.content

def generate_local(model, tokenizer, input_texts, ref_texts, style_encoder, device):
    outputs = []
    for i, (input_text, ref_text) in enumerate(tqdm(zip(input_texts, ref_texts), total=len(input_texts), desc="Generating MyModel")):
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)

        with torch.no_grad():
            style_vec = extract_full_style_vector(ref_text).unsqueeze(0).to(device)
            encoded_vec = style_encoder(style_vec)

        gen_ids = model.generate(input_ids, encoded_vec, max_length=256)
        output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        outputs.append(output)

    return outputs


def generate_flan_t5(model, tokenizer, texts, device):
    outputs = []
    for text in tqdm(texts, desc="Generating Flan-T5"):
        input_ids = tokenizer(text, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
        gen_ids = model.generate(input_ids=input_ids, max_length=256)
        outputs.append(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
    return outputs
