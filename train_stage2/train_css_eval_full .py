
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from model.style_encoder import StyleEncoder
from model.ContrastiveStyleTrainer import ContrastiveStyleTrainer
from style_eval import extract_style_features, compute_css_score

# === 配置参数 ===
batch_size = 16
num_epochs = 100
lr = 5e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# === 加载模型 ===
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
style_encoder = StyleEncoder(output_dim=base_model.config.hidden_size).to(device)
style_encoder.load_state_dict(torch.load("/root/autodl-tmp/code_perference/checkpoints/20250420_0321_style_encoder_epoch200.pt"))
trainer = ContrastiveStyleTrainer(base_model, style_encoder, device=device, use_lora=False)
optimizer = torch.optim.AdamW(trainer.style_encoder.parameters(), lr=lr)

# === 加载数据集 ===
dataset = load_from_disk("/root/code_perference/datasets/dataset_cleaned")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

def code_collate_fn(batch, alpha=1.0):
    code1_list = [item["code"] for item in batch]
    code2_list = [item["python"] for item in batch]
    style1 = torch.tensor([item["style1"] for item in batch], dtype=torch.float32)
    style2 = torch.tensor([item["style2"] for item in batch], dtype=torch.float32)
    style_interp = alpha * style2 + (1 - alpha) * style1
    return {"code1": code1_list, "code2": code2_list, "style1_interp": style_interp}

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=lambda b: code_collate_fn(b, alpha=0.7))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: code_collate_fn(b, alpha=0.7))

# === 验证集评估函数 ===
def evaluate_css_on_val(model, style_encoder, tokenizer, val_loader, device):
    model.eval()
    style_encoder.eval()
    css_scores = []
    for batch in tqdm(val_loader, desc="🧪 评估CSS"):
        code1_list = batch["code1"]
        style1 = batch["style1_interp"].to(device)
        for i in range(len(code1_list)):
            input_ids = tokenizer(code1_list[i], return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
            style_vec = style1[i].unsqueeze(0)
            with torch.no_grad():
                gen_ids = model.generate(input_ids=input_ids, max_length=256, do_sample=False)
                generated_code = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                style_feat = extract_style_features(generated_code).to(device)
                pred_style_vec = style_encoder(style_feat.unsqueeze(0))
                css = compute_css_score(pred_style_vec, style_vec)
                css_scores.append(css)
    return sum(css_scores) / len(css_scores)

# === 训练主循环 ===
loss_history = []
css_history = []
best_css = -1
patience = 5
patience_counter = 0
history_log = []

for epoch in range(num_epochs):
    trainer.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"🔥 Epoch {epoch+1}/{num_epochs}")
    for batch in loop:
        code1_input = tokenizer(batch["code1"], return_tensors="pt", padding=True, truncation=True).to(device)
        code2_input = tokenizer(batch["code2"], return_tensors="pt", padding=True, truncation=True).to(device)
        style_vec = batch["style1_interp"].to(device)

        loss = trainer(code1_input, code2_input, style_vec)
        if loss is None:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.style_encoder.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)

    # === CSS 评估 ===
    avg_css = evaluate_css_on_val(base_model, style_encoder, tokenizer, val_loader, device)
    css_history.append(avg_css)
    print(f"✅ Epoch {epoch+1} | Loss: {avg_loss:.4f} | CSS: {avg_css:.4f}")

    history_log.append({"epoch": epoch+1, "loss": avg_loss, "css": avg_css})

    # === 模型保存与早停控制 ===
    save_path = f"checkpoints/{timestamp}_style_encoder_epoch{epoch+1}.pt"
    torch.save(style_encoder.state_dict(), save_path)
    if avg_css > best_css:
        best_css = avg_css
        patience_counter = 0
        torch.save(style_encoder.state_dict(), f"checkpoints/{timestamp}_best_style_encoder.pt")
        print("💾 保存最佳 CSS 模型")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ 提前终止训练（CSS无提升）")
            break

# === 日志保存与可视化 ===
log_df = pd.DataFrame(history_log)
log_df.to_csv(f"logs/train_css_log_{timestamp}.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(log_df["epoch"], log_df["loss"], label="Loss", marker="o")
plt.plot(log_df["epoch"], log_df["css"], label="CSS", marker="x")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Training Loss vs CSS")
plt.legend()
plt.grid(True)
plt.savefig(f"logs/loss_css_curve_{timestamp}.png")
plt.show()
