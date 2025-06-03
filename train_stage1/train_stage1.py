import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model.ContrastiveStyleTrainer import ContrastiveStyleTrainer
from model.style_encoder import StyleEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from datasets import load_from_disk
from sklearn.manifold import TSNE
from extract.extract_full_style_vector import extract_full_code_style_vector
from torch.nn.utils.rnn import pad_sequence
from model.collate_fn import code_collate_fn

# ✅ 超参数
batch_size = 16
num_epochs = 30
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 加载模型
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
style_encoder = StyleEncoder().to(device)
trainer = ContrastiveStyleTrainer(model, style_encoder, device=device, use_lora=False, temperature=0.07)
optimizer = torch.optim.AdamW(trainer.style_encoder.parameters(), lr=lr)

# ✅ 加载数据集
dataset = load_from_disk("/root/autodl-tmp/code_perference/datasets/dataset_cleaned")
train_dataset = dataset["train"]

# ✅ collate_fn



def code_collate_fn(batch):
    code1_list = [item["code"] for item in batch]
    code2_list = [item["python"] for item in batch]
    all_code = code1_list + code2_list

    vec_seqs = [extract_full_code_style_vector(code) for code in all_code]
    stacked_vecs = torch.stack(vec_seqs)  # [2*B, 33]

    return {
        "code_input": all_code,
        "style_vec": stacked_vecs   # ✅ 直接返回，名字也建议改清楚
    }



# ✅ t-SNE 可视化
def visualize_style_space(style_encoder, dataset, save_path="style_space_tsne.png", num_samples=100):
    style_encoder.eval()
    encoded_vecs, labels = [], []
    samples = dataset.select(range(min(len(dataset), num_samples)))

    with torch.no_grad():
        for item in samples:
            for tag, code in [("style1", item["code"]), ("style2", item["python"])]:
                vec_seq = extract_full_code_style_vector(code)
                input_tensor = vec_seq.unsqueeze(0).to(device)  # [1, 33]
                embed = style_encoder(input_tensor)     # [1, D]
                encoded_vecs.append(embed.squeeze(0).cpu())
                labels.append(tag)
                
    reduced = TSNE(n_components=2, random_state=42).fit_transform(torch.stack(encoded_vecs))
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="Set2")
    plt.title("StyleEncoder Output Space (t-SNE)")
    plt.savefig(save_path)
    plt.close()


# ✅ DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=code_collate_fn
)

# ✅ 创建保存目录
os.makedirs("checkpoints_1", exist_ok=True)
loss_history = []
loss_history_1 = []
loss_history_2 = []
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# ✅ 训练循环
for epoch in range(num_epochs):
    trainer.train()
    total_loss = 0
    total_loss_1 = 0
    total_loss_2 = 0
    loop = tqdm(train_loader, desc=f"🔥 Epoch {epoch+1}/{num_epochs}")

    for batch in loop:
        code_input = tokenizer(batch["code_input"], return_tensors="pt", padding=True, truncation=True)
        code_input = {k: v.to(device) for k, v in code_input.items()}
        style_vec = batch["style_vec"].to(device)  # ✅ 直接拿 style_vec
        
        B = style_vec.shape[0] // 2
        code1_input = {k: v[:B] for k, v in code_input.items()}
        code2_input = {k: v[B:] for k, v in code_input.items()}
        style1_vec = style_vec[:B]
        style2_vec = style_vec[B:]
        
        loss_1 = trainer(code1_input, code1_input, style1_vec)  # ✅ trainer只传style_vec
        loss_2 = trainer(code2_input, code2_input, style2_vec)


        if loss_1 is None or torch.isnan(loss_1) or loss_2 is None or torch.isnan(loss_2):
            continue

        loss = 0.5 * (loss_1 + loss_2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.style_encoder.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_2 += loss_2.item()
        loop.set_postfix(loss=loss.item())

    # ✅ 记录 loss
    avg_loss = total_loss / len(train_loader)
    avg_loss_1 = total_loss_1 / len(train_loader)
    avg_loss_2 = total_loss_2 / len(train_loader)
    loss_history.append(avg_loss)
    loss_history_1.append(avg_loss_1)
    loss_history_2.append(avg_loss_2)
    print(f"✅ Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | code1 Loss: {avg_loss_1:.4f} | code2 Loss: {avg_loss_2:.4f}")

    # ✅ 每 50 轮保存
    if (epoch + 1) % 50 == 0 or (epoch + 1) == num_epochs:
        save_model = f"checkpoints_1/{timestamp}_style_encoder_epoch{epoch+1}.pt"
        torch.save(trainer.style_encoder.state_dict(), save_model)
        print(f"💾 模型保存到: {save_model}")

        # ✅ 保存 Loss 曲线
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epoch + 2), loss_history_1, label="Loss: code1 + style1 → code1")
        plt.plot(range(1, epoch + 2), loss_history_2, label="Loss: code2 + style2 → code2")
        plt.plot(range(1, epoch + 2), loss_history,  label="Avg Loss", linestyle="--", color="black")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Dual-View Contrastive Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"checkpoints_1/{timestamp}_loss_epoch{epoch+1}.png")
        plt.close()

        # ✅ 保存风格空间图
        tsne_path = f"checkpoints_1/{timestamp}_tsne_epoch{epoch+1}.png"
        visualize_style_space(style_encoder, train_dataset, save_path=tsne_path)
