import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from extract.extract_full_style_vector import extract_full_code_style_vector
from validator import validate
from plotter import plot_loss_css

def train(model, tokenizer, style_encoder, optimizer, train_loader, val_loader, train_sampler,
          device, paths, num_epochs, lambda_style, warmup_epochs, is_main, resume_from=None):
    os.makedirs(paths["save_dir"], exist_ok=True)
    os.makedirs(paths["log_dir"], exist_ok=True)

    best_css = -1
    css_history = []
    log = []
    start_epoch = 0

    # ✅ 加载 checkpoint（兼容旧模型）
    if resume_from and os.path.exists(resume_from):
        print(f"🔁 Resuming training from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            print("✅ Loaded full checkpoint with optimizer and epoch.")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            best_css = checkpoint.get("best_css", -1)
        else:
            print("⚠️ Old checkpoint detected: only model state_dict.")
            model.load_state_dict(checkpoint)
            # optimizer 和 epoch 无法恢复

    for epoch in range(start_epoch, num_epochs):
        model.train()
        ce_loss_1_sum, ce_loss_2_sum = 0, 0
        style_loss_1_sum, style_loss_2_sum = 0, 0
        current_lambda_style = 0.0 if epoch < warmup_epochs else lambda_style

        for code1, code2, style1, style2 in tqdm(train_loader, desc=f"🔥 Epoch {epoch+1}"):
            style1, style2 = style1.to(device), style2.to(device)

            try:
                # code1 + style2 → code2
                inputs = tokenizer(code1, padding=True, truncation=True, max_length=378, return_tensors="pt").to(device)
                targets = tokenizer(code2, padding=True, truncation=True, max_length=378, return_tensors="pt").input_ids.to(device)
                style2_encoded = style_encoder(style2)
                loss_ce_1, logits_1 = model(inputs["input_ids"], inputs["attention_mask"], style2_encoded, labels=targets)
                preds_1 = torch.argmax(logits_1, dim=-1)
                decoded_1 = tokenizer.batch_decode(preds_1, skip_special_tokens=True)
                feat_1 = torch.stack([extract_full_code_style_vector(txt) for txt in decoded_1]).to(device)
                style_loss_1 = F.mse_loss(feat_1, style2)

                # code2 + style1 → code1
                inputs_rev = tokenizer(code2, padding=True, truncation=True, max_length=378, return_tensors="pt").to(device)
                targets_rev = tokenizer(code1, padding=True, truncation=True, max_length=378, return_tensors="pt").input_ids.to(device)
                style1_encoded = style_encoder(style1)
                loss_ce_2, logits_2 = model(inputs_rev["input_ids"], inputs_rev["attention_mask"], style1_encoded, labels=targets_rev)
                preds_2 = torch.argmax(logits_2, dim=-1)
                decoded_2 = tokenizer.batch_decode(preds_2, skip_special_tokens=True)
                feat_2 = torch.stack([extract_full_code_style_vector(txt) for txt in decoded_2]).to(device)
                style_loss_2 = F.mse_loss(feat_2, style1)

                loss = (loss_ce_1 + current_lambda_style * style_loss_1 + loss_ce_2 + current_lambda_style * style_loss_2) / 2

                if not torch.isfinite(loss):
                    print(f"⚠️ Skipping non-finite loss at epoch {epoch+1}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                ce_loss_1_sum += loss_ce_1.item()
                ce_loss_2_sum += loss_ce_2.item()
                style_loss_1_sum += style_loss_1.item()
                style_loss_2_sum += style_loss_2.item()

            except RuntimeError as e:
                print(f"⚠️ Training error: {e}")
                torch.cuda.empty_cache()
                continue

        avg_ce1 = ce_loss_1_sum / len(train_loader)
        avg_ce2 = ce_loss_2_sum / len(train_loader)
        avg_style1 = style_loss_1_sum / len(train_loader)
        avg_style2 = style_loss_2_sum / len(train_loader)
        avg_total = (
            avg_ce1 + current_lambda_style * avg_style1 +
            avg_ce2 + current_lambda_style * avg_style2
        ) / 2

        if (epoch + 1) % 2 == 0:
            css_fwd, css_bwd, css = validate(model, val_loader, tokenizer, style_encoder, device)
        else:
            css_fwd, css_bwd, css = css_history[-1] if css_history else (0.0, 0.0, 0.0)

        css_history.append((css_fwd, css_bwd, css))

        log.append({
            "epoch": epoch+1,
            "total_loss": avg_total,
            "ce_loss_1": avg_ce1,
            "ce_loss_2": avg_ce2,
            "style_loss_1": avg_style1,
            "style_loss_2": avg_style2,
            "css_forward": css_fwd,
            "css_backward": css_bwd,
            "css": css,
        })

        print(f"✅ Epoch {epoch+1} | Total Loss: {avg_total:.4f} | CSS: {css:.4f}")

        # ✅ 保存最佳模型（新版格式）
        if css > best_css:
            best_css = css
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_css': best_css
            }, os.path.join(paths["save_dir"], "best.pt"))
            print("📂 Saved best model")

        # ✅ 每2轮保存一次 checkpoint（新版格式）
        if (epoch + 1) % 2 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_css': best_css
            }, os.path.join(paths["save_dir"], f"epoch{epoch+1}.pt"))

        plot_loss_css(log, paths["log_dir"])
