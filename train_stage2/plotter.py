import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_loss_css(log, log_dir):
    log_df = pd.DataFrame(log)

    # Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(log_df["epoch"], log_df["total_loss"], marker="o", label="Total Loss")
    plt.plot(log_df["epoch"], log_df["ce_loss_1"], marker="x", label="CE Loss (code1→code2)")
    plt.plot(log_df["epoch"], log_df["ce_loss_2"], marker="x", label="CE Loss (code2→code1)")
    plt.plot(log_df["epoch"], log_df["style_loss_1"], marker="s", label="Style Loss (code1→code2)")
    plt.plot(log_df["epoch"], log_df["style_loss_2"], marker="s", label="Style Loss (code2→code1)")
    plt.title("Training Losses", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "loss_curve.png"), dpi=200)
    plt.close()

    # CSS 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(log_df["epoch"], log_df["css_forward"], marker="d", label="Forward CSS")
    plt.plot(log_df["epoch"], log_df["css_backward"], marker="d", label="Backward CSS")
    plt.plot(log_df["epoch"], log_df["css"], marker="d", label="Average CSS")
    plt.title("Validation CSS Scores", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("CSS Score", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "css_curve.png"), dpi=200)
    plt.close()

    # 保存日志表格
    log_df.to_csv(os.path.join(log_dir, "train_log.csv"), index=False)
