import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer

def analyze_lengths(dataset_path, max_len=512):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    dataset = load_from_disk(dataset_path)

    for split in ["train", "validation"]:
        if split not in dataset:
            continue

        data = dataset[split]
        print(f"Split: {split}, Total examples: {len(data)}")

        code_char_lens = []
        code_token_lens = []
        py_char_lens = []
        py_token_lens = []

        for example in data:
            code = example["code"]
            py = example["python"]

            code_char_lens.append(len(code))
            py_char_lens.append(len(py))

            code_token_lens.append(len(tokenizer.encode(code)))
            py_token_lens.append(len(tokenizer.encode(py)))

        def plot_hist(data, title, xlabel, filename):
            plt.figure(figsize=(8, 4))
            plt.hist(data, bins=50, color="skyblue", edgecolor="black")
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        plot_hist(code_char_lens, f"{split.capitalize()} Code Char Length", "Characters", f"{split}_code_char_len.png")
        plot_hist(py_char_lens, f"{split.capitalize()} Python Char Length", "Characters", f"{split}_py_char_len.png")
        plot_hist(code_token_lens, f"{split.capitalize()} Code Token Length", "Tokens", f"{split}_code_token_len.png")
        plot_hist(py_token_lens, f"{split.capitalize()} Python Token Length", "Tokens", f"{split}_py_token_len.png")

        print(f"\n=== {split.upper()} Length Stats ===")
        for name, arr in [("Code Tokens", code_token_lens), ("Python Tokens", py_token_lens)]:
            print(f"{name}: max={max(arr)}, min={min(arr)}, avg={sum(arr)//len(arr)}, >{max_len}={sum(x > max_len for x in arr)}")

if __name__ == "__main__":
    analyze_lengths("/root/autodl-tmp/code_perference/datasets/dataset_cleaned", max_len=378)
