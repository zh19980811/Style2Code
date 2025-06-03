# === save_results_markdown.py ===
import os
from datetime import datetime

def save_outputs_as_markdown(inputs, references, generations, save_dir="logs"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(save_dir, f"generated_results_{timestamp}.md")

    with open(md_path, "w", encoding="utf-8") as f:
        for i, (inp, ref) in enumerate(zip(inputs, references)):
            f.write(f"## Sample {i}\n\n")

            f.write("**Input Prompt**  \n")
            f.write("```text\n" + inp.strip() + "\n```\n\n")

            f.write("**Reference Code (code2)**  \n")
            f.write("```python\n" + ref.strip() + "\n```\n\n")

            for model_name, model_outputs in generations.items():
                f.write(f"**{model_name}**  \n")
                f.write("```python\n" + model_outputs[i].strip() + "\n```\n\n")

    print(f"✅ Markdown file saved to {md_path}")
