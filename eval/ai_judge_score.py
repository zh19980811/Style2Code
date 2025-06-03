# === ai_judge_score.py ===
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key="your-api-key", base_url="https://api.openai.com/v1")

JUDGE_PROMPT_TEMPLATE = (
    "You are an AI code evaluator. Given two code snippets, answer strictly with '0' if the rewritten version correctly preserves the functionality and style of the reference, or '1' if it has clear issues."
    "\n\nReference Code:\n{ref}\n\nGenerated Code:\n{gen}\n\nAnswer:"
)

def ai_judge_score(generated_list, reference_list):
    scores = []
    for ref, gen in tqdm(zip(reference_list, generated_list), total=len(generated_list), desc="AI Judging"):
        prompt = JUDGE_PROMPT_TEMPLATE.format(ref=ref.strip(), gen=gen.strip())
        try:
            response = client.chat.completions.create(
                model="gpt-4",  # or your preferred model
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            score = 1 if '1' in content else 0
        except Exception as e:
            score = 1  # penalize on error
        scores.append(score)
    return scores
