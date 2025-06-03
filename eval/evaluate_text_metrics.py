import json
from tqdm import tqdm
from rouge import rouge
from bleu import compute_bleu

def load_json_data(data_path):
    with open(data_path, 'r') as f:
        data_list = json.load(f)
    return data_list

def evaluate_bleu_rouge_from_file(json_path: str):
    """
    Evaluate BLEU-4 and ROUGE scores on a JSON file with keys:
    - code_lables: reference code
    - code_reply: generated code

    Returns:
        dict: contains bleu-4 and rouge-1/2/l scores (F, P, R)
    """
    code_reply_data = load_json_data(json_path)
    epoch = 0
    total_val = 0.0
    rouge_map = {
      "rouge_1/f_score": 0.0, "rouge_1/r_score": 0.0, "rouge_1/p_score": 0.0,
      "rouge_2/f_score": 0.0, "rouge_2/r_score": 0.0, "rouge_2/p_score": 0.0,
      "rouge_l/f_score": 0.0, "rouge_l/r_score": 0.0, "rouge_l/p_score": 0.0
    }
    for data in tqdm(code_reply_data, desc="Evaluating BLEU and ROUGE"):
        tokens_test_lables = [data['code_lables'].strip('\n')]
        tokens_predict = [data['code_reply']]
        tokens_test_lables_list = [tokens_test_lables]

        result = compute_bleu(tokens_test_lables_list, tokens_predict, 4)
        epoch += 1
        total_val += result[0]

        ROUGE = rouge(tokens_test_lables, tokens_predict)
        for (k, v) in ROUGE.items():
            rouge_map[k] += v

    average_val = total_val / epoch
    for (k, v) in rouge_map.items():
        rouge_map[k] = v / epoch

    metrics = {"bleu_4": average_val, **rouge_map}
    return metrics

def print_metrics(metrics: dict):
    print("BLEU-4 = {:.4f}".format(metrics["bleu_4"]))
    for k, v in metrics.items():
        if k != "bleu_4":
            print("{} {:7.4f}".format(k, v))

if __name__ == "__main__":
    print("Running BLEU/ROUGE evaluation...")
    path = "/home/develop/dzl/PreferCodeLlama/out_predict/softprompt_Short1121.json"
    results = evaluate_bleu_rouge_from_file(path)
    print_metrics(results)
    
# === evaluate_text_metrics.py ===
from rouge import rouge
from bleu import compute_bleu


def evaluate_text_metrics(references, predictions):
    assert len(references) == len(predictions), "Reference and prediction count mismatch"
    references = [[ref.strip()] for ref in references]
    predictions = [pred.strip() for pred in predictions]

    # Compute BLEU
    bleu_score, _, _, _, _, _ = compute_bleu(references, predictions, max_order=4, smooth=True)

    # Compute ROUGE
    rouge_scores = rouge(predictions, [ref[0] for ref in references])

    return {
        "bleu_4": round(bleu_score, 4),
        **{k: round(v, 4) for k, v in rouge_scores.items()}
    }

