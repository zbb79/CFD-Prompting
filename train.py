from cdf_prompt import  CFD_CoT
import pandas as pd
from tqdm import tqdm
import json
import os
import csv


DATA_PATH = "data/sampled_hotpotqa.csv"
OUTPUT_DIR = "out2/denghotpot2_cfd_llama33"
RESULTS_FILE = f"{OUTPUT_DIR}/results.csv"
STATS_FILE = f"{OUTPUT_DIR}/stats.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)


sciq_data = pd.read_csv(DATA_PATH)


required_columns = {'question', 'correct_answer', 'support'}
if not required_columns.issubset(sciq_data.columns):
    raise ValueError(f"wrong")


stats = {
    'total_em': 0,
    'total_f1': 0,
    'total_questions': 0,
    'last_processed_index': -1,
    'running_em': 0.0,
    'running_f1': 0.0
}


if os.path.exists(STATS_FILE):
    with open(STATS_FILE, 'r') as f:
        stats = json.load(f)
    start_idx = stats['last_processed_index'] + 1
else:
    start_idx = 0


if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'correct_answer', 'predicted_answer'])
        writer.writeheader()


total_questions = len(sciq_data) - start_idx
pbar = tqdm(
    range(start_idx, len(sciq_data)),
    desc="Processing questions",
    initial=start_idx,
    total=len(sciq_data),
    dynamic_ncols=True,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


for index in pbar:
    row = sciq_data.iloc[index]
    question = row['question']
    correct_answer = row['correct_answer']
    support = row['support']

    try:
        predicted_answer=  CFD_CoT(question,support)
    except Exception as e:

            predicted_answer = ""
            vote_answer=""

    if predicted_answer is None:
        em = 0  
        f1 = 0
    else:
        em = int(predicted_answer.strip().lower() == correct_answer.strip().lower())
        correct_tokens = set(correct_answer.lower().split())
        predicted_tokens = set(predicted_answer.lower().split())
        common_tokens = correct_tokens & predicted_tokens
        if correct_tokens and predicted_tokens:
            precision = len(common_tokens) / len(predicted_tokens)
            recall = len(common_tokens) / len(correct_tokens)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            f1 = 0


    stats['total_em'] += em
    stats['total_f1'] += f1
    stats['total_questions'] += 1
    stats['last_processed_index'] = index
    stats['running_em'] = stats['total_em'] / stats['total_questions']
    stats['running_f1'] = stats['total_f1'] / stats['total_questions']

    pbar.set_postfix({
        'EM': f"{stats['running_em']:.4f}",
        'F1': f"{stats['running_f1']:.4f}"
    })

    with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'correct_answer', 'predicted_answer'])
        writer.writerow({
            'question': question,
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
        })


    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=4)


pbar.close()


average_em = stats['total_em'] / stats['total_questions'] if stats['total_questions'] > 0 else 0
average_f1 = stats['total_f1'] / stats['total_questions'] if stats['total_questions'] > 0 else 0


print(f"\nFinal Results:")
print(f"Total Questions Processed: {stats['total_questions']}")
print(f"EM Score: {average_em:.4f}")
print(f"F1 Score: {average_f1:.4f}")
