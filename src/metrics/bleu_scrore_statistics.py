import re
import numpy as np
from scipy import stats
def parse_data(file_path):
    epoch_data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        current_epoch = None
        for line in lines:
            epoch_match = re.match(r'\[Epoch (\d+)\]', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                if current_epoch not in epoch_data:
                    epoch_data[current_epoch] = []
            bleu_match = re.search(r'BLEU : ([\d\.]+)', line) # работает!!
            if bleu_match and current_epoch is not None:
                bleu_score = float(bleu_match.group(1))
                epoch_data[current_epoch].append(bleu_score)

    return epoch_data
def compute_statistics(bleu_scores):
    if not bleu_scores:
        return None
    n = len(bleu_scores)
    mean = np.mean(bleu_scores)
    std = np.std(bleu_scores, ddof=1)
    median = np.median(bleu_scores)
    confidence_level = 0.95
    t_value = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
    se = std / np.sqrt(n)
    margin_error = t_value * se
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    return {
        'mean': mean,
        'std': std,
        'median': median,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_samples': n
    }
def process_and_print_statistics(file_path):
    epoch_data = parse_data(file_path)
    for epoch, bleu_scores in sorted(epoch_data.items()):
        if bleu_scores:
            stats = compute_statistics(bleu_scores)
            print(f"\nЭпоха {epoch}:")
            print(f"Выборочное среднее: {stats['mean']:.2f}")
            print(f"Выборочное стандартное отклонение: {stats['std']:.2f}")
            print(f"Медиана: {stats['median']:.2f}")
            print(f"Доверительный интервал 95%: [{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]")
print(process_and_print_statistics("/Users/semencinman/Downloads/Losses_renet.txt"))