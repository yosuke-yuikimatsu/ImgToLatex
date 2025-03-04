import os
import sys
import logging
import argparse
import numpy as np
from PIL import Image

def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_path,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate rendered images.')
    parser.add_argument('--images-dir', required=True,
                        help='Directory with images_gold and images_pred folders')
    parser.add_argument('--log-path', default='evaluate.log',
                        help='Path to save evaluation logs')
    return parser.parse_args()

def load_image(path):
    try:
        return Image.open(path).convert('L')
    except:
        return None

def image_to_bits(img):
    arr = np.array(img).T
    return [''.join(['1' if p < 128 else '0' for p in row]) for row in arr]

def levenshtein_distance(s, t):
    """Вычисляет расстояние Левенштейна между двумя последовательностями."""
    m = len(s)
    n = len(t)
    d = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s[i-1] == t[j-1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i-1][j] + 1,   
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )
    return d[m][n]

def calculate_edit_distance(seq1, seq2):
    seq1_int = [int(bits, 2) for bits in seq1]
    seq2_int = [int(bits, 2) for bits in seq2]
    edit_dist = levenshtein_distance(seq1_int, seq2_int)
    return edit_dist, len(seq1_int)

def process_pair(gold_path, pred_path):
    gold_img = load_image(gold_path)
    pred_img = load_image(pred_path)
    
    if gold_img is None or pred_img is None:
        return 0, 0, False, False

    gold_bits = image_to_bits(gold_img)
    pred_bits = image_to_bits(pred_img) if pred_img else []
    
    edit_dist, ref_len = calculate_edit_distance(gold_bits, pred_bits)
    exact_match = (edit_dist == 0)
    
    gold_clean = [bits for bits in gold_bits if '1' in bits]
    pred_clean = [bits for bits in pred_bits if '1' in bits] if pred_img else []
    relaxed_match = (len(gold_clean) == len(pred_clean)) and all(g == p for g, p in zip(gold_clean, pred_clean))
    
    return edit_dist, ref_len, exact_match, relaxed_match

def main():
    args = parse_args()
    setup_logging(args.log_path)
    
    gold_dir = os.path.join(args.images_dir, 'images_gold')
    pred_dir = os.path.join(args.images_dir, 'images_pred')
    
    total_edit = 0
    total_ref = 0
    exact_matches = 0
    relaxed_matches = 0
    total = 0
    
    for gold_file in os.listdir(gold_dir):
        gold_path = os.path.join(gold_dir, gold_file)
        pred_path = os.path.join(pred_dir, gold_file)
        
        if not os.path.exists(pred_path):
            logging.warning(f"Missing prediction for {gold_file}")
            continue
            
        edit_dist, ref_len, exact, relaxed = process_pair(gold_path, pred_path)
        total_edit += edit_dist
        total_ref += ref_len
        exact_matches += exact
        relaxed_matches += relaxed
        total += 1
        
        if total % 100 == 0:
            logging.info(f"Processed {total} pairs | "
                        f"Current Accuracy: {exact_matches/total:.3f} "
                        f"({relaxed_matches/total:.3f}) | "
                        f"Edit Distance: {1 - total_edit/total_ref:.3f}")
    
    logging.info("\nFinal Evaluation Results:")
    logging.info(f"Exact Match Accuracy: {exact_matches/total:.4f}")
    logging.info(f"Relaxed Match Accuracy: {relaxed_matches/total:.4f}")
    logging.info(f"Normalized Edit Distance: {1 - total_edit/total_ref:.4f}")

if __name__ == '__main__':
    main()