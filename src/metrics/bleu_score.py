import argparse
from collections import Counter
import math
import sys


def n_gram_counts(tokens, n):
    return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])


def count_clip(candidate, references, n):
    counts = n_gram_counts(candidate, n)
    if not counts:
        return 0, 0
    max_counts = {}
    for reference in references:
        reference_counts = n_gram_counts(reference, n)
        for n_gram in counts:
            if n_gram in max_counts:
                max_counts[n_gram] = max(
                    max_counts[n_gram], reference_counts[n_gram])
            else:
                max_counts[n_gram] = reference_counts[n_gram]
    clipped_counts = {ng: min(count, max_counts[ng])
                      for ng, count in counts.items()}
    return sum(clipped_counts.values()), sum(counts.values())


def closest_reference_length(candidate, references):
    candidate_len = len(candidate)
    ref_lens = (len(ref) for ref in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len: (
        abs(ref_len - candidate_len), ref_len))
    return closest_ref_len


def brevity_penalty(trans_len, ref_len):
    if trans_len > ref_len:
        return 1
    if trans_len == 0:
        return 0
    return math.exp(1 - ref_len / trans_len)


def compute_bleu(candidate, references, max_n=4):
    weights = [0.25] * max_n
    p_ns = [0] * max_n
    candidate_len = len(candidate)
    ref_len = closest_reference_length(candidate, references)
    for i in range(max_n):
        c, t = count_clip(candidate, references, i+1)
        if t == 0:
            p_ns[i] = 0
        else:
            p_ns[i] = c / t
    if min(p_ns) > 0:
        s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns))
        s = math.exp(s)
    else:
        s = 0
    bp = brevity_penalty(candidate_len, ref_len)
    bleu = s * bp
    return bleu * 100


def main():
    parser = argparse.ArgumentParser(
        description="Compute BLEU score for machine translation.")
    parser.add_argument("reference", type=argparse.FileType(
        'r'), help="Reference translation file.")
    parser.add_argument("candidate", type=argparse.FileType(
        'r'), help="Candidate translation file.")
    args = parser.parse_args()

    references = [line.strip().split() for line in args.reference]
    candidates = [line.strip().split() for line in args.candidate]

    if len(references) != len(candidates):
        print("Error: The number of lines in the reference and candidate files must be the same.")
        sys.exit(1)

    total_bleu = 0
    for candidate, reference in zip(candidates, references):
        total_bleu += compute_bleu(candidate, [reference])

    average_bleu = total_bleu / len(candidates)
    print(f"BLEU = {average_bleu:.2f}")


if __name__ == "__main__":
    main()
