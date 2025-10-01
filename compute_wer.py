import sys
from jiwer import wer, compute_measures

def evaluate_wer(reference, hypothesis):
    measures = compute_measures(reference, hypothesis)
    score = wer(reference, hypothesis)
    return measures, score

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compute_wer.py <reference.txt> <hypothesis.txt>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        reference = f.read().strip()
    with open(sys.argv[2], "r") as f:
        hypothesis = f.read().strip()

    measures, score = evaluate_wer(reference, hypothesis)
    print("WER Score:", score)
    print("Detailed Measures:", measures)
