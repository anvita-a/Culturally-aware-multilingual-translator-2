"""
eval package

Usage:
    python eval/bleu_comet.py --lang ja --n 100
    python eval/ablation.py --lang ja --n 50
"""

from eval.bleu_comet import run_evaluation, compute_bleu, compute_comet
from eval.ablation import run_ablation, run_all_ablations

__all__ = [
    "run_evaluation",
    "compute_bleu",
    "compute_comet",
    "run_ablation",
    "run_all_ablations",
]