"""
eval/ablation.py
----------------
Anvita's Module 7 — Ablation Study

Proves that each component of the cultural intelligence layer adds
measurable value independently. This is the most important table in
the paper — without it, a reviewer can argue the improvement comes
from the LLM alone.

Four conditions tested:
  A: Raw Claude, no cultural context, no CoT
  B: Claude + formality classifier only (no CSI, no CoT)
  C: Claude + CoT only (no formality, no CSI)
  D: Claude + full cultural context object — our complete system

Why this structure:
  - Condition A is the true baseline: what you get if you just ask Claude
    to translate with a one-line prompt, like any naive implementation
  - Conditions B and C isolate individual contributions
  - Condition D is our full system
  - The delta between A and D is the total improvement
  - The deltas between B/C and D show which component contributes more
  - If D >> A but B ≈ C ≈ A, that would mean our contributions don't
    work individually — but we expect D >> C >> B >> A

Usage:
    python eval/ablation.py --lang ja --n 50
    python eval/ablation.py --all --n 30
"""

import os
import json
import logging
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SUPPORTED_LANGS = ["fr", "es", "ja", "hi", "ar", "zh", "de", "sw", "pt", "ko"]
RESULTS_DIR = Path("eval/results")

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish",
    "ja": "Japanese", "hi": "Hindi", "ar": "Arabic",
    "zh": "Mandarin Chinese", "de": "German",
    "sw": "Swahili", "pt": "Brazilian Portuguese", "ko": "Korean",
}


def load_flores_sample(target_lang: str, n: int = 50) -> list[dict]:
    """Load n sentences from FLORES-200 devtest. Reuses loader from bleu_comet.py."""
    from eval.bleu_comet import load_flores_sample as _load
    return _load(target_lang, n)


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus BLEU. Returns score as float."""
    from eval.bleu_comet import compute_bleu as _bleu
    result = _bleu(hypotheses, references)
    return result["score"]


def compute_comet(sources, hypotheses, references) -> float | None:
    """Compute COMET score. Returns float or None if unavailable."""
    from eval.bleu_comet import compute_comet as _comet
    result = _comet(sources, hypotheses, references)
    return result["score"]


# ─── Condition A: Raw Claude (no cultural context, no CoT) ────────────────────

def condition_a_raw_claude(samples: list[dict], target_lang: str) -> list[str]:
    """
    Simplest possible translation prompt — just 'Translate this to X'.
    This is what any naive implementation would do.
    No formality control, no CSI detection, no CoT, no few-shot examples.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    hypotheses = []

    for i, sample in enumerate(samples):
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": f"Translate to {lang_name}: {sample['source']}"
                }],
            )
            hypotheses.append(msg.content[0].text.strip())
        except Exception as e:
            logger.warning(f"Condition A failed for sample {i}: {e}")
            hypotheses.append("")

        if (i + 1) % 10 == 0:
            logger.info(f"  Condition A: {i+1}/{len(samples)}")

    return hypotheses


# ─── Condition B: Claude + Formality Only ─────────────────────────────────────

def condition_b_formality_only(samples: list[dict], target_lang: str) -> list[str]:
    """
    Claude with formality detection injected into the prompt,
    but NO CSI detection, NO few-shot examples, NO CoT.
    Isolates the contribution of formality analysis.
    """
    import anthropic
    from pipeline.formality_classifier import classify_formality

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    hypotheses = []

    for i, sample in enumerate(samples):
        try:
            formality, _ = classify_formality(sample["source"], "en")
            prompt = (
                f"Translate the following text to {lang_name}. "
                f"The source text is written in a {formality} register. "
                f"Match the register in your translation.\n\n"
                f"Text: {sample['source']}\n\n"
                f"Translation:"
            )
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            hypotheses.append(msg.content[0].text.strip())
        except Exception as e:
            logger.warning(f"Condition B failed for sample {i}: {e}")
            hypotheses.append("")

        if (i + 1) % 10 == 0:
            logger.info(f"  Condition B: {i+1}/{len(samples)}")

    return hypotheses


# ─── Condition C: Claude + CoT Only ──────────────────────────────────────────

def condition_c_cot_only(samples: list[dict], target_lang: str) -> list[str]:
    """
    Claude with CoT prompting but NO formality classifier,
    NO CSI detection, NO few-shot examples.
    Isolates the contribution of chain-of-thought reasoning.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    hypotheses = []

    for i, sample in enumerate(samples):
        try:
            prompt = (
                f"Translate the following text to {lang_name}.\n\n"
                f"Think step by step:\n"
                f"1. Identify any idioms or culturally specific phrases\n"
                f"2. Consider the appropriate register for {lang_name}\n"
                f"3. Produce the best translation\n\n"
                f"Text: {sample['source']}\n\n"
                f"Respond with only the translation, no explanation."
            )
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            hypotheses.append(msg.content[0].text.strip())
        except Exception as e:
            logger.warning(f"Condition C failed for sample {i}: {e}")
            hypotheses.append("")

        if (i + 1) % 10 == 0:
            logger.info(f"  Condition C: {i+1}/{len(samples)}")

    return hypotheses


# ─── Condition D: Full System ─────────────────────────────────────────────────

def condition_d_full_system(samples: list[dict], target_lang: str) -> list[str]:
    """
    Our complete pipeline: formality + CSI + few-shot + CoT.
    Uses the formal variant as the primary output (most comparable
    to what Google Translate and DeepL produce by default).
    """
    from pipeline.variant_formatter import run
    hypotheses = []

    for i, sample in enumerate(samples):
        try:
            translation, _ = run(
                text=sample["source"],
                target_lang=target_lang,
                source_lang="en",
            )
            hypotheses.append(translation["formal"])
        except Exception as e:
            logger.warning(f"Condition D failed for sample {i}: {e}")
            hypotheses.append("")

        if (i + 1) % 10 == 0:
            logger.info(f"  Condition D: {i+1}/{len(samples)}")

    return hypotheses


# ─── Main ablation runner ─────────────────────────────────────────────────────

def run_ablation(target_lang: str, n: int = 50) -> pd.DataFrame:
    """
    Run all four ablation conditions for one language pair.
    Saves results to eval/results/ablation_{lang}.csv.

    Args:
        target_lang: ISO 639-1 target language code
        n:           Number of FLORES sentences to evaluate

    Returns:
        DataFrame with columns: condition, bleu, comet, delta_bleu_vs_A
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Running ablation study: en → {target_lang} ({n} sentences)")
    logger.info(f"{'='*60}\n")

    samples = load_flores_sample(target_lang, n)
    sources = [s["source"] for s in samples]
    references = [s["reference"] for s in samples]

    conditions = {
        "A: Raw Claude (no context)": condition_a_raw_claude,
        "B: + Formality only": condition_b_formality_only,
        "C: + CoT only": condition_c_cot_only,
        "D: Full system (ours)": condition_d_full_system,
    }

    rows = []
    baseline_bleu = None

    for condition_name, condition_fn in conditions.items():
        logger.info(f"Running condition: {condition_name}")
        hypotheses = condition_fn(samples, target_lang)

        bleu = compute_bleu(hypotheses, references)
        comet = compute_comet(sources, hypotheses, references)

        if baseline_bleu is None:
            baseline_bleu = bleu  # Condition A sets the baseline

        delta = round(bleu - baseline_bleu, 2) if baseline_bleu is not None else 0.0

        rows.append({
            "condition": condition_name,
            "lang": target_lang,
            "bleu": bleu,
            "comet": comet,
            "delta_bleu_vs_A": f"+{delta}" if delta >= 0 else str(delta),
            "n": n,
        })

        logger.info(f"  BLEU: {bleu} | COMET: {comet} | Δ vs A: {delta:+.2f}")

    df = pd.DataFrame(rows)

    # Save CSV
    output_path = RESULTS_DIR / f"ablation_{target_lang}.csv"
    df.to_csv(output_path, index=False)

    # Pretty print for terminal
    print(f"\n{'='*60}")
    print(f"ABLATION RESULTS — en → {LANGUAGE_NAMES.get(target_lang, target_lang)}")
    print(f"{'='*60}")
    print(df[["condition", "bleu", "comet", "delta_bleu_vs_A"]].to_string(index=False))
    print(f"\nSaved to: {output_path}")

    return df


def run_all_ablations(n: int = 30) -> pd.DataFrame:
    """
    Run ablation study across all 10 target languages.
    Warning: this makes many API calls. At n=30 per language,
    expect ~120 API calls per language = 1200 total calls.
    Estimated cost: ~$1.50. Estimated time: ~20 minutes.
    """
    all_results = []
    for lang in SUPPORTED_LANGS:
        try:
            df = run_ablation(lang, n=n)
            all_results.append(df)
        except Exception as e:
            logger.error(f"Ablation failed for {lang}: {e}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = RESULTS_DIR / "ablation_all_languages.csv"
        combined.to_csv(output_path, index=False)
        logger.info(f"\nCombined ablation results saved to {output_path}")
        return combined

    return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--lang",
        type=str,
        default="ja",
        choices=SUPPORTED_LANGS,
        help="Target language (default: ja)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 10 languages (slow, ~20 mins)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of FLORES sentences per language (default: 50)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    if args.all:
        run_all_ablations(n=args.n)
    else:
        run_ablation(args.lang, n=args.n)