"""
data/generate_formality_labels.py
----------------------------------
Generates formality labels for languages that have no supervised
formality dataset: Japanese, Korean, Hindi, Arabic, German, Mandarin, Swahili.

This is itself a publishable contribution — we are the first paper to apply
LLM-assisted formality annotation to these language pairs and validate it
against native-speaker spot-checks.

Methodology (cited in paper as following Wang et al., 2023 — "Is ChatGPT
a Good Annotator?"):
  1. Sample sentences from OPUS-100 in each target language
  2. Ask Claude to classify each sentence as formal/neutral/casual
     with a chain-of-thought explanation
  3. Save all labels to data/llm_formality_labels.jsonl
  4. A random 10% sample (per language) is then spot-checked manually
     or by a second LLM (GPT-4) to compute inter-annotator agreement (Cohen's κ)

Usage:
    python data/generate_formality_labels.py
    python data/generate_formality_labels.py --lang ja --n 500
    python data/generate_formality_labels.py --compute-kappa  # after spot-check

Estimated cost: ~$0.40 for 500 sentences × 7 languages = 3,500 labels
"""

import os
import json
import time
import logging
import argparse
import random
from pathlib import Path
from dotenv import load_dotenv
from pathlib import Path

# Load .env relative to the project root (parent of data/)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

logger = logging.getLogger(__name__)

OUTPUT_FILE = Path("data/llm_formality_labels.jsonl")
SPOTCHECK_FILE = Path("data/spotcheck_sample.jsonl")

# Languages that need LLM-generated labels
# (English, French, Portuguese are covered by GYAFC + X-FORMAL)
TARGET_LANGUAGES = {
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "ar": "Arabic",
    "de": "German",
    "zh": "Mandarin Chinese",
    "sw": "Swahili",
}

VALID_LABELS = {"formal", "neutral", "casual"}


def load_target_language_sentences(target_lang: str, n: int) -> list[str]:
    """
    Load n sentences in target_lang from OPUS-100.
    These are the TARGET side of the translation pairs (not the English source).
    """
    try:
        from datasets import load_dataset

        for config in [f"en-{target_lang}", f"{target_lang}-en"]:
            try:
                dataset = load_dataset(
                    "Helsinki-NLP/opus-100",
                    config,
                    split="train",
                    streaming=True,
                )
                sentences = []
                for item in dataset:
                    trans = item.get("translation", {})
                    sent = trans.get(target_lang, "")
                    if sent and len(sent.split()) >= 5:  # skip very short sentences
                        sentences.append(sent)
                    if len(sentences) >= n:
                        break
                if sentences:
                    return sentences
            except Exception:
                continue
    except ImportError:
        logger.error("datasets not installed.")

    return []


def classify_formality_with_llm(
    sentence: str,
    lang: str,
    lang_name: str,
) -> dict | None:
    """
    Classify one sentence formality using Groq (primary), Gemini, or OpenAI fallback.
    Returns {"text": str, "label": str, "lang": str, "explanation": str} or None.
    """
    prompt = (
        f"You are an expert linguist specialising in {lang_name}.\n\n"
        f"Classify the following {lang_name} sentence as FORMAL, NEUTRAL, or CASUAL.\n\n"
        f"Definitions:\n"
        f"  FORMAL: Polite/honorific grammar, formal vocabulary, professional contexts.\n"
        f"  NEUTRAL: Everyday standard language, neither formal nor casual.\n"
        f"  CASUAL: Informal vocabulary, slang, contractions, friendly contexts.\n\n"
        f"Sentence: {sentence}\n\n"
        f'Return ONLY JSON: {{"label": "formal|neutral|casual", "explanation": "one sentence"}}'
    )

    # ── Try Groq first (free, fast, no rate limit issues) ──────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            label = data.get("label", "").lower().strip()
            if label in VALID_LABELS:
                return {
                    "text":        sentence,
                    "label":       label,
                    "lang":        lang,
                    "explanation": data.get("explanation", ""),
                }
        except Exception as e:
            logger.warning(f"Groq classification failed: {type(e).__name__}: {e}")

    # ── Try Gemini ─────────────────────────────────────────────────────────────
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=128,
                )
            )
            response = model.generate_content(prompt)
            data = json.loads(response.text)
            label = data.get("label", "").lower().strip()
            if label in VALID_LABELS:
                return {
                    "text":        sentence,
                    "label":       label,
                    "lang":        lang,
                    "explanation": data.get("explanation", ""),
                }
        except Exception as e:
            logger.warning(f"Gemini classification failed: {type(e).__name__}: {e}")

    # ── Try OpenAI ─────────────────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini", max_tokens=128, temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(resp.choices[0].message.content)
            label = data.get("label", "").lower().strip()
            if label in VALID_LABELS:
                return {
                    "text":        sentence,
                    "label":       label,
                    "lang":        lang,
                    "explanation": data.get("explanation", ""),
                }
        except Exception as e:
            logger.warning(f"OpenAI classification failed: {type(e).__name__}: {e}")

    logger.warning(
        f"All LLM backends failed for this sentence. "
        f"Check GROQ_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY in your .env file."
    )
    return None

def generate_labels_for_language(lang: str, lang_name: str, n: int = 500):
    """
    Generate n formality labels for one language.
    Appends to OUTPUT_FILE (JSONL format, one record per line).
    Skips sentences already processed (resumable).
    """
    # Load already-processed sentences to allow resuming
    existing = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["lang"] == lang:
                    existing.add(rec["text"])

    logger.info(f"\nGenerating labels for {lang_name} ({lang})")
    logger.info(f"Already have {len(existing)} labels for {lang}. Need {n}.")

    if len(existing) >= n:
        logger.info(f"Already have enough labels for {lang}. Skipping.")
        return

    sentences = load_target_language_sentences(lang, n * 2)  # load extra to allow filtering
    sentences = [s for s in sentences if s not in existing]
    sentences = sentences[:n - len(existing)]

    if not sentences:
        logger.warning(f"No sentences loaded for {lang}.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i, sentence in enumerate(sentences):
            result = classify_formality_with_llm(sentence, lang, lang_name)
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                success += 1

            if (i + 1) % 50 == 0:
                logger.info(f"  {lang}: {i+1}/{len(sentences)} processed, {success} successful")
                time.sleep(0.5)  # gentle rate limiting

    logger.info(f"Generated {success} labels for {lang_name}")


def generate_spotcheck_sample(spotcheck_pct: float = 0.10):
    """
    Sample 10% of generated labels per language for human/GPT-4 verification.
    Saves to data/spotcheck_sample.jsonl.

    Instructions for the spot-checker:
      - Open spotcheck_sample.jsonl
      - For each entry, verify whether the label is correct
      - Add a "verified_label" field with your assessment
      - Save the file — compute_kappa() will use it
    """
    if not OUTPUT_FILE.exists():
        logger.error("No labels file found. Run generation first.")
        return

    by_lang: dict[str, list] = {}
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_lang.setdefault(rec["lang"], []).append(rec)

    sample = []
    for lang, records in by_lang.items():
        n_sample = max(1, int(len(records) * spotcheck_pct))
        sampled = random.sample(records, n_sample)
        # Add an empty verified_label field for the spot-checker to fill in
        for rec in sampled:
            rec["verified_label"] = ""
            sample.append(rec)

    with open(SPOTCHECK_FILE, "w", encoding="utf-8") as f:
        for rec in sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(sample)} spot-check samples to {SPOTCHECK_FILE}")
    logger.info(
        "\nNext steps:\n"
        f"  1. Open {SPOTCHECK_FILE}\n"
        "  2. For each entry, fill in the 'verified_label' field\n"
        "     (native speaker or GPT-4 second-pass)\n"
        "  3. Run: python data/generate_formality_labels.py --compute-kappa"
    )


def compute_kappa():
    """
    Compute Cohen's κ between LLM-generated labels and human-verified labels.
    Requires spotcheck_sample.jsonl to have verified_label fields filled in.

    Reports:
      - Overall κ across all languages
      - Per-language κ
      - Agreement breakdown by label category

    The κ score goes in the paper as evidence that LLM annotation quality
    is sufficient for training the formality classifier.
    Target: κ > 0.6 (substantial agreement per Landis & Koch, 1977)
    """
    if not SPOTCHECK_FILE.exists():
        logger.error("No spot-check file found. Run --spotcheck first.")
        return

    from sklearn.metrics import cohen_kappa_score

    records = []
    with open(SPOTCHECK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("verified_label") and rec["verified_label"] in VALID_LABELS:
                records.append(rec)

    if not records:
        logger.error(
            "No verified labels found in spot-check file. "
            "Fill in the 'verified_label' field for each entry."
        )
        return

    llm_labels = [r["label"] for r in records]
    human_labels = [r["verified_label"] for r in records]

    overall_kappa = cohen_kappa_score(llm_labels, human_labels)
    logger.info(f"\n{'='*50}")
    logger.info(f"INTER-ANNOTATOR AGREEMENT (Cohen's κ)")
    logger.info(f"{'='*50}")
    logger.info(f"Overall κ: {overall_kappa:.3f}")

    if overall_kappa >= 0.8:
        interpretation = "Almost perfect agreement"
    elif overall_kappa >= 0.6:
        interpretation = "Substantial agreement — sufficient for paper"
    elif overall_kappa >= 0.4:
        interpretation = "Moderate agreement — marginal for paper, discuss as limitation"
    else:
        interpretation = "Fair/poor agreement — revise annotation prompt"

    logger.info(f"Interpretation: {interpretation}")

    # Per-language breakdown
    from collections import defaultdict
    by_lang = defaultdict(list)
    for rec in records:
        by_lang[rec["lang"]].append(rec)

    logger.info("\nPer-language κ:")
    for lang, lang_records in by_lang.items():
        if len(lang_records) < 5:
            logger.info(f"  {lang}: too few samples ({len(lang_records)})")
            continue
        llm = [r["label"] for r in lang_records]
        human = [r["verified_label"] for r in lang_records]
        try:
            k = cohen_kappa_score(llm, human)
            logger.info(f"  {lang}: κ = {k:.3f} (n={len(lang_records)})")
        except Exception:
            logger.info(f"  {lang}: could not compute (single class?)")

    # Save kappa report
    report_path = Path("data/kappa_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall κ: {overall_kappa:.3f} ({interpretation})\n")
        f.write(f"N verified: {len(records)}\n")
    logger.info(f"\nKappa report saved to {report_path}")
    logger.info("This number goes in Table 1 of the paper.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate formality labels via LLM")
    parser.add_argument("--lang", type=str, choices=list(TARGET_LANGUAGES.keys()),
                        help="Single language to process (default: all)")
    parser.add_argument("--n", type=int, default=100,
                        help="Labels per language (default: 500)")
    parser.add_argument("--spotcheck", action="store_true",
                        help="Generate spot-check sample after labelling")
    parser.add_argument("--compute-kappa", action="store_true",
                        help="Compute Cohen's κ from verified spot-check file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    if args.compute_kappa:
        compute_kappa()
    elif args.lang:
        lang_name = TARGET_LANGUAGES[args.lang]
        generate_labels_for_language(args.lang, lang_name, args.n)
        if args.spotcheck:
            generate_spotcheck_sample()
    else:
        for lang, lang_name in TARGET_LANGUAGES.items():
            generate_labels_for_language(lang, lang_name, args.n)
        if args.spotcheck:
            generate_spotcheck_sample()
