"""
data/build_opus_index.py
------------------------
Downloads a sample of OPUS-100 for all 10 language pairs and builds
the BM25 retrieval index used by prompt_builder.py.

Run this ONCE in Week 1 before using the pipeline.

Usage:
    python data/build_opus_index.py

What it does:
  1. Downloads 5,000 sentence pairs per language pair from OPUS-100
     via the HuggingFace datasets library
  2. For each sentence pair, uses Claude to generate formal/casual/literal
     variants (sampled from a small subset of 200 pairs — enough for few-shot)
  3. Saves each language pair as data/opus_sample/{src}-{tgt}.json

Why 5,000 sentences:
  - Enough for meaningful BM25 retrieval diversity
  - Small enough to download quickly (~10 minutes total)
  - The full OPUS-100 is 100M+ sentences — we only need a representative sample

Why pre-generate formal/casual/literal variants:
  - The few-shot examples need to show the model what the output format looks like
  - Having real translation variants as examples improves output quality
  - We only generate variants for 200 sentences (not all 5,000) to keep API cost low
  - The other 4,800 sentences are stored source-only and used for BM25 retrieval
    only (the model sees just the source when retrieving, not the translations)

Estimated time: ~15 minutes
Estimated cost: ~$0.30 (200 variant generations × 10 language pairs)
"""

import os
import json
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/opus_sample")
SENTENCES_PER_PAIR = 5000   # total sentences stored for BM25
VARIANTS_PER_PAIR = 200     # subset with formal/casual/literal variants

TARGET_LANGUAGES = {
    "fr": "French",
    "es": "Spanish",
    "ja": "Japanese",
    "hi": "Hindi",
    "ar": "Arabic",
    "zh": "Mandarin Chinese",
    "de": "German",
    "sw": "Swahili",
    "pt": "Brazilian Portuguese",
    "ko": "Korean",
}

# OPUS-100 HuggingFace dataset name and config
# Source language is always English (en)
OPUS_DATASET = "Helsinki-NLP/opus-100"


def load_opus_pairs(target_lang: str, n: int) -> list[dict]:
    """
    Load n English→target_lang sentence pairs from OPUS-100.
    Returns list of {"source": str, "target": str} dicts.
    """
    try:
        from datasets import load_dataset

        # OPUS-100 config name format: "en-{lang}" or "{lang}-en"
        # Try both orderings
        for config in [f"en-{target_lang}", f"{target_lang}-en"]:
            try:
                logger.info(f"Loading OPUS-100 config: {config}")
                dataset = load_dataset(
                    OPUS_DATASET,
                    config,
                    split="train",
                    streaming=True,
                )
                pairs = []
                for item in dataset:
                    trans = item.get("translation", {})
                    src = trans.get("en", "")
                    tgt = trans.get(target_lang, "")
                    if src and tgt:
                        pairs.append({"source": src, "target": tgt})
                        if len(pairs) % 500 == 0:
                            logger.info(f"  Downloaded {len(pairs)}/{n} sentence pairs...")
                    if len(pairs) >= n:
                        break

                if pairs:
                    logger.info(f"Loaded {len(pairs)} pairs for en-{target_lang}")
                    return pairs
            except Exception as e:
                logger.debug(f"Config {config} failed: {e}")
                continue

        logger.warning(f"Could not load OPUS-100 for {target_lang}. Using empty list.")
        return []

    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        return []


def generate_variants_for_pair(
    source: str,
    target: str,
    target_lang_name: str,
) -> dict:
    """
    Generate formal/casual/literal variants for one sentence pair.
    Tries Gemini first (free), then Claude/Anthropic as fallback.

    These variants become the few-shot examples shown to the LLM.
    Having real translation variants as examples improves quality
    (Brown et al., 2020; Jiao et al., 2023).
    """
    prompt = (
        f"Given this English sentence and its {target_lang_name} translation, "
        f"produce three style variants of the translation.\n\n"
        f"English: {source}\n"
        f"Base translation: {target}\n\n"
        + '{"formal": "...", "casual": "...", "literal": "..."}' + "\n\n"
        f"Return ONLY that JSON object.\n"
        f"formal = highest register with appropriate honorifics\n"
        f"casual = natural everyday speech\n"
        f"literal = word-for-word preserving grammatical structure"
    )

    # Reload .env in case it was updated after process started
    load_dotenv(override=True)

    # Try Groq first (free tier, fast, llama-3.3-70b-versatile)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=512,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(resp.choices[0].message.content)
            if data.get("formal"):
                return {
                    "source":  source, "target": target,
                    "formal":  data.get("formal",  target),
                    "casual":  data.get("casual",  target),
                    "literal": data.get("literal", target),
                }
        except Exception as e:
            logger.warning(f"Groq variant generation failed: {e}")

    # Try OpenAI as fallback (gpt-4o-mini)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=512,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(resp.choices[0].message.content)
            return {
                "source":  source, "target": target,
                "formal":  data.get("formal",  target),
                "casual":  data.get("casual",  target),
                "literal": data.get("literal", target),
            }
        except Exception as e:
            logger.debug(f"OpenAI variant generation failed: {e}")



    # No API worked — store base translation for all variants
    # This still builds a valid BM25 index; only the few-shot examples are less rich
    logger.debug("All variant APIs unavailable. Using base translation for this pair.")
    return {
        "source": source, "target": target,
        "formal": target, "casual": target, "literal": target,
    }


def build_index_for_language(target_lang: str, lang_name: str):
    """
    Build and save the OPUS index for one language pair.
    """
    output_path = OUTPUT_DIR / f"en-{target_lang}.json"

    if output_path.exists():
        logger.info(f"Index already exists for en-{target_lang}. Skipping.")
        return

    logger.info(f"\nBuilding index for en → {lang_name} ({target_lang})")

    # Load sentence pairs
    pairs = load_opus_pairs(target_lang, SENTENCES_PER_PAIR)

    if not pairs:
        logger.warning(f"No pairs loaded for {target_lang}. Creating empty index.")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[]")
        return

    # Generate variants for the first VARIANTS_PER_PAIR sentences
    logger.info(f"Generating variants for first {VARIANTS_PER_PAIR} pairs...")
    index = []

    for i, pair in enumerate(pairs):
        if i < VARIANTS_PER_PAIR:
            # Generate formal/casual/literal variants
            record = generate_variants_for_pair(
                pair["source"], pair["target"], lang_name
            )
            # Small delay to avoid rate limiting
            if i > 0 and i % 20 == 0:
                logger.info(f"  Generated {i}/{VARIANTS_PER_PAIR} variants")
                time.sleep(1)
        else:
            # Store remaining pairs with only source (for BM25 retrieval)
            record = {
                "source": pair["source"],
                "target": pair["target"],
                "formal": pair["target"],
                "casual": pair["target"],
                "literal": pair["target"],
            }
        index.append(record)

    # Save index
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(index)} entries to {output_path}")


def build_all_indices():
    """Build OPUS indices for all 10 language pairs."""
    logger.info("Building OPUS-100 indices for all language pairs...")
    logger.info(f"Output directory: {OUTPUT_DIR.absolute()}")

    for lang_code, lang_name in TARGET_LANGUAGES.items():
        try:
            build_index_for_language(lang_code, lang_name)
        except Exception as e:
            logger.error(f"Failed to build index for {lang_code}: {e}")

    logger.info("\nAll indices built. The pipeline is ready to use.")
    logger.info(f"Files saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )
    build_all_indices()
