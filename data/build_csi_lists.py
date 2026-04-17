"""
build_csi_lists.py
------------------
Downloads FLORES-200 and uses it as a base corpus to build the CSI term list.

What this script does:
  1. Downloads 1,012 English sentences from FLORES-200 devtest (Meta's multilingual benchmark)
  2. For each sentence, uses the Gemini API to identify culture-specific items
  3. Aggregates all found CSI terms across all sentences
  4. Merges with a curated seed list of important terms
  5. Saves the final de-duplicated list to data/csi_terms.json

Why FLORES-200:
  This is the standard evaluation benchmark for multilingual MT research.
  Using it as our annotation corpus means we can cite:
  "We identified CSI prevalence on FLORES-200 English devtest (n=1,012 sentences)"
  which is a credible, reproducible, peer-reviewed source.

Usage:
  python3 data/build_csi_lists.py
  python3 data/build_csi_lists.py --sentences 100    # faster, for testing
  python3 data/build_csi_lists.py --sentences 1012   # full corpus

Output:
  data/csi_terms.json           — the CSI term lists used by csi_detector.py
  data/flores_csi_stats.json    — statistics for your paper (CSI prevalence, etc.)

Prerequisites:
  pip install datasets google-generativeai
  Set GEMINI_API_KEY in .env
"""

import os
import sys
import json
import time
import argparse
import logging
from collections import Counter, defaultdict
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Seed terms ────────────────────────────────────────────────────────────────
# These are the guaranteed-important terms that go into csi_terms.json
# regardless of what FLORES-200 contains.
# The dataset enriches this list; it doesn't replace it.

SEED_TERMS = {
    "culturally_embedded": [
        # English idioms
        "break the ice", "bite the bullet", "piece of cake", "under the weather",
        "once in a blue moon", "spill the beans", "let the cat out of the bag",
        "beat around the bush", "burn bridges", "costs an arm and a leg",
        "kill two birds with one stone", "speak of the devil",
        "raining cats and dogs", "over the moon", "miss the boat",
        "pull someone's leg", "the best of both worlds", "time flies",
        "blessing in disguise", "better late than never",
        "actions speak louder than words", "hit the ground running",
        "back to the drawing board", "every cloud has a silver lining",
        "face the music", "no pain no gain", "on thin ice",
        "read between the lines", "see eye to eye", "throw in the towel",
        "the tip of the iceberg", "up in the air", "let sleeping dogs lie",
        "the elephant in the room", "burning the midnight oil", "drop the ball",
        # Untranslatable concepts
        "schadenfreude", "weltanschauung", "angst", "zeitgeist",
        "wanderlust", "gemütlichkeit", "doppelganger", "kaizen", "ikigai",
        "wabi-sabi", "mono no aware", "honne and tatemae", "gaman",
        "nemawashi", "karoshi", "hygge", "lagom", "ubuntu", "saudade",
        "joie de vivre", "esprit de corps", "savoir faire", "je ne sais quoi",
        "raison d'être", "karma", "dharma", "nirvana", "zen", "yin and yang",
        "feng shui", "saving face", "losing face", "machismo",
        # Beliefs and superstitions
        "knock on wood", "touch wood", "evil eye", "jinx",
        "friday the 13th", "four-leaf clover", "beginner's luck",
        "murphy's law", "crossed fingers",
        # Cultural humour
        "knock knock", "that's what she said", "ghosting", "gaslighting",
        "humble brag", "cancel culture", "fomo", "yolo",
        "the american dream", "keeping up with the joneses",
        "the rat race", "glass ceiling", "going postal",
        "dad joke", "banter", "dry humour", "dry humor",
        # Proverbs
        "the early bird catches the worm", "a penny saved is a penny earned",
        "don't count your chickens before they hatch", "you reap what you sow",
        "the grass is always greener on the other side",
        "when in rome do as the romans do", "a stitch in time saves nine",
        # Cultural humour
        "knock knock", "that's what she said", "ghosting", "gaslighting",
        "humble brag", "cancel culture", "the american dream",
        "keeping up with the joneses", "the rat race", "glass ceiling",
    ],
    "proper_name": [
        # Holidays
        "christmas", "christmas eve", "thanksgiving", "thanksgiving dinner",
        "halloween", "trick or treat", "diwali", "eid", "ramadan",
        "hanukkah", "kwanzaa", "chinese new year", "lunar new year",
        "day of the dead", "mardi gras", "st patrick's day", "passover",
        "holi", "dussehra", "navratri", "ganesh chaturthi", "onam", "pongal",
        "chuseok", "golden week", "obon", "hanami", "matsuri", "boxing day",
        "bonfire night", "bastille day", "cinco de mayo", "oktoberfest",
        # Japanese foods
        "mochi", "onigiri", "ramen", "sushi", "sashimi", "tempura",
        "yakitori", "gyoza", "takoyaki", "okonomiyaki", "udon", "soba",
        "matcha", "wagashi", "anko", "natto", "bento", "izakaya",
        "dorayaki", "taiyaki", "sakura mochi",
        # Chinese foods
        "dim sum", "baozi", "xiaolongbao", "peking duck", "mooncake",
        "tangyuan", "nian gao", "hot pot", "char siu",
        # Korean foods
        "kimchi", "bibimbap", "bulgogi", "tteokbokki", "samgyeopsal",
        "banchan", "mukbang", "japchae",
        # Indian foods
        "biryani", "naan", "roti", "samosa", "dal", "paneer", "chaat",
        "pani puri", "lassi", "masala chai", "gulab jamun", "jalebi",
        "thali", "tiffin", "paratha",
        # Middle Eastern foods
        "hummus", "falafel", "shawarma", "mezze", "baklava", "halal",
        "iftar", "suhoor",
        # British / European foods
        "fish and chips", "bangers and mash", "full english", "scones",
        "sunday roast", "shepherd's pie", "haggis", "bratwurst", "schnitzel",
        "baguette", "croissant", "paella", "tapas", "tiramisu", "gelato",
        "stroopwafel", "pretzels",
        # Sports / media
        "super bowl", "world series", "march madness", "wimbledon",
        "the masters", "premier league", "world cup", "bollywood",
        "hollywood", "k-pop", "k-drama", "anime", "manga", "kabuki",
        "telenovela",
        # Institutions
        "nfl", "nba", "nhs", "ivy league", "oxbridge", "gcse", "a-levels",
    ],
    "institutional": [
        "llc", "inc", "ltd", "plc", "gmbh", "ngo",
        "ceo", "cfo", "cto", "401k", "ira",
        "medicare", "medicaid", "social security",
        "gdpr", "hipaa", "fda", "sec", "irs", "hmrc",
        "ofsted", "osha", "epa", "miranda rights",
        "electoral college", "filibuster", "gerrymandering",
        "hmo", "ppo", "common core", "prom", "homecoming",
    ],
    "pragmatic": [
        "yours sincerely", "yours faithfully", "yours truly",
        "kind regards", "best regards", "warm regards",
        "think nothing of it", "don't mention it", "my pleasure",
        "the pleasure is mine", "no worries", "not a problem",
        "cheers", "fair dinkum", "how do you do",
    ],
}


def download_flores200(n_sentences: int = 200) -> List[str]:
    """
    Download English sentences from FLORES-200 devtest split.
    Returns up to n_sentences sentences.
    """
    logger.info(f"Downloading FLORES-200 devtest ({n_sentences} sentences)...")
    try:
        from datasets import load_dataset

        # Try multiple dataset paths (HuggingFace paths change over time)
        paths_to_try = [
            ("facebook/flores", "eng_Latn"),
            ("Muennighoff/flores200", "eng_Latn"),
        ]

        for path, config in paths_to_try:
            try:
                data = load_dataset(path, config, split="devtest")
                sentences = [row["sentence"] for row in data][:n_sentences]
                logger.info(f"Downloaded {len(sentences)} sentences from {path}")
                return sentences
            except Exception as e:
                logger.warning(f"  {path} failed: {e}")
                continue

        raise RuntimeError("All FLORES-200 paths failed")

    except ImportError:
        raise RuntimeError("datasets library not installed. Run: pip install datasets")


def extract_csi_from_sentences_llm(
    sentences: List[str],
    batch_size: int = 10,
    sleep_between_batches: float = 1.0,
) -> Dict[str, List[dict]]:
    """
    Use Gemini to identify CSI terms in each sentence.
    Returns a dict mapping category -> list of {term, count} dicts.
    """
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No GEMINI_API_KEY — skipping LLM extraction. Using seed terms only.")
        return defaultdict(list)

    import google.generativeai as genai
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    all_found: Dict[str, Counter] = defaultdict(Counter)
    total_with_csi = 0

    for batch_start in range(0, len(sentences), batch_size):
        batch = sentences[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(sentences) + batch_size - 1) // batch_size
        logger.info(f"Processing batch {batch_num}/{total_batches}...")

        # Send all sentences in batch together for efficiency
        sentences_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))

        prompt = (
            f"You are a cultural linguistics expert. For each sentence below, "
            f"identify ALL culture-specific items (CSIs) — things that are "
            f"culturally specific and cannot be translated literally:\n\n"
            f"Include: culturally specific foods, idioms that don't translate, "
            f"cultural concepts (karma, hygge, etc.), superstitions, humour, "
            f"proverbs, holidays, institutional terms, media references.\n\n"
            f"{sentences_text}\n\n"
            f"Return a JSON array of ALL unique CSI terms found across all sentences:\n"
            f'[{{"term": "exact text", "category": "culturally_embedded|proper_name|institutional|pragmatic", "explanation": "why it is culturally specific"}}]\n\n'
            f"Only include genuinely culture-specific items. Return ONLY the JSON array."
        )

        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()

            if raw.startswith("["):
                items = json.loads(raw)
            else:
                import re
                match = re.search(r"\[[\s\S]*\]", raw)
                items = json.loads(match.group(0)) if match else []

            batch_terms = 0
            for item in items:
                term = item.get("term", "").lower().strip()
                cat  = item.get("category", "culturally_embedded")
                if term and cat in SEED_TERMS:
                    all_found[cat][term] += 1
                    batch_terms += 1

            if batch_terms > 0:
                total_with_csi += 1
                logger.info(f"  Batch {batch_num}: found {batch_terms} CSI terms")

        except Exception as e:
            logger.warning(f"  Batch {batch_num} failed: {e}")

        if batch_num < total_batches:
            time.sleep(sleep_between_batches)

    return all_found


def build_csi_terms_json(
    sentences: List[str],
    llm_found: Dict[str, Counter],
    min_count: int = 1,
) -> dict:
    """
    Merge seed terms with LLM-discovered terms.
    Terms found by LLM in multiple sentences get higher priority.
    """
    result = {}

    for category in SEED_TERMS:
        # Start with seed terms
        terms_set = set(SEED_TERMS[category])

        # Add LLM-discovered terms (if found in at least min_count sentences)
        for term, count in llm_found.get(category, {}).items():
            if count >= min_count:
                terms_set.add(term)

        result[category] = sorted(terms_set)

    return result


def compute_stats(sentences: List[str], csi_terms: dict) -> dict:
    """
    Compute CSI prevalence statistics for the paper.
    Loads the csi_detector using the new terms and runs it on all sentences.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    # Temporarily write csi_terms.json so csi_detector can load it
    terms_path = os.path.join(os.path.dirname(__file__), "csi_terms.json")
    with open(terms_path, "w", encoding="utf-8") as f:
        json.dump(csi_terms, f, ensure_ascii=False, indent=2)

    from pipeline.csi_detector import detect_csi_spans

    sentences_with_csi = 0
    total_spans = 0
    cat_counts = Counter()
    examples = []

    for sentence in sentences:
        # Mock a minimal PipelineInput
        pipeline_input = {
            "text":        sentence,
            "source_lang": "en",
            "target_lang": "ja",
            "modality":    "text",
            "confidence":  1.0,
        }
        spans = detect_csi_spans(pipeline_input)
        if spans:
            sentences_with_csi += 1
            total_spans += len(spans)
            for s in spans:
                cat_counts[s["category"]] += 1
            if len(examples) < 5:
                examples.append({
                    "sentence": sentence,
                    "csi":      [{"span": s["span"], "cat": s["category"]} for s in spans],
                })

    stats = {
        "corpus":             "FLORES-200 English devtest",
        "sentences_analysed": len(sentences),
        "sentences_with_csi": sentences_with_csi,
        "csi_prevalence_pct": round(sentences_with_csi / len(sentences) * 100, 1),
        "total_spans":        total_spans,
        "avg_spans_per_sentence": round(total_spans / len(sentences), 2),
        "by_category":        dict(cat_counts),
        "examples":           examples,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build CSI term lists from FLORES-200 corpus"
    )
    parser.add_argument(
        "--sentences", type=int, default=200,
        help="Number of FLORES-200 sentences to process (default: 200, max: 1012)"
    )
    parser.add_argument(
        "--seed-only", action="store_true",
        help="Skip LLM extraction, only use seed terms (no API calls)"
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    terms_path = os.path.join(output_dir, "csi_terms.json")
    stats_path = os.path.join(output_dir, "flores_csi_stats.json")

    logger.info("=" * 60)
    logger.info("Building CSI term lists from FLORES-200")
    logger.info("=" * 60)

    # Step 1: Download FLORES-200
    try:
        sentences = download_flores200(args.sentences)
    except RuntimeError as e:
        logger.warning(f"Could not download FLORES-200: {e}")
        logger.info("Using seed terms only (will still produce valid csi_terms.json)")
        sentences = []

    # Step 2: Extract CSI from sentences using LLM
    llm_found: Dict[str, Counter] = defaultdict(Counter)
    if sentences and not args.seed_only:
        llm_found = extract_csi_from_sentences_llm(sentences)
        total_llm = sum(len(v) for v in llm_found.values())
        logger.info(f"LLM extraction complete: {total_llm} unique terms found")
    else:
        logger.info("Skipping LLM extraction (using seed terms only)")

    # Step 3: Build merged term list
    csi_terms = build_csi_terms_json(sentences, llm_found)

    # Step 4: Save csi_terms.json
    with open(terms_path, "w", encoding="utf-8") as f:
        json.dump(csi_terms, f, ensure_ascii=False, indent=2)

    total_terms = sum(len(v) for v in csi_terms.values())
    logger.info(f"\nSaved {total_terms} CSI terms to {terms_path}")
    for cat, terms in csi_terms.items():
        logger.info(f"  {cat}: {len(terms)} terms")

    # Step 5: Compute stats
    if sentences:
        logger.info("\nComputing CSI prevalence statistics on FLORES-200...")
        stats = compute_stats(sentences, csi_terms)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("PAPER STATISTICS (cite these in your paper)")
        logger.info("=" * 60)
        logger.info(f"Corpus:          {stats['corpus']}")
        logger.info(f"Sentences:       {stats['sentences_analysed']}")
        logger.info(f"With CSI:        {stats['sentences_with_csi']} ({stats['csi_prevalence_pct']}%)")
        logger.info(f"Total spans:     {stats['total_spans']}")
        logger.info(f"Avg per sentence:{stats['avg_spans_per_sentence']}")
        logger.info(f"By category:     {stats['by_category']}")
        logger.info(f"\nSample CSI detections:")
        for ex in stats["examples"][:3]:
            logger.info(f"  '{ex['sentence'][:60]}...'")
            for c in ex["csi"]:
                logger.info(f"    [{c['cat']}] '{c['span']}'")
        logger.info(f"\nStats saved to {stats_path}")

    logger.info("\nDone. csi_detector.py will now load from csi_terms.json.")


if __name__ == "__main__":
    main()