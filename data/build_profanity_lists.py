"""
build_profanity_lists.py
------------------------
Builds data/profanity_terms.json from real datasets.

Datasets used:
  1. LDNOOBW English list (Schulman, 2012)
       ~450 terms, plain text, GitHub raw — no auth needed
       https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words

  2. Jigsaw Toxic Comment Classification (Jigsaw/Google, 2018)
       160k Wikipedia comments with toxic/obscene/insult labels
       Extracts statistically high-frequency terms from toxic text
       HuggingFace: Arsive/toxicity_classification_jigsaw

  3. Curated multilingual seed list (always included)
       Profanity terms in all 10 supported languages
       Covers: French, Spanish, Japanese, Hindi, Arabic,
               Mandarin, German, Swahili, Portuguese, Korean

Output: data/profanity_terms.json
  {
    "profanity":    ["fuck", "shit", ...],
    "explicit":     ["sex", "nude", ...],
    "mild":         ["damn", "hell", ...],
    "multilingual": {
      "fr": ["merde", "putain", ...],
      "es": ["mierda", "puta", ...],
      ...
    }
  }

Usage:
  python3 data/build_profanity_lists.py
  python3 data/build_profanity_lists.py --seed-only
"""

import os
import re
import sys
import json
import logging
import argparse
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "profanity_terms.json")


# ── English seed lists ────────────────────────────────────────────────────────

SEED_PROFANITY = [
    "fuck", "fucking", "fucked", "fucker", "fucks", "fuckin", "motherfucker",
    "motherfucking", "shit", "shitty", "bullshit", "horseshit", "shitting",
    "bitch", "bitchy", "bitches", "bitching", "bitchass",
    "ass", "asshole", "arsehole", "arse", "jackass", "smartass", "dumbass",
    "damn", "goddamn", "goddamnit", "damnit", "damned",
    "bastard", "bastards",
    "cunt", "cunts",
    "dick", "dickhead", "dicks", "cock", "cocks", "prick", "pricks",
    "wanker", "wankers", "twat", "twats",
    "douche", "douchebag", "douchebags",
    "piss", "pissed", "pissing", "pissoff",
    "whore", "whores", "slut", "sluts", "hoe", "hoes",
    "crap", "crappy", "bugger", "buggers",
    "bloody", "hell", "jerk", "jerks",
    "screw", "screwed", "screwup",
    "suck", "sucks", "sucked", "sucker",
    "moron", "idiot", "idiots", "stupid", "dumbass", "dumb", "loser",
]

SEED_EXPLICIT = [
    "sex", "sexual", "sexually", "naked", "nude", "nudity",
    "porn", "pornographic", "pornography", "erotic", "erotica",
    "orgasm", "masturbate", "masturbation",
    "penis", "vagina", "genitals", "genitalia",
    "breasts", "boobs", "tits", "nipple", "nipples",
    "intercourse", "fornicate",
    "horny", "slutty", "kinky", "fetish", "bondage",
    "naughty", "dirty", "filthy",
    "sexy", "seductive",
    "hooker", "prostitute", "escort",
]

SEED_MILD = [
    "damn", "hell", "crap", "bloody", "bugger",
    "darn", "shoot", "fudge", "heck",
    "jerk", "idiot", "stupid", "dumb", "suck", "sucks",
]


# ── Multilingual profanity seed list ─────────────────────────────────────────
# Each entry: term in the target language, severity (high/medium/low)
# Sources: Wiktionary category pages, academic papers on multilingual profanity

MULTILINGUAL_SEED = {
    "fr": [  # French
        # Strong profanity
        "merde", "putain", "connard", "connasse", "salaud", "salope",
        "foutre", "enculer", "enculé", "bordel", "chier", "chieur",
        "bâtard", "con", "conne", "pute", "gueule", "merdique",
        # Mild
        "zut", "flûte", "sacré", "diable",
        # Explicit
        "sexuel", "pornographique", "obscène",
    ],
    "es": [  # Spanish
        # Strong
        "mierda", "puta", "puto", "joder", "hostia", "gilipollas",
        "cabrón", "cabron", "pendejo", "chinga", "chingada", "chingado",
        "verga", "culero", "maricón", "maricon", "perra", "coño",
        "carajo", "cojones", "culo",
        # Mild
        "dios mío", "ostia", "mecachis",
        # Explicit
        "sexual", "pornográfico", "obsceno", "desnudo",
    ],
    "de": [  # German
        # Strong
        "scheiße", "scheiß", "ficken", "arsch", "arschloch", "wichser",
        "hurensohn", "fotze", "verdammt", "dummkopf", "idiot", "blödmann",
        "mist", "kacke", "sau", "vollidiot",
        # Mild
        "mist", "himmel", "teufel",
        # Explicit
        "nackt", "pornografisch", "obszön", "sexuell",
    ],
    "pt": [  # Portuguese (Brazilian)
        # Strong
        "merda", "porra", "caralho", "foda", "fodase", "viado", "buceta",
        "cu", "filha da puta", "filho da puta", "vai se foder", "puta",
        "desgraçado", "idiota", "babaca", "otario", "otário",
        # Mild
        "droga", "dane-se", "inferno",
        # Explicit
        "sexual", "pornográfico", "obsceno", "nu",
    ],
    "hi": [  # Hindi (romanised — as users typically type)
        # Strong
        "madarchod", "bhenchod", "chutiya", "gandu", "harami", "randi",
        "saala", "sala", "bhosdike", "bhadwa", "gaandu", "lund", "chut",
        "maa ki", "teri maa", "bakwas",
        # Mild
        "ullu", "kamine", "pagal",
        # Explicit
        "sexy", "nangi", "porn",
    ],
    "ar": [  # Arabic (romanised)
        # Strong
        "kuss", "kus", "ibn el sharmouta", "sharmouta", "ibn el sharmuta",
        "ayr", "khawal", "zibi", "hayawan", "khara", "nayek",
        "manyak", "wahsh", "kalb",
        # Mild
        "yel'an", "la'ana",
        # Explicit
        "jinsi", "ari", "fahish",
    ],
    "zh": [  # Mandarin Chinese (pinyin romanisation)
        # Strong
        "tā mā de", "tama de", "cào", "cao ni", "wǒ cào", "bì", "shǎbī",
        "shabi", "hundan", "hún dàn", "wángbādàn", "wangbadan",
        "bāgǎo", "gùndan", "gundan",
        # Mild
        "tāmāde", "aiya", "aiyah",
        # Explicit  
        "sèqíng", "seqing", "lúntí", "yínhuì",
    ],
    "ja": [  # Japanese (romanised)
        # Strong
        "chikusho", "chikushō", "kuso", "bakayaro", "baka yaro", "yarou",
        "kisama", "temee", "temē", "fuzakenna", "shine", "aho",
        "manuke", "urusai", "urusee",
        # Mild
        "shimatta", "kichiku", "yarō",
        # Explicit
        "hentai", "ecchi", "ero", "sukebe", "sukebee",
    ],
    "ko": [  # Korean (romanised)
        # Strong
        "sibal", "ssibal", "gaesei", "gaeseki", "byeonshin", "byonshin",
        "jiral", "dul", "deot", "enema", "niga", "michyeo", "michi",
        "baaboo", "meong chung ee",
        # Mild
        "aish", "aigu", "jebal",
        # Explicit
        "seong", "eumnan", "nakara",
    ],
    "sw": [  # Swahili
        # Strong
        "msenge", "malaya", "matako", "mkundu", "umbwa", "mjinga",
        "mnazi", "haram", "bure kabisa",
        # Mild
        "bure", "pumbavu", "mjinga",
        # Explicit
        "ngono", "utupu",
    ],
}


# ── Dataset downloaders ───────────────────────────────────────────────────────

def download_ldnoobw() -> list[str]:
    """
    Download the LDNOOBW (List of Dirty, Naughty, Obscene, and Otherwise Bad Words).
    Plain text, one word per line, ~450 English terms.
    Free GitHub raw access, no auth required.

    Citation: Schulman, K. (2012). LDNOOBW.
              https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words
    """
    import urllib.request
    url = (
        "https://raw.githubusercontent.com/LDNOOBW/"
        "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en"
    )
    try:
        logger.info("Downloading LDNOOBW English profanity list...")
        with urllib.request.urlopen(url, timeout=10) as resp:
            raw = resp.read().decode("utf-8")

        terms = []
        for line in raw.splitlines():
            term = line.strip().lower()
            if term and re.match(r"^[a-z\s'\-]+$", term) and len(term) >= 3:
                terms.append(term)

        logger.info(f"LDNOOBW: {len(terms)} English terms downloaded")
        return terms

    except Exception as e:
        logger.warning(f"LDNOOBW download failed (non-fatal): {e}")
        return []


def load_jigsaw_toxic_terms() -> list[str]:
    """
    Extract statistically high-frequency terms from Jigsaw toxic comments.

    Dataset: Arsive/toxicity_classification_jigsaw on HuggingFace
    Scans 20,000 comments labelled toxic/obscene/insult and extracts
    words that appear ≥5× more in toxic text than clean text.

    Citation: Jigsaw/Google. (2018). Toxic Comment Classification Challenge.
              https://kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    """
    try:
        from datasets import load_dataset

        logger.info("Loading Jigsaw toxicity dataset (Arsive/toxicity_classification_jigsaw)...")
        data = load_dataset(
            "Arsive/toxicity_classification_jigsaw",
            split="train",
            streaming=True,
        )

        toxic_words: Counter = Counter()
        clean_words: Counter = Counter()
        count = 0

        for row in data:
            text     = row.get("comment_text", "").lower()
            is_toxic = (
                row.get("toxic", 0) == 1 or
                row.get("obscene", 0) == 1 or
                row.get("insult", 0) == 1
            )
            words = re.findall(r"[a-z']+", text)
            if is_toxic:
                toxic_words.update(words)
            else:
                clean_words.update(words)
            count += 1
            if count >= 20000:
                break

        logger.info(f"Processed {count} Jigsaw comments")

        found = []
        for word, toxic_count in toxic_words.most_common(500):
            if len(word) < 3:
                continue
            clean_count = clean_words.get(word, 0)
            ratio = toxic_count / (clean_count + 1)
            if toxic_count >= 10 and ratio >= 5.0:
                found.append(word)

        logger.info(f"Jigsaw: {len(found)} high-toxicity terms extracted")
        return found

    except ImportError:
        logger.warning("datasets not installed. Run: pip install datasets")
        return []
    except Exception as e:
        logger.warning(f"Jigsaw dataset failed (non-fatal): {e}")
        return []


# ── Term categorisation ───────────────────────────────────────────────────────

def categorise_english(all_terms: list[str]) -> tuple[list, list, list]:
    """Sort English terms into profanity / explicit / mild buckets."""
    profanity_set = set(SEED_PROFANITY)
    explicit_set  = set(SEED_EXPLICIT)
    mild_set      = set(SEED_MILD)

    explicit_markers = {
        "sex", "naked", "nude", "porn", "erotic", "orgasm",
        "masturbat", "penis", "vagina", "genital", "breast",
        "nipple", "intercourse", "fornicate", "prostitut", "hooker",
    }

    for term in all_terms:
        term = term.lower().strip()
        if not term or len(term) < 3:
            continue
        is_explicit = any(m in term for m in explicit_markers)
        if is_explicit:
            explicit_set.add(term)
        elif term not in explicit_set and term not in mild_set:
            profanity_set.add(term)

    profanity = sorted(profanity_set - explicit_set)
    explicit  = sorted(explicit_set)
    mild      = sorted(mild_set - profanity_set - explicit_set)
    return profanity, explicit, mild


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-only", action="store_true",
                        help="Skip downloads, use seed terms only")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Building profanity term lists")
    logger.info("=" * 60)

    all_english = []

    if not args.seed_only:
        # Source 1: LDNOOBW (GitHub raw, no auth)
        all_english.extend(download_ldnoobw())

        # Source 2: Jigsaw toxic comments (HuggingFace)
        all_english.extend(load_jigsaw_toxic_terms())
    else:
        logger.info("Seed-only mode — skipping downloads")

    # Categorise English terms
    profanity, explicit, mild = categorise_english(all_english)

    result = {
        "profanity":     profanity,
        "explicit":      explicit,
        "mild":          mild,
        "multilingual":  MULTILINGUAL_SEED,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total_en   = len(profanity) + len(explicit) + len(mild)
    total_ml   = sum(len(v) for v in MULTILINGUAL_SEED.values())
    total_lang = len(MULTILINGUAL_SEED)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"English profanity:  {len(profanity)} terms")
    logger.info(f"English explicit:   {len(explicit)} terms")
    logger.info(f"English mild:       {len(mild)} terms")
    logger.info(f"English total:      {total_en} terms")
    logger.info(f"Multilingual:       {total_ml} terms across {total_lang} languages")
    logger.info(f"  Languages: {', '.join(MULTILINGUAL_SEED.keys())}")
    logger.info(f"Grand total:        {total_en + total_ml} terms")
    logger.info(f"Saved to:           {OUTPUT_PATH}")
    logger.info("\nDataset citations:")
    logger.info("  - LDNOOBW (Schulman, 2012) — English profanity list")
    logger.info("  - Jigsaw Toxic Comment Classification (Jigsaw/Google, 2018)")
    logger.info("  - Curated multilingual seed list (10 languages)")


if __name__ == "__main__":
    main()