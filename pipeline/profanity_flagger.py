"""
profanity_flagger.py
--------------------
Detects profanity, offensive language, and sexually explicit/suggestive content
in source text, regardless of translation direction and source language.

DATASET SOURCE:
  Term lists loaded from data/profanity_terms.json, built by:
    python3 data/build_profanity_lists.py

  English terms from:
    - LDNOOBW English profanity list (Schulman, 2012) ~450 terms
    - Jigsaw Toxic Comment Classification (Jigsaw/Google, 2018) 114 extracted terms

  Multilingual terms: curated romanised forms for all 10 supported languages
    French, Spanish, German, Portuguese, Hindi, Arabic, Mandarin,
    Japanese, Korean, Swahili

Covers both directions:
  English → any language   (English source text scanned)
  Any language → English   (romanised forms of non-English profanity scanned)
"""

import re
import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PROFANITY_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "profanity_terms.json"
)

_terms_cache: Optional[dict] = None


def _load_terms() -> dict:
    """Load term lists from data/profanity_terms.json."""
    global _terms_cache
    if _terms_cache is not None:
        return _terms_cache

    if os.path.exists(_PROFANITY_JSON_PATH):
        try:
            with open(_PROFANITY_JSON_PATH, encoding="utf-8") as f:
                _terms_cache = json.load(f)
            en_total = sum(
                len(_terms_cache.get(k, []))
                for k in ("profanity", "explicit", "mild")
            )
            ml_total = sum(
                len(v) for v in _terms_cache.get("multilingual", {}).values()
            )
            logger.info(
                f"Profanity terms loaded: {en_total} English + "
                f"{ml_total} multilingual"
            )
            return _terms_cache
        except Exception as e:
            logger.error(f"Failed to load profanity_terms.json: {e}")

    # Fallback to seeds from formality_classifier
    logger.warning(
        "profanity_terms.json not found. "
        "Run: python3 data/build_profanity_lists.py"
    )
    try:
        from pipeline.formality_classifier import PROFANITY_TERMS, SEXUALLY_EXPLICIT_TERMS
        _terms_cache = {
            "profanity":    sorted(PROFANITY_TERMS),
            "explicit":     sorted(SEXUALLY_EXPLICIT_TERMS),
            "mild":         ["damn", "hell", "crap", "bloody", "bugger"],
            "multilingual": {},
        }
    except ImportError:
        _terms_cache = {
            "profanity": [], "explicit": [], "mild": [], "multilingual": {}
        }

    return _terms_cache


class ProfanityFlag:
    """Result of profanity/explicit content detection."""

    def __init__(self, found, level, terms_found, message, language="en"):
        self.found       = found
        self.level       = level
        self.terms_found = terms_found
        self.message     = message
        self.language    = language   # which language the terms were found in

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __bool__(self):
        return self.found

    def __getitem__(self, key):
        return getattr(self, key)


def _scan_terms(text_lower: str, words: set, term_list: list) -> list:
    """Check text against a list of terms, handling both single and multi-word."""
    found = []
    for term in term_list:
        if " " in term:
            if term in text_lower:
                found.append(term)
        elif term in words:
            found.append(term)
    return found


def detect_profanity(text: str, source_lang: str = "en") -> ProfanityFlag:
    """
    Detect profanity and explicit content in source text.

    Scans:
      1. English terms (always — catches English profanity in any input)
      2. Language-specific terms for source_lang (if available in multilingual dict)
         These are stored as romanised forms so they match user-typed input

    Args:
        text:        Source text in any language
        source_lang: ISO 639-1 code of the source language (default "en")

    Returns:
        ProfanityFlag — found=False if nothing detected,
        found=True with level, terms_found, message if detected.

    Examples:
        >>> detect_profanity("holy shit that was amazing").found
        True
        >>> detect_profanity("putain c'est incroyable", source_lang="fr").found
        True
        >>> detect_profanity("merde alors", source_lang="fr").level
        'profanity'
        >>> detect_profanity("Please tell them I will be late").found
        False
    """
    terms     = _load_terms()
    text_lower = text.lower()
    # ASCII word tokens (for English terms)
    words_ascii = set(re.findall(r"[a-z']+", text_lower))
    # Unicode word tokens (for terms with accented/special chars like ß, é, ñ)
    words_unicode = set(re.findall(r"[\w']+", text_lower, re.UNICODE))

    # ── Scan English terms (always) ───────────────────────────────────────────
    found_profanity = _scan_terms(text_lower, words_ascii, terms.get("profanity", []))
    found_explicit  = _scan_terms(text_lower, words_ascii, terms.get("explicit",  []))

    # ── Scan source language terms ────────────────────────────────────────────
    lang_found = []
    detected_lang = "en"

    if source_lang != "en":
        ml = terms.get("multilingual", {})
        lang_terms = ml.get(source_lang, [])
        if lang_terms:
            # Use unicode tokens so ß, é, ñ, etc. match correctly
            lang_found = _scan_terms(text_lower, words_unicode, lang_terms)
            if lang_found:
                detected_lang = source_lang
                # All multilingual hits go into profanity bucket for simplicity
                found_profanity.extend(lang_found)

    # ── Deduplicate ───────────────────────────────────────────────────────────
    found_profanity = list(dict.fromkeys(found_profanity))
    found_explicit  = list(dict.fromkeys(found_explicit))

    if not found_profanity and not found_explicit:
        return ProfanityFlag(
            found=False, level="none",
            terms_found=[], message="", language="en"
        )

    if found_profanity and found_explicit:
        level = "both"
    elif found_explicit:
        level = "explicit"
    else:
        level = "profanity"

    all_terms = found_profanity + [t for t in found_explicit if t not in found_profanity]

    messages = {
        "profanity": (
            "This text contains offensive or profane language. "
            "The translation will reflect the same register — "
            "the translated output may also contain strong language."
        ),
        "explicit": (
            "This text contains sexually suggestive or explicit content. "
            "The translation will reflect the same content."
        ),
        "both": (
            "This text contains offensive language and sexually explicit content. "
            "The translation will reflect the same register and content."
        ),
    }

    logger.info(
        f"Profanity detected [{source_lang}]: "
        f"{all_terms[:3]} level={level}"
    )

    return ProfanityFlag(
        found=True,
        level=level,
        terms_found=all_terms,
        message=messages[level],
        language=detected_lang,
    )