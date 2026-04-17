"""
text_preprocessor.py
--------------------
Anvita's Module 1 — Text Preprocessing

Responsibilities:
  - Unicode normalization
  - Whitespace and punctuation cleaning
  - Language detection (text modality only)
  - Produces a clean PipelineInput object (Handoff A format)

Why these choices:
  - unicodedata.normalize('NFC') standardises composed vs decomposed characters
    (critical for Arabic, Hindi, Korean where the same glyph can have multiple
    Unicode representations — inconsistency breaks tokenization downstream)
  - langdetect gives a probability distribution over languages; we take the top
    result but store confidence so downstream modules can flag low-confidence inputs
  - We do NOT do aggressive cleaning (removing punctuation, lowercasing) because
    punctuation and capitalisation carry formality signals that the formality
    classifier needs
"""

import unicodedata
import re
import logging
from typing import Optional
from pipeline.interfaces import PipelineInput

logger = logging.getLogger(__name__)

# Supported language codes for our 10 target languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "ja": "Japanese",
    "hi": "Hindi",
    "ar": "Arabic",
    "zh": "Mandarin Chinese",
    "de": "German",
    "sw": "Swahili",
    "pt": "Portuguese (Brazilian)",
    "ko": "Korean",
}


def detect_language(text: str) -> tuple[str, float]:
    """
    Detect the language of input text.
    Returns (iso_code, confidence).

    Uses langdetect with a fallback to 'en' on failure.
    In production (Anshul's speech pipeline), this is replaced by the
    Whisper + fastText dual-verification system.
    """
    try:
        from langdetect import detect_langs
        results = detect_langs(text)
        if results:
            top = results[0]
            lang = top.lang
            confidence = top.prob
            # langdetect uses zh-cn / zh-tw — normalise to zh
            if lang.startswith("zh"):
                lang = "zh"
            return lang, round(confidence, 3)
    except Exception as e:
        logger.warning(f"Language detection failed: {e}. Defaulting to 'en'.")
    return "en", 0.5


def normalize_unicode(text: str) -> str:
    """
    Apply NFC Unicode normalization.
    Critical for Arabic (hamza variants), Hindi (vowel marks),
    and Korean (precomposed vs decomposed Hangul syllables).
    """
    return unicodedata.normalize("NFC", text)


def clean_whitespace(text: str) -> str:
    """
    Collapse multiple spaces/tabs/newlines into a single space.
    Strip leading/trailing whitespace.
    Preserve single newlines as sentence boundaries.
    """
    # Replace multiple whitespace (not newlines) with single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse multiple newlines into one
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def remove_control_characters(text: str) -> str:
    """
    Remove non-printable control characters (except tab and newline).
    These can cause issues in API calls and tokenization.
    """
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cc", "Cf")
        or ch in ("\t", "\n")
    )


def preprocess_text(
    text: str,
    target_lang: str,
    source_lang: Optional[str] = None,
    modality: str = "text",
) -> PipelineInput:
    """
    Main preprocessing function. Takes raw user input and returns
    a clean PipelineInput object ready for the Cultural Intelligence layer.

    Args:
        text:        Raw input string from the user
        target_lang: ISO 639-1 code for desired output language
        source_lang: ISO 639-1 code if known; auto-detected if None
        modality:    "text" (default) or "speech" (set by Anshul's module)

    Returns:
        PipelineInput TypedDict (Handoff A format)

    Example:
        >>> result = preprocess_text("Please send the report ASAP!!", "ja")
        >>> result["text"]
        'Please send the report ASAP!!'
        >>> result["source_lang"]
        'en'
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    if target_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Target language '{target_lang}' not supported. "
            f"Choose from: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    # Step 1: Remove control characters
    text = remove_control_characters(text)

    # Step 2: Unicode normalization
    text = normalize_unicode(text)

    # Step 3: Clean whitespace
    text = clean_whitespace(text)

    # Step 4: Language detection (if not provided)
    if source_lang is None:
        detected_lang, confidence = detect_language(text)
        source_lang = detected_lang
    else:
        _, confidence = detect_language(text)

    # Step 5: Validate detected/provided source language
    # If confidence is very low, we still proceed but log a warning
    if confidence < 0.5:
        logger.warning(
            f"Low language detection confidence ({confidence}) for text: '{text[:50]}...'"
        )

    result: PipelineInput = {
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "modality": modality,
        "confidence": confidence,
    }

    logger.info(
        f"Preprocessed: lang={source_lang}→{target_lang}, "
        f"confidence={confidence}, modality={modality}, "
        f"length={len(text)} chars"
    )

    return result


# ─── Stub for Asmi's cultural intelligence modules ───────────────────────────
# These will be replaced by Asmi's actual implementations.
# Anvita's pipeline uses these stubs so the full pipeline runs
# even before Week 2 integration.

def stub_csi_detection(pipeline_input: PipelineInput) -> list:
    """Stub: returns empty CSI spans. Replace with Asmi's mDeBERTa module."""
    logger.debug("Using CSI detection stub — replace with Asmi's module")
    return []


def stub_sensitivity_flags(pipeline_input: PipelineInput) -> list:
    """Stub: returns no flags. Replace with Asmi's sensitivity flagging module."""
    logger.debug("Using sensitivity flagging stub — replace with Asmi's module")
    return []


def stub_session_history() -> dict:
    """Stub: returns a fresh session. Replace with Asmi's Context Memory module."""
    return {
        "preferred_formality": None,
        "csi_categories_seen": [],
        "active_warnings": [],
        "turn_count": 0,
    }