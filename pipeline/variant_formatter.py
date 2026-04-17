"""
variant_formatter.py
--------------------
Anvita's Module 5 — Pipeline Orchestrator (updated to use Asmi's real modules)

All modules are now wired together. The stubs have been replaced by
the real implementations from Asmi's three files.
"""

import logging
from typing import Optional
from pipeline.interfaces import (
    PipelineInput,
    CulturalContextObject,
    TranslationObject,
    LearningLayerInput,
    SessionHistory,
)
from pipeline.text_preprocessor import preprocess_text
from pipeline.formality_classifier import classify_formality
from pipeline.llm_engine import translate, translate_with_cefr_adjustment

# Asmi's real modules — replacing the stubs
from pipeline.csi_detector import detect_csi_spans
from pipeline.sensitivity_flagger import flag_sensitivity
from pipeline.context_memory import create_session, update_session
from pipeline.profanity_flagger import detect_profanity

logger = logging.getLogger(__name__)


def _post_process_variant(text: str, target_lang: str) -> str:
    text = text.strip()
    text = text.strip('"').strip("'")
    text = text.replace("\\n", " ").replace('\\"', '"')
    return text


def run(
    text: str,
    target_lang: str,
    source_lang: Optional[str] = None,
    user_cefr_level: str = "B2",
    session_history: Optional[dict] = None,
    modality: str = "text",
    # These parameters are kept for backward compatibility and testing.
    # Pass custom functions to override the defaults during unit tests.
    csi_detection_fn=None,
    sensitivity_fn=None,
) -> tuple[TranslationObject, LearningLayerInput]:
    """
    Main pipeline entry point. Calls every module in order.

    Args:
        text:              Raw input text
        target_lang:       ISO 639-1 target language code
        source_lang:       ISO 639-1 source language (auto-detected if None)
        user_cefr_level:   User's CEFR level ("A1"–"C2")
        session_history:   Dict from previous turn (None = new session)
        modality:          "text" or "speech"
        csi_detection_fn:  Override for testing (uses detect_csi_spans by default)
        sensitivity_fn:    Override for testing (uses flag_sensitivity by default)

    Returns:
        (TranslationObject, LearningLayerInput)
    """

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    pipeline_input: PipelineInput = preprocess_text(
        text=text,
        target_lang=target_lang,
        source_lang=source_lang,
        modality=modality,
    )

    # ── Step 2: CSI Detection + Sensitivity Flagging (Asmi's modules) ─────────
    _csi_fn  = csi_detection_fn or detect_csi_spans
    _sens_fn = sensitivity_fn   or flag_sensitivity

    csi_spans         = _csi_fn(pipeline_input)
    sensitivity_flags = _sens_fn(pipeline_input)

    # ── Profanity / explicit content detection ────────────────────────────────
    profanity_flag = detect_profanity(pipeline_input["text"], source_lang=pipeline_input["source_lang"])

    # ── Step 3: Formality Analysis ────────────────────────────────────────────
    formality_label, formality_confidence = classify_formality(
        pipeline_input["text"],
        pipeline_input["source_lang"],
    )

    # ── Step 4: Session History (Asmi's context memory) ───────────────────────
    if session_history is None:
        session_history = create_session()

    session_history = update_session(
        session=session_history,
        detected_formality=formality_label,
        csi_spans=csi_spans,
        sensitivity_flags=sensitivity_flags,
    )

    # ── Step 5: Build CulturalContextObject ───────────────────────────────────
    context: CulturalContextObject = {
        "text":                 pipeline_input["text"],
        "source_lang":          pipeline_input["source_lang"],
        "target_lang":          pipeline_input["target_lang"],
        "modality":             pipeline_input["modality"],
        "csi_spans":            csi_spans,
        "source_formality":     formality_label,
        "formality_confidence": formality_confidence,
        "session_history":      session_history,
        "sensitivity_flags":    sensitivity_flags,
    }

    # ── Step 6: Translate ─────────────────────────────────────────────────────
    translation, learning_input = translate(context, user_cefr_level=user_cefr_level)

    # ── Step 7: Post-process ──────────────────────────────────────────────────
    for variant in ("formal", "casual", "literal"):
        translation[variant] = _post_process_variant(
            translation[variant], target_lang
        )

    # Attach profanity flag to translation object for the UI to display
    translation["profanity_flag"] = profanity_flag

    logger.info("Pipeline run complete.")
    return translation, learning_input


def run_with_cefr_adjustment(
    text: str,
    target_lang: str,
    detected_cefr_level: str,
    user_declared_level: str,
    existing_translation: TranslationObject,
    context: CulturalContextObject,
) -> TranslationObject:
    CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
    detected_idx = CEFR_ORDER.index(detected_cefr_level) if detected_cefr_level in CEFR_ORDER else 3
    declared_idx = CEFR_ORDER.index(user_declared_level)  if user_declared_level  in CEFR_ORDER else 3
    if detected_idx > declared_idx:
        return translate_with_cefr_adjustment(context, existing_translation, user_declared_level)
    return existing_translation