"""
interfaces.py
-------------
Shared data contracts (TypedDicts) for all handoff points in the pipeline.
All three team members import from this file so integration is seamless.

HANDOFF SUMMARY:
  A: SpeechInput/TextInput  → produced by Anshul, consumed by Anvita
  B: CulturalContextObject  → produced by Asmi + Anvita, consumed by Anvita (prompt builder)
  C: TranslationObject      → produced by Anvita, consumed by Asmi (explainability) + Anshul (TTS)
  D: LearningLayerInput     → produced by Anvita, consumed by Anshul (learning layer)
"""

from typing import TypedDict, List, Optional


# ─────────────────────────────────────────────
# HANDOFF A  (Anshul → Anvita)
# ─────────────────────────────────────────────

class PipelineInput(TypedDict):
    """
    Raw input after text preprocessing or ASR.
    Produced by: text_preprocessor.py (Anvita) or Anshul's speech module.
    Consumed by: Cultural Intelligence layer (Asmi + Anvita).
    """
    text: str               # cleaned source text
    source_lang: str        # ISO 639-1 code e.g. "en", "ja", "hi"
    target_lang: str        # ISO 639-1 code e.g. "fr", "ko", "ar"
    modality: str           # "text" or "speech"
    confidence: float       # language detection confidence 0-1


# ─────────────────────────────────────────────
# HANDOFF B  (Asmi + Anvita → LLM Engine)
# ─────────────────────────────────────────────

class CSISpan(TypedDict):
    """A single culture-specific item detected in the source text."""
    span: str               # the actual text span
    start: int              # character offset start
    end: int                # character offset end
    category: str           # "proper_name" | "culturally_embedded" | "institutional" | "pragmatic"
    explanation: str        # plain English explanation of why this is culturally significant


class SensitivityFlag(TypedDict):
    """A culturally sensitive or potentially offensive span."""
    span: str
    warning_type: str       # e.g. "taboo", "offensive_in_target", "religious", "political"
    severity: str           # "low" | "medium" | "high"
    suggestion: str         # recommended alternative phrasing


class SessionHistory(TypedDict):
    """Carries formality and cultural preferences across dialogue turns."""
    preferred_formality: Optional[str]      # "formal" | "casual" | None (not yet determined)
    csi_categories_seen: List[str]          # accumulated list of CSI categories in this session
    active_warnings: List[SensitivityFlag]  # warnings from current and previous turns
    turn_count: int


class CulturalContextObject(TypedDict):
    """
    HANDOFF B — the full cultural intelligence package fed into the prompt builder.
    Produced by: Asmi's CSI/sensitivity modules + Anvita's formality classifier.
    Consumed by: Anvita's prompt_builder.py.
    """
    text: str
    source_lang: str
    target_lang: str
    modality: str
    csi_spans: List[CSISpan]
    source_formality: str           # "formal" | "neutral" | "casual"
    formality_confidence: float     # classifier confidence 0-1
    session_history: SessionHistory
    sensitivity_flags: List[SensitivityFlag]


# ─────────────────────────────────────────────
# HANDOFF C  (Anvita → Asmi + Anshul)
# ─────────────────────────────────────────────

class TranslationObject(TypedDict):
    """
    HANDOFF C — the full translation result from the LLM engine.
    Produced by: Anvita's llm_engine.py.
    Consumed by: Asmi (explainability UI, attention heatmap) + Anshul (TTS, learning layer).
    """
    formal: str                     # formal register translation
    casual: str                     # casual register translation
    literal: str                    # word-by-word breakdown in target language with meanings
    cultural_notes: str             # plain English cultural adaptation notes
    confidence: float               # overall translation confidence 0-1
    cot_reasoning: str              # full chain-of-thought reasoning trace
    source_lang: str
    target_lang: str
    source_text: str
    csi_spans: List[CSISpan]        # passed through from CulturalContextObject
    sensitivity_flags: List[SensitivityFlag]
    detected_formality: str         # detected source register: "formal"|"neutral"|"casual"
    tone_recommendation_reason: str # why this register is used
    formal_pronunciation: str       # romanized pronunciation of formal translation
    casual_pronunciation: str       # romanized pronunciation of casual translation
    literal_pronunciation: str      # pronunciation of the full literal translation
    is_gendered: bool               # True if translation differs by gender
    formal_male: str                # male formal variant (if is_gendered)
    formal_female: str              # female formal variant (if is_gendered)
    casual_male: str                # male casual variant (if is_gendered)
    casual_female: str              # female casual variant (if is_gendered)
    formal_male_pronunciation: str
    formal_female_pronunciation: str
    casual_male_pronunciation: str
    casual_female_pronunciation: str
    gender_note: str                # why gender affects this translation
    word_breakdown: List[dict]
    flashcards: List[dict]
    cefr_level: str
    model_used: str


# ─────────────────────────────────────────────
# HANDOFF D  (Anvita → Anshul)
# ─────────────────────────────────────────────

class LearningLayerInput(TypedDict):
    """
    HANDOFF D — input for Anshul's learning layer modules.
    Produced by: Anvita's pipeline after translation.
    Consumed by: Anshul's word_breakdown, flashcard_gen, difficulty_gauge modules.
    """
    best_translation: str           # the formal variant by default
    target_lang: str
    source_text: str
    source_lang: str
    user_cefr_level: str            # "A1" | "A2" | "B1" | "B2" | "C1" | "C2"
    csi_spans: List[CSISpan]
    cot_reasoning: str
