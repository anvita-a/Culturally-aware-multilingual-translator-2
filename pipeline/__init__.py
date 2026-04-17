"""
pipeline package

Public API — import from here in your code:

    from pipeline import run
    from pipeline.interfaces import TranslationObject, LearningLayerInput

Quick usage:
    from pipeline import run
    translation, learning = run("Hello, how are you?", target_lang="ja")
    print(translation["formal"])
"""

from pipeline.variant_formatter import run, run_with_cefr_adjustment
from pipeline.interfaces import (
    PipelineInput,
    CulturalContextObject,
    TranslationObject,
    LearningLayerInput,
    CSISpan,
    SensitivityFlag,
    SessionHistory,
)

__all__ = [
    "run",
    "run_with_cefr_adjustment",
    "PipelineInput",
    "CulturalContextObject",
    "TranslationObject",
    "LearningLayerInput",
    "CSISpan",
    "SensitivityFlag",
    "SessionHistory",
]