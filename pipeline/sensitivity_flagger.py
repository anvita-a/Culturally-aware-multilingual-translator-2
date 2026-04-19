"""
sensitivity_flagger.py
----------------------
Asmi's Module 2 — Sensitivity Flagging

Checks source text for culturally sensitive, offensive, or taboo expressions
relative to the TARGET culture.

Same word can mean very different things across cultures. For example:
  "coger" — neutral in Spain ('to grab'), sexual obscenity in Latin America
  "left hand" — neutral in English, considered unclean in Arab culture
  "number 4" — just a number in English, associated with death in Japanese/Chinese/Korean

Two-tier approach:
  Tier 1: Lexicon lookup against data/sensitivity_lexicon.json (always runs, instant)
  Tier 2: LLM-based detection for subtle contextual cases (runs if Gemini key set)

Flags appear as warning banners in the UI before the translations, giving users
the chance to adjust their message before it causes offence.

Plug-in point:
  variant_formatter.run(..., sensitivity_fn=flag_sensitivity)
"""

import os
import re
import json
import logging
from typing import List, Dict

from pipeline.interfaces import PipelineInput, SensitivityFlag

logger = logging.getLogger(__name__)

# Path to lexicon — relative to this file's directory
_LEXICON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "sensitivity_lexicon.json"
)

# Module-level cache so we only read the file once
_lexicon_cache: Dict = {}


def _load_lexicon() -> Dict:
    global _lexicon_cache
    if _lexicon_cache:
        return _lexicon_cache
    try:
        with open(_LEXICON_PATH, encoding="utf-8") as f:
            _lexicon_cache = json.load(f)
        total = sum(len(v) for v in _lexicon_cache.values())
        logger.info(f"Sensitivity lexicon loaded: {len(_lexicon_cache)} languages, {total} entries")
    except FileNotFoundError:
        logger.warning(f"Sensitivity lexicon not found at {_LEXICON_PATH} — flagging disabled")
        _lexicon_cache = {}
    except json.JSONDecodeError as e:
        logger.error(f"Sensitivity lexicon JSON error: {e}")
        _lexicon_cache = {}
    return _lexicon_cache


def _lexicon_flags(text: str, target_lang: str) -> List[SensitivityFlag]:
    """
    Check text against the lexicon entries for the target language.
    Matches whole words/phrases case-insensitively.
    """
    lexicon = _load_lexicon()
    entries = lexicon.get(target_lang, [])
    if not entries:
        return []

    flags: List[SensitivityFlag] = []
    seen: set = set()
    text_lower = text.lower()

    for entry in entries:
        term = entry.get("term", "").lower().strip()
        if not term or term in seen:
            continue

        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        if re.search(pattern, text_lower):
            seen.add(term)
            flags.append({
                "span":         term,
                "warning_type": entry.get("warning_type", "cultural_sensitivity"),
                "severity":     entry.get("severity", "low"),
                "suggestion":   entry.get("suggestion", "Use culturally appropriate alternative."),
            })

    return flags


def _llm_flags(text: str, target_lang: str) -> List[SensitivityFlag]:
    """
    Ask Gemini to find subtle sensitivity issues the lexicon misses.
    Returns [] if no Gemini key or on any error.
    """
    from dotenv import load_dotenv
    load_dotenv()

    # Try Groq first, then Gemini
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not groq_key and not gemini_key:
        return []

    try:
        if groq_key:
            from groq import Groq as _SensGroq
            _sens_client = _SensGroq(api_key=groq_key)
            _use_groq = True
        else:
            from google import genai as _genai_new
            _genai_client = _genai_new.Client(api_key=gemini_key)
            _use_groq = False

        prompt = (
            f'Review this English text for cultural sensitivity issues when '
            f'translated into language code "{target_lang}".\n\n'
            f'Text: "{text}"\n\n'
            f'Return a JSON array of issues found. Each element has:\n'
            f'  span: the problematic text\n'
            f'  warning_type: one of "taboo", "offensive_in_target", "religious", '
            f'"political", "cultural_norm", "historical_sensitivity"\n'
            f'  severity: "low", "medium", or "high"\n'
            f'  suggestion: specific recommended alternative or advice\n\n'
            f'Only flag genuine cultural sensitivity issues. '
            f'If no issues, return []. Return ONLY the JSON array.'
        )

        if _use_groq:
            resp = _sens_client.chat.completions.create(
                model="llama-3.3-70b-versatile", max_tokens=512, temperature=0.1,
                response_format={"type": "json_object"},
                messages=[{"role":"system","content":"You are a cultural sensitivity expert. Always respond with JSON object containing 'issues' array."},
                          {"role":"user","content":prompt+"\n\nRespond as: {\"issues\": [...array...]}"}],
            )
            data = json.loads(resp.choices[0].message.content)
            items_raw = data.get("issues", data) if isinstance(data, dict) else data
            raw = json.dumps(items_raw) if isinstance(items_raw, list) else "[]"
        else:
            response = _genai_client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
            raw = response.text.strip()

        if raw.startswith("["):
            items = json.loads(raw)
        else:
            match = re.search(r"\[[\s\S]*\]", raw)
            items = json.loads(match.group(0)) if match else []

        flags: List[SensitivityFlag] = []
        for item in items:
            if all(k in item for k in ["span", "warning_type", "severity", "suggestion"]):
                flags.append({
                    "span":         item["span"],
                    "warning_type": item["warning_type"],
                    "severity":     item["severity"],
                    "suggestion":   item["suggestion"],
                })

        logger.info(f"LLM sensitivity detection: {len(flags)} additional flags")
        return flags

    except Exception as e:
        logger.warning(f"LLM sensitivity detection failed (non-fatal): {e}")
        return []


def flag_sensitivity(pipeline_input: PipelineInput) -> List[SensitivityFlag]:
    """
    Flag culturally sensitive expressions in the source text relative
    to the target culture.

    This is the function that plugs into variant_formatter.run() via the
    sensitivity_fn parameter.

    Args:
        pipeline_input: PipelineInput from text_preprocessor

    Returns:
        List[SensitivityFlag] — sorted high severity first, deduplicated

    Example:
        from pipeline.text_preprocessor import preprocess_text
        from pipeline.sensitivity_flagger import flag_sensitivity

        inp = flag_sensitivity(preprocess_text(
            "Pass the gift with your left hand", "ar"
        ))
        # inp[0] == {"span": "left hand", "severity": "high", ...}
    """
    text        = pipeline_input["text"]
    target_lang = pipeline_input["target_lang"]

    # Tier 1 — lexicon lookup (always runs)
    lex_flags = _lexicon_flags(text, target_lang)

    # Tier 2 — LLM (only if Gemini key set)
    llm_flags = _llm_flags(text, target_lang)

    # Merge — deduplicate by span, keep highest severity
    severity_rank = {"low": 0, "medium": 1, "high": 2}
    merged: Dict[str, SensitivityFlag] = {}

    for flag in lex_flags + llm_flags:
        key = flag["span"].lower()
        if key not in merged:
            merged[key] = flag
        else:
            if severity_rank.get(flag["severity"], 0) > severity_rank.get(merged[key]["severity"], 0):
                merged[key] = flag

    result = list(merged.values())
    result.sort(key=lambda f: -severity_rank.get(f["severity"], 0))

    logger.info(
        f"Sensitivity flagging complete: {len(lex_flags)} lexicon + "
        f"{len(llm_flags)} LLM = {len(result)} total"
    )
    return result