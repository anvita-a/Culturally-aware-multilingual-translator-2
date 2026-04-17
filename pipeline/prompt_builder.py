"""
prompt_builder.py
-----------------
Anvita's Module 3 — Prompt Construction

Responsibilities:
  - Take the CulturalContextObject (Handoff B) as input
  - Retrieve 3 culturally-similar few-shot examples from OPUS-100 via BM25
  - Assemble the three-part prompt: system + few-shot + CoT instruction
  - Return a fully formed prompt string ready for the Claude API

Why Chain-of-Thought (CoT):
  - Lyu et al. (2024) showed CoT prompting significantly improves cultural
    adaptation quality over direct translation prompting
  - Forcing the model to reason step-by-step ("identify CSI items → consider
    formality → produce variants") means the model engages with the cultural
    dimension before generating output
  - The CoT reasoning trace is also reused as the explainability output for
    the user — one prompt generates both translation AND explanation

Why BM25 for few-shot retrieval:
  - BM25 is fast, requires no embedding model, and works well when the query
    and corpus share vocabulary (same source language)
  - Dense retrieval would add latency and an embedding model dependency for
    no meaningful quality gain in this use case
  - The 3 retrieved examples are in the same language pair and are semantically
    similar to the input, which improves translation quality by grounding the
    model in domain-appropriate vocabulary

Why 3 examples (not 5 or 1):
  - 1 example is not enough for the model to infer the full output format
  - 5+ examples risks hitting the context window limit for long inputs
  - 3 is the standard in the few-shot MT literature (Jiao et al., 2023)
"""

import json
import os
import logging
from typing import List, Optional
from pipeline.interfaces import CulturalContextObject, CSISpan, SensitivityFlag

logger = logging.getLogger(__name__)

# Path to the OPUS-100 index (created by data/build_opus_index.py)
OPUS_INDEX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "opus_sample"
)


class BM25Retriever:
    """
    Retrieves culturally-similar few-shot examples from OPUS-100
    using BM25 lexical matching.

    The index is a dict keyed by "{source_lang}-{target_lang}" containing
    a list of {"source": str, "formal": str, "casual": str, "literal": str}
    records loaded from preprocessed OPUS-100 data.
    """

    def __init__(self, index_path: str = OPUS_INDEX_PATH):
        self.index_path = index_path
        self._indices = {}  # cache loaded language-pair indices
        self._bm25_objects = {}  # cache fitted BM25 objects

    def _load_index(self, source_lang: str, target_lang: str) -> list:
        """Load the OPUS sample for a given language pair."""
        key = f"{source_lang}-{target_lang}"
        if key in self._indices:
            return self._indices[key]

        filepath = os.path.join(self.index_path, f"{key}.json")
        if not os.path.exists(filepath):
            logger.warning(
                f"No OPUS index found for {key} at {filepath}. "
                "Returning empty example list."
            )
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._indices[key] = data
        logger.info(f"Loaded {len(data)} OPUS examples for {key}")
        return data

    def _get_bm25(self, source_lang: str, target_lang: str):
        """Build or retrieve a cached BM25 object for a language pair."""
        key = f"{source_lang}-{target_lang}"
        if key in self._bm25_objects:
            return self._bm25_objects[key], self._indices.get(key, [])

        corpus = self._load_index(source_lang, target_lang)
        if not corpus:
            return None, []

        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [doc["source"].lower().split() for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            self._bm25_objects[key] = bm25
            return bm25, corpus
        except ImportError:
            logger.warning("rank_bm25 not installed. Run: pip install rank_bm25")
            return None, corpus

    def retrieve(
        self,
        query: str,
        source_lang: str,
        target_lang: str,
        n: int = 3,
    ) -> List[dict]:
        """
        Retrieve the top-n most similar examples for the query.

        Args:
            query:       The source text to find similar examples for
            source_lang: ISO 639-1 source language code
            target_lang: ISO 639-1 target language code
            n:           Number of examples to retrieve (default: 3)

        Returns:
            List of dicts with keys: source, formal, casual, literal
            Returns [] if no index is available (pipeline continues with 0 examples)
        """
        bm25, corpus = self._get_bm25(source_lang, target_lang)

        if bm25 is None or not corpus:
            return []

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top-n indices, excluding exact matches (score == max possible)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n]

        return [corpus[i] for i in top_indices]


# Module-level singleton
_retriever_instance: Optional[BM25Retriever] = None


def get_retriever() -> BM25Retriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = BM25Retriever()
    return _retriever_instance


# ─── Language name mapping (for prompt readability) ───────────────────────────

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish",
    "ja": "Japanese", "hi": "Hindi", "ar": "Arabic",
    "zh": "Mandarin Chinese", "de": "German",
    "sw": "Swahili", "pt": "Brazilian Portuguese", "ko": "Korean",
}


def _format_csi_spans(csi_spans: List[CSISpan]) -> str:
    """Format CSI spans for inclusion in the system prompt."""
    if not csi_spans:
        return "None detected."
    lines = []
    for span in csi_spans:
        lines.append(
            f"  - \"{span['span']}\" → {span['category']}: {span['explanation']}"
        )
    return "\n".join(lines)


def _format_sensitivity_flags(flags: List[SensitivityFlag]) -> str:
    """Format sensitivity flags for inclusion in the system prompt."""
    if not flags:
        return "None."
    lines = []
    for flag in flags:
        lines.append(
            f"  - \"{flag['span']}\" [{flag['severity']} severity, {flag['warning_type']}]: "
            f"Suggested alternative: {flag['suggestion']}"
        )
    return "\n".join(lines)


def _format_few_shot_examples(examples: List[dict], source_lang: str, target_lang: str) -> str:
    """Format retrieved OPUS examples as few-shot demonstrations."""
    if not examples:
        return ""

    src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)

    lines = [f"\n### Reference Examples ({src_name} → {tgt_name})\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Source: {ex['source']}")
        if ex.get("formal"):
            lines.append(f"  Formal: {ex['formal']}")
        if ex.get("casual"):
            lines.append(f"  Casual: {ex['casual']}")
        if ex.get("literal"):
            lines.append(f"  Literal: {ex['literal']}")
        lines.append("")

    return "\n".join(lines)


def build_prompt(context: CulturalContextObject) -> str:
    """
    Assemble the three-part prompt from the CulturalContextObject.

    Part 1 — System prompt: encodes cultural context, formality, CSI spans
    Part 2 — Few-shot block: 3 BM25-retrieved OPUS examples
    Part 3 — CoT instruction: step-by-step reasoning instruction + output format

    Args:
        context: CulturalContextObject from Handoff B

    Returns:
        A single prompt string ready to be sent to the Claude API

    The returned prompt instructs Claude to return ONLY valid JSON matching
    the TranslationObject schema (Handoff C).
    """
    src_name = LANGUAGE_NAMES.get(context["source_lang"], context["source_lang"])
    tgt_name = LANGUAGE_NAMES.get(context["target_lang"], context["target_lang"])
    formality = context["source_formality"]
    turn = context["session_history"]["turn_count"]
    history_formality = context["session_history"].get("preferred_formality")

    # Retrieve few-shot examples
    retriever = get_retriever()
    examples = retriever.retrieve(
        query=context["text"],
        source_lang=context["source_lang"],
        target_lang=context["target_lang"],
        n=3,
    )
    few_shot_block = _format_few_shot_examples(
        examples, context["source_lang"], context["target_lang"]
    )

    # ── PART 1: System prompt ──────────────────────────────────────────────────
    system_prompt = f"""You are an expert multilingual translator specialising in culturally-aware translation. Your task is to translate text from {src_name} to {tgt_name} while preserving cultural appropriateness, adapting register, and providing transparent reasoning.

## Source Analysis
- Source language: {src_name}
- Target language: {tgt_name}
- Detected source register: {formality}
- Dialogue turn: {turn + 1}
{f"- User's established formality preference from previous turns: {history_formality}" if history_formality else ""}

## Culture-Specific Items Detected
{_format_csi_spans(context["csi_spans"])}

## Sensitivity Warnings
{_format_sensitivity_flags(context["sensitivity_flags"])}

## Your Responsibilities
1. Produce THREE translation variants: formal, casual, and literal
2. The FORMAL variant should use the highest appropriate register in {tgt_name}, including correct honorifics and polite grammar
3. The CASUAL variant should use natural everyday speech; if the source was already casual, the casual variant should reflect that register faithfully
4. The LITERAL variant should translate word-for-word to help language learners understand the grammatical structure
5. For any culture-specific items listed above, adapt them appropriately rather than translating literally, and explain what you changed
6. If any sensitivity warnings are present, avoid the flagged expressions in all three variants
7. Provide a cultural_notes field explaining any adaptations made and why

{few_shot_block}"""

    # ── PART 3: CoT instruction + output format ────────────────────────────────
    cot_instruction = f"""## Task

Translate the following {src_name} text into {tgt_name}. 

Think step by step:
STEP 1: Identify any culture-specific items, idioms, or pragmatic conventions in the source text that require adaptation (not just translation).
STEP 2: Determine the appropriate formality level for each variant in {tgt_name}, accounting for the source register ({formality}) and any honorific grammar rules specific to {tgt_name}.
STEP 3: Produce the formal variant, ensuring correct use of formal grammar, honorifics, and polite vocabulary appropriate to {tgt_name} culture.
STEP 4: Produce the casual variant using natural everyday speech in {tgt_name}.
STEP 5: Produce the literal variant as a word-for-word translation to show grammatical structure.
STEP 6: Write the cultural_notes explaining what cultural adaptations you made and why a direct translation would have been inappropriate or insufficient.

SOURCE TEXT:
"{context["text"]}"

Respond ONLY with a valid JSON object. No text before or after the JSON. Use this exact schema:
{{
  "cot_reasoning": "<your full step-by-step reasoning from steps 1-6 above, as a single string>",
  "formal": "<formal translation>",
  "casual": "<casual translation>",
  "literal": "<literal word-for-word translation>",
  "cultural_notes": "<plain English explanation of cultural adaptations made>",
  "confidence": <float between 0 and 1 reflecting your confidence in the cultural appropriateness>
}}"""

    # Combine all three parts
    full_prompt = system_prompt + "\n\n" + cot_instruction
    return full_prompt