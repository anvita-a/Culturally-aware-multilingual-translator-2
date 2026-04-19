"""
llm_engine.py — LLM API Integration with fallback chain

Fallback order:
  1. Gemini 1.5 Flash  (free, works on campus WiFi)
  2. GPT-4o-mini       (best quality)
  3. GPT-3.5-turbo     (cheaper OpenAI)
  4. Groq/Llama        (free, may be blocked on campus)
"""

import os
import json
import re
import logging
import time
from dotenv import load_dotenv
from pipeline.interfaces import (
    CulturalContextObject,
    TranslationObject,
    LearningLayerInput,
)

load_dotenv()
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish",
    "ja": "Japanese", "hi": "Hindi", "ar": "Arabic",
    "zh": "Mandarin Chinese", "de": "German",
    "sw": "Swahili", "pt": "Brazilian Portuguese", "ko": "Korean",
}

PRONUNCIATION_SYSTEM = {
    "ja": "romaji (Hepburn)",
    "zh": "pinyin with tone numbers",
    "ko": "Revised Romanization",
    "hi": "IAST romanization",
    "ar": "Arabic romanization (ALA-LC)",
    "fr": "IPA pronunciation",
    "de": "IPA pronunciation",
    "es": "IPA pronunciation",
    "sw": "standard romanization",
    "pt": "IPA pronunciation",
    "en": "IPA pronunciation",
}

# Words that signal gender ambiguity in the source text
GENDER_TRIGGER_WORDS = {
    "them", "they", "their", "theirs", "themselves",
    "friend", "colleague", "coworker", "classmate", "teammate",
    "teacher", "student", "doctor", "nurse", "patient",
    "manager", "boss", "employee", "assistant", "intern",
    "neighbor", "neighbour", "stranger", "person", "someone",
    "partner", "spouse", "sibling", "cousin", "relative",
    "client", "customer", "guest", "visitor",
}


# ── LLM backends ──────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, max_tokens: int) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("No GEMINI_API_KEY")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text
    except ImportError:
        raise EnvironmentError("google-genai not installed. Run: pip install google-genai")


def _call_openai(prompt: str, max_tokens: int, model: str) -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("No OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _call_groq(prompt: str, max_tokens: int) -> str:
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("No GROQ_API_KEY")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content


def _call_llm(prompt: str, max_tokens: int = 1500) -> tuple[str, str]:
    attempts = [
        ("groq-llama-3.3",   lambda: _call_groq(prompt, max_tokens)),
        ("gemini-2.0-flash", lambda: _call_gemini(prompt, max_tokens)),
        ("gpt-4o-mini",      lambda: _call_openai(prompt, max_tokens, "gpt-4o-mini")),
        ("gpt-3.5-turbo",    lambda: _call_openai(prompt, max_tokens, "gpt-3.5-turbo")),
    ]
    last_error = None
    for model_name, caller in attempts:
        try:
            logger.info(f"Trying {model_name}...")
            result = caller()
            logger.info(f"Success with {model_name}")
            return result, model_name
        except EnvironmentError as e:
            logger.debug(f"{model_name}: skipped ({e})")
            continue
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in [
                "quota", "429", "insufficient", "rate_limit", "billing",
                "authentication", "401", "403", "access denied", "network"
            ]):
                logger.warning(f"{model_name}: {e}, trying next")
                last_error = e
                continue
            else:
                logger.warning(f"{model_name}: unexpected error: {e}, trying next")
                last_error = e
                continue
    raise EnvironmentError(
        f"All LLM backends failed. Last error: {last_error}\n"
        "Solutions:\n"
        "  1. Add GEMINI_API_KEY to .env (free, get at aistudio.google.com)\n"
        "  2. Add billing to OpenAI at platform.openai.com/settings/billing\n"
        "  3. Try on a different network (hotspot) if on campus WiFi"
    )


# ── Gender detection (Python-side) ───────────────────────────────────────────

def _detect_gender_ambiguity(text: str, target_lang: str) -> bool:
    """
    Detect in Python whether the source text has gender-ambiguous elements
    that would affect translation in the target language.
    This is more reliable than asking the LLM to detect it.
    """
    gendered_langs = {"fr", "es", "de", "hi", "ar", "pt", "ko", "ja"}
    if target_lang not in gendered_langs:
        return False
    text_lower = text.lower()
    words = set(re.findall(r'\b[a-z]+\b', text_lower))
    # Check for gender trigger words
    if words & GENDER_TRIGGER_WORDS:
        return True
    # Check for first-person statements that differ by gender in some languages
    if target_lang in ("ja", "ko") and re.search(r'\b(i am|i feel|i was|i became)\b', text_lower):
        return True
    if target_lang in ("fr", "es", "de", "hi", "ar", "pt"):
        if re.search(r'\b(i am|i feel|i was|i became|i am feeling|i\'m)\b', text_lower):
            return True
    return False


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_prompt(context: CulturalContextObject) -> str:
    src_name   = LANGUAGE_NAMES.get(context["source_lang"], context["source_lang"])
    tgt_name   = LANGUAGE_NAMES.get(context["target_lang"], context["target_lang"])
    tgt_lang   = context["target_lang"]
    formality  = context["source_formality"]
    text       = context["text"]
    pron_sys   = PRONUNCIATION_SYSTEM.get(tgt_lang, "romanization")
    sep        = "\u00b7"
    is_gendered = _detect_gender_ambiguity(text, tgt_lang)

    cultural_context = ""
    if context["csi_spans"]:
        items = [f'"{s["span"]}" ({s["category"]})' for s in context["csi_spans"][:3]]
        cultural_context += f"Culture-specific items: {', '.join(items)}. Adapt rather than translate literally.\n"
    if context["sensitivity_flags"]:
        flags = [f'"{f["span"]}"' for f in context["sensitivity_flags"][:2]]
        cultural_context += f"Avoid: {', '.join(flags)}.\n"

    gender_section = ""
    if is_gendered:
        gender_section = f"""
GENDER VARIANTS REQUIRED:
The source text contains gender-ambiguous words (e.g. "them", "they", "friend", "colleague").
In {tgt_name}, the translation differs based on the gender of that person.
You MUST provide all four variants below — do not leave them empty.
Set is_gendered to true."""
    else:
        gender_section = "Set is_gendered to false. Leave formal_male, formal_female, casual_male, casual_female as empty strings."

    # ── BM25 few-shot examples from OPUS-100 ─────────────────────────────────
    few_shot_block = ""
    try:
        from pipeline.prompt_builder import get_retriever, _format_few_shot_examples
        retriever = get_retriever()
        examples  = retriever.retrieve(
            query=text,
            source_lang=context["source_lang"],
            target_lang=tgt_lang,
            n=3,
        )
        if examples:
            few_shot_block = _format_few_shot_examples(
                examples, context["source_lang"], tgt_lang
            )
            logger.info(f"BM25 retrieved {len(examples)} few-shot examples for {context['source_lang']}-{tgt_lang}")
        else:
            logger.debug("No OPUS index available — proceeding without few-shot examples")
    except Exception as e:
        logger.debug(f"Few-shot retrieval skipped (non-fatal): {e}")

    return f"""You are an expert translator and cultural linguist for {src_name} to {tgt_name}.

SOURCE TEXT: "{text}"
DETECTED SOURCE REGISTER: {formality}
{cultural_context}
{few_shot_block}
{gender_section}

Return a JSON object with EXACTLY these keys. Every key must be present.

detected_formality: "{formality}"

formal: {formality}-register {tgt_name} translation using correct honorifics and polite forms
  MUST NOT be empty — write actual {tgt_name} text
casual: casual everyday {tgt_name} translation
  MUST NOT be empty — write actual {tgt_name} text

formal_pronunciation: {pron_sys} romanization of formal translation
casual_pronunciation: {pron_sys} romanization of casual translation

literal: word-by-word breakdown. Format depends on translation direction:

  CASE A — target language uses NON-LATIN script (Japanese, Chinese, Korean, Hindi, Arabic, etc):
    Show: targetword(English_meaning){sep}targetword(English_meaning)
    Correct: こんにちは(hello){sep}お元気(wellbeing){sep}ですか(question?)
    WRONG:   hello(konnichiwa){sep}how(dou)

  CASE B — source language uses NON-LATIN script AND target is English:
    Show: sourceword(romanisation/English_meaning){sep}sourceword(romanisation/English_meaning)
    The source words keep their original script, followed by (romanisation/meaning) together
    Example (Hindi→English): यह(yah/this){sep}एक(ek/one){sep}सरल(saral/simple){sep}हिंदी(hindi/Hindi){sep}वाक्य(vakya/sentence){sep}है(hai/is)
    Example (Japanese→English): 会議(kaigi/meeting){sep}に(ni/at){sep}遅れ(okure/late){sep}ます(masu/will be)
    Example (Korean→English): 안녕(annyeong/hello){sep}하세요(haseyo/formal greeting)
    CRITICAL: BOTH romanisation AND meaning must appear inside the brackets, separated by /

  CASE C — both source and target use Latin script (French→English, Spanish→German, etc):
    Show the formal target translation broken into content words with their source equivalent
    Example (French→English): Bonjour(hello){sep}comment(how){sep}allez-vous(are you)

literal_pronunciation: {pron_sys} pronunciation of the full formal translation as a single readable string
  Example for Japanese: "Kyōju, kono koto wa kinkyū desu. Sugu ni kite kudasai"
  Example for French: "pro-feh-sur, say-see oor-jahn"
  Example for Hindi→English: leave as empty string (target is English, no romanisation needed)

is_gendered: {"true" if is_gendered else "false"}

formal_male: {"formal " + tgt_name + " translation assuming the ambiguous person is MALE — MUST NOT be empty" if is_gendered else ""}
formal_female: {"formal " + tgt_name + " translation assuming the ambiguous person is FEMALE — MUST NOT be empty" if is_gendered else ""}
casual_male: {"casual " + tgt_name + " translation for male — MUST NOT be empty" if is_gendered else ""}
casual_female: {"casual " + tgt_name + " translation for female — MUST NOT be empty" if is_gendered else ""}
formal_male_pronunciation: {"" if not is_gendered else pron_sys + " of formal_male"}
formal_female_pronunciation: {"" if not is_gendered else pron_sys + " of formal_female"}
casual_male_pronunciation: {"" if not is_gendered else pron_sys + " of casual_male"}
casual_female_pronunciation: {"" if not is_gendered else pron_sys + " of casual_female"}
gender_note: {"one sentence explaining why gender changes this translation in " + tgt_name if is_gendered else ""}

cultural_notes: explain {tgt_name} cultural norms affecting register for this specific text
tone_recommendation_reason: one sentence on why {formality} register is used here

word_breakdown: array of top 4 content words from formal translation:
  [{{"word": "{tgt_name} script", "romanization": "pronunciation", "pos": "noun/verb/etc", "meaning": "English", "note": "grammar or cultural note"}}]

flashcards: array of 3 vocabulary cards:
  [{{"front": "{tgt_name} word", "back": "English meaning", "pronunciation": "romanized", "example": "{tgt_name} sentence", "example_translation": "English"}}]

cefr_level: A1 or A2 or B1 or B2 or C1 or C2
confidence: float 0.0 to 1.0
cot_reasoning: brief step-by-step reasoning about cultural choices"""


def _build_fallback_prompt(context: CulturalContextObject) -> str:
    src_name    = LANGUAGE_NAMES.get(context["source_lang"], context["source_lang"])
    tgt_name    = LANGUAGE_NAMES.get(context["target_lang"], context["target_lang"])
    tgt_lang    = context["target_lang"]
    text        = context["text"]
    pron_sys    = PRONUNCIATION_SYSTEM.get(tgt_lang, "romanization")
    sep         = "\u00b7"
    is_gendered = _detect_gender_ambiguity(text, tgt_lang)

    EXAMPLES = {
        "ja": (
            '{"formal":"\u6559\u6388\u3001\u3053\u306e\u3053\u3068\u306f\u7dca\u6025\u3067\u3059\u3002\u3059\u3050\u306b\u6765\u3066\u304f\u3060\u3055\u3044",'
            '"casual":"\u6559\u6388\u3001\u6025\u3044\u3067\u304f\u3060\u3055\u3044\u3002\u3059\u3050\u6765\u3066",'
            '"formal_pronunciation":"Kyōju, kono koto wa kinkyū desu. Sugu ni kite kudasai",'
            '"casual_pronunciation":"Kyōju, isoide kudasai. Sugu kite",'
            '"literal":"\u6559\u6388(professor)\u00b7\u3053\u306e\u3053\u3068(this matter)\u00b7\u7dca\u6025(urgent)\u00b7\u3067\u3059(is)\u00b7\u3059\u3050(soon)\u00b7\u6765\u3066(come)\u00b7\u304f\u3060\u3055\u3044(please)",'
            '"literal_pronunciation":"Kyōju, kono koto wa kinkyū desu. Sugu ni kite kudasai",'
            '"detected_formality":"formal",'
            '"is_gendered":false,'
            '"formal_male":"","formal_female":"","casual_male":"","casual_female":"",'
            '"formal_male_pronunciation":"","formal_female_pronunciation":"",'
            '"casual_male_pronunciation":"","casual_female_pronunciation":"",'
            '"gender_note":"",'
            '"cultural_notes":"Japanese requests to authority figures use polite forms and honorific titles.",'
            '"tone_recommendation_reason":"Professor is an authority figure requiring formal keigo in Japanese.",'
            '"word_breakdown":[{"word":"\u6559\u6388","romanization":"kyōju","pos":"noun","meaning":"professor","note":"honorific title"}],'
            '"flashcards":[{"front":"\u7dca\u6025","back":"urgent","pronunciation":"kinkyū","example":"\u3053\u308c\u306f\u7dca\u6025\u3067\u3059\u3002","example_translation":"This is urgent."}],'
            '"cefr_level":"B1","confidence":0.9,"cot_reasoning":"Used polite forms appropriate for addressing a professor."}'
        ),
        "fr": (
            '{"formal":"Professeur, c\'est urgent. Venez s\'il vous plaît immédiatement",'
            '"casual":"Prof, c\'est urgent. Viens vite",'
            '"formal_pronunciation":"pro-feh-sur, say oor-jahn. veh-nay seel voo play ee-may-dya-teh-mahn",'
            '"casual_pronunciation":"prof, say oor-jahn. vyahn veet",'
            '"literal":"Professeur(Professor)\u00b7c\'est(it is)\u00b7urgent(urgent)\u00b7venez(come-formal)\u00b7s\'il vous plaît(please)",'
            '"literal_pronunciation":"pro-feh-sur, say oor-jahn, veh-nay, seel voo play",'
            '"detected_formality":"formal","is_gendered":false,'
            '"formal_male":"","formal_female":"","casual_male":"","casual_female":"",'
            '"formal_male_pronunciation":"","formal_female_pronunciation":"",'
            '"casual_male_pronunciation":"","casual_female_pronunciation":"",'
            '"gender_note":"",'
            '"cultural_notes":"French uses vous with authority figures like professors.",'
            '"tone_recommendation_reason":"Addressing a professor requires vous form in French.",'
            '"word_breakdown":[{"word":"urgent","romanization":"oor-jahn","pos":"adjective","meaning":"urgent","note":"same as English"}],'
            '"flashcards":[{"front":"urgent","back":"urgent","pronunciation":"oor-jahn","example":"C\'est urgent.","example_translation":"This is urgent."}],'
            '"cefr_level":"B1","confidence":0.9,"cot_reasoning":"Vous form used for professor as authority figure."}'
        ),
    }

    gender_keys = ""
    if is_gendered:
        gender_keys = (
            f'"is_gendered":true,'
            f'"formal_male":"formal {tgt_name} translation if the person is male",'
            f'"formal_female":"formal {tgt_name} translation if the person is female",'
            f'"casual_male":"casual {tgt_name} translation if the person is male",'
            f'"casual_female":"casual {tgt_name} translation if the person is female",'
            f'"formal_male_pronunciation":"{pron_sys} of formal_male",'
            f'"formal_female_pronunciation":"{pron_sys} of formal_female",'
            f'"casual_male_pronunciation":"{pron_sys} of casual_male",'
            f'"casual_female_pronunciation":"{pron_sys} of casual_female",'
            f'"gender_note":"why gender changes this translation in {tgt_name}",'
        )
    else:
        gender_keys = '"is_gendered":false,"formal_male":"","formal_female":"","casual_male":"","casual_female":"","formal_male_pronunciation":"","formal_female_pronunciation":"","casual_male_pronunciation":"","casual_female_pronunciation":"","gender_note":"",'

    example = EXAMPLES.get(tgt_lang, EXAMPLES["ja"])

    return (
        f"Translate \"{text}\" from {src_name} to {tgt_name}.\n\n"
        f"Study this example JSON and produce the SAME structure for \"{text}\":\n"
        f"{example}\n\n"
        f"Rules:\n"
        f"- formal and casual MUST contain actual {tgt_name} script — never empty strings\n"
        f"- literal: "
        + ("use SOURCE script words with (romanisation/English_meaning) e.g. "
           "यह(yah/this)·एक(ek/one)·है(hai/is) "
           if context.get('target_lang') == 'en' else
           f"use {tgt_name} script words with English in brackets: word(meaning){sep}word(meaning)")
        + "\n"
        f"- literal_pronunciation: {pron_sys} pronunciation of the full formal translation\n"
        f"- {gender_keys}\n"
        f"Output ONLY valid JSON. No markdown."
    )


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _extract_json_from_text(raw: str) -> str:
    raw = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        raw = fence_match.group(1).strip()
    brace_match = re.search(r"\{[\s\S]*\}", raw)
    if brace_match:
        raw = brace_match.group(0)
    return raw


def _repair_json(raw: str) -> str:
    try:
        from json_repair import repair_json
        return repair_json(raw)
    except ImportError:
        pass
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    if not raw.endswith("}"):
        raw = raw + "}"
    return raw


def _parse_response(raw_text: str) -> dict:
    logger.debug(f"Raw response: {raw_text[:400]}")
    for strategy in ["direct", "extract", "repair"]:
        try:
            if strategy == "direct":
                return json.loads(raw_text)
            elif strategy == "extract":
                return json.loads(_extract_json_from_text(raw_text))
            elif strategy == "repair":
                return json.loads(_repair_json(_extract_json_from_text(raw_text)))
        except Exception:
            continue
    raise ValueError(f"Could not parse JSON. Raw: {raw_text[:300]}")


def _fill_missing_fields(data: dict, context: CulturalContextObject) -> dict:
    src_name   = LANGUAGE_NAMES.get(context["source_lang"], context["source_lang"])
    tgt_name   = LANGUAGE_NAMES.get(context["target_lang"], context["target_lang"])
    formality  = context.get("source_formality", "neutral")
    is_gendered = _detect_gender_ambiguity(context["text"], context["target_lang"])

    def is_empty(val):
        return val is None or (isinstance(val, str) and len(val.strip()) < 2)

    defaults = {
        "formal":                       data.get("formal") or "",
        "casual":                       data.get("casual") or "",
        "literal":                      data.get("literal") or context["text"],
        "detected_formality":           data.get("detected_formality") or formality,
        "formal_pronunciation":         data.get("formal_pronunciation") or data.get("pronunciation") or "",
        "casual_pronunciation":         data.get("casual_pronunciation") or "",
        "literal_pronunciation":        data.get("literal_pronunciation") or "",
        "cultural_notes":               data.get("cultural_notes") or f"Translation from {src_name} to {tgt_name}.",
        "tone_recommendation_reason":   data.get("tone_recommendation_reason") or
                                        f"{formality.title()} register detected from source text.",
        "is_gendered":                  bool(data.get("is_gendered", is_gendered)),
        "formal_male":                  data.get("formal_male") or "",
        "formal_female":                data.get("formal_female") or "",
        "casual_male":                  data.get("casual_male") or "",
        "casual_female":                data.get("casual_female") or "",
        "formal_male_pronunciation":    data.get("formal_male_pronunciation") or "",
        "formal_female_pronunciation":  data.get("formal_female_pronunciation") or "",
        "casual_male_pronunciation":    data.get("casual_male_pronunciation") or "",
        "casual_female_pronunciation":  data.get("casual_female_pronunciation") or "",
        "gender_note":                  data.get("gender_note") or "",
        "word_breakdown":               data.get("word_breakdown") or [],
        "flashcards":                   data.get("flashcards") or [],
        "cefr_level":                   data.get("cefr_level") or "B1",
        "confidence":                   data.get("confidence") or 0.85,
        "cot_reasoning":                data.get("cot_reasoning") or "Translation with cultural awareness.",
    }

    for key, default in defaults.items():
        val = data.get(key)
        if val is None or (isinstance(val, str) and is_empty(val)):
            data[key] = default

    # Override is_gendered with Python detection if LLM missed it
    if is_gendered and not data.get("is_gendered"):
        data["is_gendered"] = True
        logger.info("Overriding is_gendered=True based on Python detection")

    # Last-resort recovery for empty formal/casual
    if is_empty(data.get("formal")) or is_empty(data.get("casual")):
        logger.warning("formal/casual still empty — attempting direct recovery call")
        tgt = LANGUAGE_NAMES.get(context["target_lang"], context["target_lang"])
        src = LANGUAGE_NAMES.get(context["source_lang"], context["source_lang"])
        recovery_prompt = (
            f'Translate "{context["text"]}" from {src} to {tgt}.\n'
            f'Return JSON: {{"formal": "polite {tgt} translation", "casual": "casual {tgt} translation"}}\n'
            f'IMPORTANT: Replace placeholder text with actual {tgt} words.'
        )
        try:
            raw, _ = _call_llm(recovery_prompt, max_tokens=256)
            recovery = json.loads(_extract_json_from_text(raw))
            if not is_empty(recovery.get("formal")):
                data["formal"] = recovery["formal"]
            if not is_empty(recovery.get("casual")):
                data["casual"] = recovery["casual"]
            logger.info("Recovery call succeeded")
        except Exception as e:
            logger.warning(f"Recovery call failed: {e}")

    return data


def _validate_translation_dict(data: dict) -> bool:
    for field in ["formal", "casual", "literal"]:
        val = data.get(field, "")
        if not val or len(val.strip()) < 2:
            logger.warning(f"Critical field empty: '{field}'")
            return False
    if not isinstance(data.get("confidence"), (int, float)):
        data["confidence"] = 0.85
    data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
    return True


# ── Main translation ──────────────────────────────────────────────────────────

def translate(
    context: CulturalContextObject,
    user_cefr_level: str = "B2",
) -> tuple[TranslationObject, LearningLayerInput]:

    main_prompt     = _build_prompt(context)
    fallback_prompt = _build_fallback_prompt(context)
    is_gendered     = _detect_gender_ambiguity(context["text"], context["target_lang"])

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Translation attempt {attempt}/{MAX_RETRIES}")
            current_prompt = main_prompt if attempt == 1 else fallback_prompt
            raw_text, model_used = _call_llm(current_prompt, max_tokens=2500)

            data = _parse_response(raw_text)
            data = _fill_missing_fields(data, context)

            if not _validate_translation_dict(data):
                raise ValueError(f"Fields empty. Raw: {raw_text[:300]}")

            translation: TranslationObject = {
                "formal":                       data["formal"],
                "casual":                       data["casual"],
                "literal":                      data["literal"],
                "detected_formality":           data.get("detected_formality", context.get("source_formality", "neutral")),
                "formal_pronunciation":         data.get("formal_pronunciation", ""),
                "casual_pronunciation":         data.get("casual_pronunciation", ""),
                "literal_pronunciation":        data.get("literal_pronunciation", ""),
                "cultural_notes":               data["cultural_notes"],
                "confidence":                   data["confidence"],
                "cot_reasoning":                data["cot_reasoning"],
                "source_lang":                  context["source_lang"],
                "target_lang":                  context["target_lang"],
                "source_text":                  context["text"],
                "csi_spans":                    context["csi_spans"],
                "sensitivity_flags":            context["sensitivity_flags"],
                "tone_recommendation_reason":   data.get("tone_recommendation_reason", ""),
                "is_gendered":                  bool(data.get("is_gendered", is_gendered)),
                "formal_male":                  data.get("formal_male", ""),
                "formal_female":                data.get("formal_female", ""),
                "casual_male":                  data.get("casual_male", ""),
                "casual_female":                data.get("casual_female", ""),
                "formal_male_pronunciation":    data.get("formal_male_pronunciation", ""),
                "formal_female_pronunciation":  data.get("formal_female_pronunciation", ""),
                "casual_male_pronunciation":    data.get("casual_male_pronunciation", ""),
                "casual_female_pronunciation":  data.get("casual_female_pronunciation", ""),
                "gender_note":                  data.get("gender_note", ""),
                "word_breakdown":               data.get("word_breakdown", []),
                "flashcards":                   data.get("flashcards", []),
                "cefr_level":                   data.get("cefr_level", "B1"),
                "model_used":                   model_used,
            }

            learning_input: LearningLayerInput = {
                "best_translation": data["formal"],
                "target_lang":      context["target_lang"],
                "source_text":      context["text"],
                "source_lang":      context["source_lang"],
                "user_cefr_level":  user_cefr_level,
                "csi_spans":        context["csi_spans"],
                "cot_reasoning":    data["cot_reasoning"],
            }

            logger.info(f"Done via {model_used}, CEFR={data.get('cefr_level')}, gendered={data.get('is_gendered')}")
            return translation, learning_input

        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    raise ValueError(f"Translation failed after {MAX_RETRIES} attempts. Last: {last_error}")


def translate_with_cefr_adjustment(
    context: CulturalContextObject,
    translation: TranslationObject,
    user_cefr_level: str,
) -> TranslationObject:
    if user_cefr_level not in ("A1", "A2", "B1"):
        return translation
    lang_name = LANGUAGE_NAMES.get(context["target_lang"], context["target_lang"])
    prompt = (
        f'Simplify this {lang_name} sentence for a {user_cefr_level} learner: "{translation["casual"]}"\n'
        f'Keep the same meaning. Return JSON: {{"simplified_casual": "..."}}'
    )
    try:
        raw_text, _ = _call_llm(prompt, max_tokens=256)
        data = _parse_response(raw_text)
        if data.get("simplified_casual"):
            translation = dict(translation)
            translation["casual"] = data["simplified_casual"]
    except Exception as e:
        logger.warning(f"CEFR simplification failed: {e}")
    return translation
