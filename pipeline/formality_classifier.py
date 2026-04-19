"""
formality_classifier.py — Formality Classification

Detects whether source text is formal, neutral, or casual.

Key design principle:
  Structural signals only BOOST an existing score, never create one.
  "how are you" = neutral. "may I please meet them" = formal.
"""

import os, re, logging
from typing import Optional

logger = logging.getLogger(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "formality_classifier")
LABELS = ["formal", "neutral", "casual"]

# ── Strong signals — 3 points each ───────────────────────────────────────────

STRONG_CASUAL_WORDS = {
    "lol","lmao","lmfao","omg","omfg","wtf","wth","smh","ngl","tbh","btw",
    "fyi","imo","imho","afaik","brb","gtg","ttyl","idk","fomo","yolo",
    "u","ur","r","pls","plz","thx","ty","np","nvm","jk","bc","bcz","cuz",
    "coz","tho","tgt","tmr","tmrw","yo","sup","wassup","wazzup","heya","hiya",
    "yep","yup","nope","nah","gonna","wanna","gotta","kinda","sorta",
    "dunno","lemme","gimme","tryna","hey","wtf","omfg",
    # Profanity — inherently casual register
    "fuck","fucking","fucked","fucker","fucks","fuckin",
    "shit","shitty","bullshit","horseshit","shitting","shits",
    "bitch","bitchy","bitches","bitching",
    "ass","asshole","arsehole","arse","asses",
    "damn","goddamn","goddamnit","damnit",
    "crap","crappy","craphole",
    "hell","bloody","bastard","bugger",
    "piss","pissed","pissing","pissoff",
    "cunt","dick","dickhead","dicks",
    "cock","cocker","prick","wanker","twat",
    "douche","douchebag",
    # Sexually suggestive terms (casual register context)
    "sexy","horny","slutty","slutty","slut","ho","hoe",
}

# PROFANITY_TERMS and SEXUALLY_EXPLICIT_TERMS are loaded from
# data/profanity_terms.json (built by data/build_profanity_lists.py).
# Importing them here so variant_formatter and profanity_flagger
# can access them as a fallback if the JSON hasn't been built yet.

def _load_profanity_sets() -> tuple[set, set]:
    import json as _json, os as _os
    _path = _os.path.join(_os.path.dirname(__file__), "..", "data", "profanity_terms.json")
    try:
        with open(_path, encoding="utf-8") as _f:
            _data = _json.load(_f)
        return set(_data.get("profanity", [])), set(_data.get("explicit", []))
    except Exception:
        pass
    # Hardcoded fallback (used only if JSON not yet built)
    return (
        {"fuck","fucking","fucked","shit","shitty","bullshit","bitch","bitches",
         "ass","asshole","damn","goddamn","bastard","cunt","dick","dickhead",
         "cock","prick","wanker","twat","douche","piss","pissed","crap","hell",
         "bloody","bugger","jerk","idiot","motherfucker","arsehole","douchebag"},
        {"sex","sexual","sexually","naked","nude","nudity","porn","pornographic",
         "pornography","erotic","erotica","orgasm","masturbate","masturbation",
         "penis","vagina","breasts","boobs","tits","nipple","intercourse",
         "horny","slutty","slut","whore","hooker","prostitute","sexy","kinky",
         "fetish","bondage","naughty","dirty"},
    )

PROFANITY_TERMS, SEXUALLY_EXPLICIT_TERMS = _load_profanity_sets()

STRONG_FORMAL_WORDS = {
    "hereby","pursuant","notwithstanding","whereas","aforementioned",
    "henceforth","therein","thereof","herein","herewith","sincerely",
    "respectfully","esteemed","cordially",
}

# ── Pattern signals — 1 point each ───────────────────────────────────────────

CASUAL_PATTERNS = [
    r"\bcan't\b",r"\bwon't\b",r"\bdon't\b",r"\bdidn't\b",r"\bisn't\b",
    r"\baren't\b",r"\bwasn't\b",r"\bweren't\b",r"\bhasn't\b",r"\bhaven't\b",
    r"\bi'm\b",r"\byou're\b",r"\bwe're\b",r"\bthey're\b",
    r"\bi've\b",r"\byou've\b",r"\bwe've\b",
    r"\bi'll\b",r"\byou'll\b",r"\bwe'll\b",
    r"\bi'd\b",r"\byou'd\b",r"\bwhat's\b",r"\bthat's\b",r"\bit's\b",
    r"\bjust saying\b",r"\bno worries\b",r"\bmy bad\b",r"\bfor real\b",
    r"!{2,}",r"\?{2,}",r"haha",r"hehe",r":\)",r":\(",r";-?\)",r"<3",
]

FORMAL_PATTERNS = [
    # Classic formal openers
    r"\bdear\b",
    r"\bsincerely\b", r"\bregards\b",
    r"\bto whom it may concern\b",
    r"\bi am writing\b",
    r"\bplease find\b",
    r"\bkindly\b",
    r"\bi would like to\b", r"\bi wish to\b",
    r"\bi would be (grateful|happy|pleased|honoured|honored)\b",
    r"\bfurthermore\b", r"\bmoreover\b", r"\bnevertheless\b", r"\bconsequently\b",
    r"\bin accordance with\b", r"\bwith reference to\b", r"\bwith regard to\b",
    r"\bpursuant to\b", r"\bon behalf of\b",
    r"\bone must\b", r"\bone should\b",
    r"\bit is (imperative|essential|required|necessary)\b",
    r"\bI (formally|hereby|respectfully|humbly)\b",
    r"\bthank you for your (consideration|time|attention|assistance)\b",
    r"\blooking forward to\b",

    # "please" as a standalone politeness marker at start or before a verb
    # Matches "please tell", "please send", "please let", "please inform" etc.
    r"\bplease\b",

    # Polite request forms
    r"\bmay I\b",
    r"\bcould you\b",
    r"\bwould you\b",
    r"\bmight I\b",
    r"\bshall I\b", r"\bshall we\b",
    r"\bI request\b", r"\bI humbly\b",
    r"\bpermission to\b",
    r"\bI seek\b", r"\bI kindly\b",
    r"\bif I may\b", r"\bif you don't mind\b", r"\bif you wouldn't mind\b",
    r"\bat your (convenience|earliest convenience)\b",
    r"\bwith (your|all due|the utmost) (permission|respect)\b",

    # Titles
    r"\b(professor|dr|mr|ms|mrs|sir|madam)\b",

    # Professional/formal vocabulary
    r"\bregarding\b", r"\bconcerning\b", r"\bpertaining to\b",
    r"\bhenceforth\b", r"\bwhereby\b", r"\bwherein\b",
    r"\battached herewith\b", r"\bas per\b",
    r"\bplease be advised\b", r"\bplease note\b",
    r"\bI trust\b", r"\bI hope this (finds|reaches)\b",

    # Professional statements (often formal even without explicit markers)
    r"\bI will be (attending|present|available|unable|late|joining)\b",
    r"\bI am (pleased|delighted|unable|regret|afraid|sorry to inform)\b",
    r"\bwe are (pleased|delighted|writing|reaching out|following up)\b",
    r"\bI wish to (inform|advise|notify|request|confirm)\b",
    r"\bplease (inform|advise|notify|confirm|let us know|be informed)\b",
    r"\b(inform|advise|notify) (them|you|us|me)\b",
    r"\bwill be (late|delayed|absent|unavailable|attending|joining)\b",
    r"\bI (regret|apologise|apologize) to\b",
    r"\bthis is to (inform|notify|advise|confirm)\b",
    r"\bkindly (note|be advised|confirm|advise|inform)\b",
]


def _llm_classify_formality(text: str, lang: str) -> Optional[tuple[str, float]]:
    """
    Use Gemini to detect formality for non-English source text.
    The heuristic classifier only knows English signals, so for Japanese,
    Hindi, Arabic, French etc. we ask the LLM directly.
    Returns (label, confidence) or None if API unavailable.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai as _genai_new
        import json
        genai.configure(api_key=api_key)

        lang_names = {
            "ja": "Japanese", "fr": "French", "es": "Spanish",
            "de": "German", "hi": "Hindi", "ar": "Arabic",
            "zh": "Mandarin Chinese", "ko": "Korean",
            "pt": "Portuguese", "sw": "Swahili",
        }
        lang_name = lang_names.get(lang, lang)

        prompt = (
            f'Classify the formality register of this {lang_name} text.\n'
            f'Text: "{text}"\n\n'
            f'Return JSON: {{"label": "formal|neutral|casual", "confidence": 0.0-1.0}}\n'
            f'formal = polite, respectful, professional\n'
            f'casual = informal, relaxed, between friends\n'
            f'neutral = neither clearly formal nor casual\n'
            f'Return ONLY the JSON.'
        )

        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )
        response = _genai_client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        data = json.loads(response.text)
        label = data.get("label", "neutral")
        conf  = float(data.get("confidence", 0.75))
        if label in ("formal", "neutral", "casual"):
            logger.info(f"LLM formality ({lang}): {label} ({conf:.0%})")
            return label, conf
    except Exception as e:
        logger.debug(f"LLM formality classification failed: {e}")

    return None


class FormalityClassifier:

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.model.eval()
                logger.info("Formality classifier loaded.")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Using heuristic.")
        else:
            logger.warning("No trained model found. Using heuristic fallback.")

    def classify(self, text: str, lang: str = "en") -> tuple[str, float]:
        # Neural model (if trained) works for all languages
        if self.model is not None and self.tokenizer is not None:
            return self._neural_classify(text)
        # Heuristic only works well for English — for other languages use LLM
        if lang != "en":
            result = _llm_classify_formality(text, lang)
            if result is not None:
                return result
        return self._heuristic_classify(text)

    def _neural_classify(self, text: str) -> tuple[str, float]:
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, pred_idx = probs.max(dim=-1)
        return LABELS[pred_idx.item()], round(confidence.item(), 3)

    def _heuristic_classify(self, text: str) -> tuple[str, float]:
        """
        Stage 1: vocabulary + pattern signals (only these drive the decision)
        Stage 2: structural boost (only if stage 1 has signal)
        Stage 3: decision
        """
        text_lower = text.lower().strip()
        words = [re.sub(r"[^a-z']", "", w) for w in text_lower.split()]

        vocab_casual = 0
        vocab_formal = 0

        # Strong word signals
        for word in words:
            clean = re.sub(r"[^a-z]", "", word)
            if clean in STRONG_CASUAL_WORDS:
                vocab_casual += 3
            if clean in STRONG_FORMAL_WORDS:
                vocab_formal += 3

        # Pattern signals
        for p in CASUAL_PATTERNS:
            if re.search(p, text_lower):
                vocab_casual += 1

        for p in FORMAL_PATTERNS:
            if re.search(p, text_lower, re.IGNORECASE):
                vocab_formal += 1

        # Structural boost (only when there's existing signal)
        structural_casual = 0.0
        if vocab_casual > 0 or vocab_formal > 0:
            if text and text[0].islower() and vocab_casual >= vocab_formal:
                structural_casual += 0.5
            if text and text[-1] not in ".!?":
                structural_casual += 0.5

        total_casual = vocab_casual + structural_casual
        total_formal = float(vocab_formal)

        logger.debug(f"formal={total_formal} casual={total_casual} text='{text[:50]}'")

        if vocab_casual == 0 and vocab_formal == 0:
            return "neutral", 0.60

        if total_casual > total_formal:
            conf = min(0.95, 0.65 + (total_casual / (total_casual + total_formal + 1)) * 0.30)
            return "casual", round(conf, 2)

        if total_formal > total_casual:
            conf = min(0.95, 0.65 + (total_formal / (total_casual + total_formal + 1)) * 0.30)
            return "formal", round(conf, 2)

        return "neutral", 0.62


_classifier_instance: Optional[FormalityClassifier] = None

def get_formality_classifier() -> FormalityClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = FormalityClassifier()
    return _classifier_instance

def classify_formality(text: str, lang: str = "en") -> tuple[str, float]:
    return get_formality_classifier().classify(text, lang)