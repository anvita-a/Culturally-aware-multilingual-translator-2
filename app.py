"""
app.py — CulturaTrans Streamlit UI
Run: streamlit run app.py
"""

import re
import json
import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="CulturaTrans",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display: none;}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.main-header h1 { color: white; font-size: 2rem; font-weight: 600; margin: 0 0 0.3rem 0; letter-spacing: -0.02em; }
.main-header p  { color: rgba(255,255,255,0.5); font-size: 0.88rem; margin: 0; }

.tone-card { border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1.2rem; border: 1px solid; }
.tone-badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; letter-spacing: 0.03em; margin-bottom: 8px; }

.hero-translation {
    background: linear-gradient(135deg, rgba(99,179,237,0.08), rgba(129,230,217,0.08));
    border: 1px solid rgba(99,179,237,0.25); border-radius: 14px; padding: 1.5rem; margin-bottom: 0.5rem;
}
.hero-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #63b3ed; margin-bottom: 8px; }
.hero-text  { font-size: 1.2rem; color: white; line-height: 1.8; word-wrap: break-word; white-space: pre-wrap; }
.pronunciation { font-size: 0.9rem; color: rgba(255,255,255,0.45); margin-top: 10px; font-style: italic; letter-spacing: 0.02em; }

.variant-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px; padding: 1.1rem; margin-bottom: 0.5rem;
}
.variant-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: rgba(255,255,255,0.4); margin-bottom: 8px; }
.variant-text  { font-size: 1rem; color: rgba(255,255,255,0.85); line-height: 1.8; word-wrap: break-word; white-space: pre-wrap; }
.variant-pron  { font-size: 12px; color: rgba(255,255,255,0.35); margin-top: 8px; font-style: italic; }
.variant-hint  { font-size: 12px; color: rgba(255,255,255,0.25); margin-top: 4px; }

.literal-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1.1rem; margin-bottom: 0.5rem;
}
.literal-text {
    font-size: 0.95rem; color: rgba(255,255,255,0.75); line-height: 2.2;
    word-wrap: break-word; overflow-wrap: break-word; white-space: normal; word-break: break-word;
}

.section-header {
    font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    color: rgba(255,255,255,0.3); margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 6px; border-bottom: 1px solid rgba(255,255,255,0.06);
}

.word-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 14px; text-align: center;
}
.word-main   { font-size: 1.3rem; color: white; font-weight: 500; margin-bottom: 4px; }
.word-rom    { font-size: 12px; color: rgba(255,255,255,0.4); font-style: italic; margin-bottom: 6px; }
.word-pos    { display: inline-block; background: rgba(99,179,237,0.15); color: #63b3ed; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px; margin-bottom: 6px; }
.word-mean   { font-size: 13px; color: rgba(255,255,255,0.7); margin-bottom: 6px; }
.word-note   { font-size: 11px; color: rgba(255,255,255,0.3); }

.flashcard {
    background: linear-gradient(135deg, rgba(118,75,162,0.15), rgba(102,126,234,0.15));
    border: 1px solid rgba(118,75,162,0.25); border-radius: 12px; padding: 16px;
}
.fc-front    { font-size: 1.2rem; color: white; font-weight: 500; margin-bottom: 4px; }
.fc-rom      { font-size: 12px; color: rgba(255,255,255,0.4); font-style: italic; margin-bottom: 8px; }
.fc-back     { font-size: 13px; color: rgba(255,255,255,0.7); margin-bottom: 8px; }
.fc-example  { font-size: 12px; color: rgba(255,255,255,0.45); border-left: 2px solid rgba(118,75,162,0.5); padding-left: 8px; }

.cefr-badge  { display: inline-block; padding: 3px 12px; border-radius: 12px; font-size: 12px; font-weight: 700; letter-spacing: 0.05em; }
.model-badge { display: inline-flex; align-items: center; gap: 5px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 3px 10px; font-size: 11px; color: rgba(255,255,255,0.4); }

.conf-bar-bg   { background: rgba(255,255,255,0.08); border-radius: 4px; height: 5px; margin-top: 4px; overflow: hidden; width: 80px; }
.conf-bar-fill { height: 5px; border-radius: 4px; background: linear-gradient(90deg, #48bb78, #63b3ed); }

.stButton > button {
    background: linear-gradient(135deg, #3182ce, #2b6cb0) !important; color: white !important;
    border: none !important; border-radius: 10px !important; padding: 0.6rem 2rem !important;
    font-weight: 500 !important; box-shadow: 0 4px 15px rgba(49,130,206,0.3) !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(49,130,206,0.4) !important; }

section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid rgba(255,255,255,0.07) !important; }
.streamlit-expanderHeader { background: rgba(255,255,255,0.03) !important; border-radius: 8px !important; border: 1px solid rgba(255,255,255,0.07) !important; }
[data-testid="metric-container"] { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 0.8rem 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
LANGUAGE_OPTIONS = {
    "🇬🇧 English":           "en",
    "🇫🇷 French":            "fr",
    "🇪🇸 Spanish":           "es",
    "🇯🇵 Japanese":          "ja",
    "🇮🇳 Hindi":             "hi",
    "🇸🇦 Arabic":            "ar",
    "🇨🇳 Mandarin Chinese":  "zh",
    "🇩🇪 German":            "de",
    "🇹🇿 Swahili":           "sw",
    "🇧🇷 Portuguese (BR)":   "pt",
    "🇰🇷 Korean":            "ko",
}

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

CEFR_COLORS = {
    "A1": "#68d391", "A2": "#68d391",
    "B1": "#63b3ed", "B2": "#63b3ed",
    "C1": "#f6ad55", "C2": "#fc8181",
}

TONE_CONFIG = {
    "formal":  {"icon": "🎩", "label": "FORMAL",  "color": "#63b3ed", "bg": "rgba(49,130,206,0.08)",  "border": "rgba(49,130,206,0.3)",  "badge_bg": "rgba(49,130,206,0.2)",  "hint": "Use with superiors, clients, strangers, or official documents."},
    "neutral": {"icon": "💬", "label": "NEUTRAL", "color": "#a0aec0", "bg": "rgba(160,174,192,0.06)", "border": "rgba(160,174,192,0.2)", "badge_bg": "rgba(160,174,192,0.15)","hint": "Appropriate for most everyday professional interactions."},
    "casual":  {"icon": "😊", "label": "CASUAL",  "color": "#f6ad55", "bg": "rgba(246,173,85,0.06)",  "border": "rgba(246,173,85,0.25)", "badge_bg": "rgba(246,173,85,0.15)", "hint": "Use with friends, family, or people you know very well."},
}

# Words that make gender explicit — if present, no gendered variants needed
EXPLICIT_GENDER_WORDS = {
    # Explicit male
    "sir", "mr", "him", "his", "he", "man", "boy", "gentleman", "husband",
    "father", "dad", "brother", "son", "uncle", "grandfather", "king", "prince",
    # Explicit female
    "ma'am", "madam", "ms", "mrs", "miss", "her", "she", "woman", "girl",
    "lady", "wife", "mother", "mom", "sister", "daughter", "aunt",
    "grandmother", "queen", "princess",
}

# Languages where gender affects translation
GENDERED_LANGS = {"fr", "es", "de", "hi", "ar", "pt", "ko", "ja"}


# ── Input validation ───────────────────────────────────────────────────────────
def validate_input(text: str) -> tuple[bool, str]:
    """
    Reject garbage strings while accepting broken/typo real language.

    Rejects:  dkjerncxi4, asdfghjkl, zxcvbnm qwerty, 123456789
    Accepts:  hew are youu, lol omg wtf, holy shit, strength training,
              こんにちは, مرحبا (foreign scripts always pass)

    Three rules (applied per meaningful word of 4+ letters):
      1. 5+ char word with zero vowels → garbage
      2. 8+ char word where longest consonant cluster ≥ 55% of word → garbage
      3. Word starting with 3 consonants where that onset is not a known
         valid English onset (str, spr, spl, scr, shr, thr, sch, squ) → garbage

    If ≥50% of meaningful words are garbage → reject the input.
    """
    import re as _re

    VALID_TRIPLE_ONSETS = {'str', 'spr', 'spl', 'scr', 'shr', 'thr', 'sch', 'squ', 'stn'}

    text = text.strip()

    if not text:
        return False, "Please enter some text to translate."
    if len(text) < 2:
        return False, "Input is too short. Please enter at least a word or two."
    if not any(c.isalpha() for c in text):
        return False, "Input contains no words. Please enter a sentence."
    if len(text) <= 4:
        return True, ""

    # Non-ASCII dominant → foreign script → always valid
    ascii_chars = [c for c in text if ord(c) < 128]
    if len(ascii_chars) <= len(text) * 0.5:
        return True, ""

    # Letter ratio check
    letters = sum(1 for c in text if c.isalpha())
    total   = len(text.replace(" ", ""))
    if total > 0 and (letters / total) < 0.4 and len(text) > 4:
        return False, "This doesn't look like translatable text. Please enter a real sentence."

    words = text.lower().split()
    garbage_count = 0

    for word in words:
        clean = _re.sub(r'[^a-z]', '', word)
        if len(clean) < 4:
            continue

        vowels   = sum(1 for c in clean if c in 'aeiou')
        clusters = _re.findall(r'[^aeiou]+', clean)
        max_c    = max((len(c) for c in clusters), default=0)

        # Rule 1: 5+ char word with zero vowels
        if len(clean) >= 5 and vowels == 0:
            garbage_count += 1
            continue

        # Rule 2: 8+ char word where longest cluster ≥ 55% of word length
        if len(clean) >= 8 and max_c / len(clean) >= 0.55:
            garbage_count += 1
            continue

        # Rule 3: starts with 3 consonants that aren't a known English onset
        if len(clean) >= 6:
            first_vowel = next((i for i, c in enumerate(clean) if c in 'aeiou'), len(clean))
            if first_vowel >= 3 and clean[:3] not in VALID_TRIPLE_ONSETS:
                garbage_count += 1
                continue

    meaningful = [w for w in words if len(_re.sub(r'[^a-z]', '', w)) >= 4]
    if meaningful and garbage_count >= len(meaningful) * 0.5:
        return False, "This looks like random characters. Please enter something to translate."

    return True, ""


def check_gender_needed(text: str, target_lang: str) -> bool:
    """
    Returns True if gendered variants should be shown.

    Logic:
    - Only applies to gendered languages
    - If the sentence contains explicit gender markers (sir/ma'am/he/she etc.),
      gender is already clear → no variants needed
    - Otherwise, any sentence directed at or talking about a person
      should show gendered variants, because "how are you" in Japanese
      differs based on who you're addressing and who is speaking
    - Exception: clearly impersonal sentences (weather, facts, objects)
    """
    if target_lang not in GENDERED_LANGS:
        return False

    text_lower = text.lower()
    # Strip punctuation for word matching
    words = set(re.sub(r'[^a-z\s]', '', text_lower).split())

    # If explicit gender word present, gender is clear → no variants
    if words & EXPLICIT_GENDER_WORDS:
        return False

    # Impersonal patterns — no gendered variants needed
    impersonal_patterns = [
        r'\b(it is|it\'s|there is|there are|the weather|today is|tomorrow)\b',
        r'\b(what time|how much|how many|where is|when is)\b',
    ]
    for pattern in impersonal_patterns:
        if re.search(pattern, text_lower):
            return False

    # If the sentence is clearly about people or directed at someone →
    # gendered variants should be shown for gendered languages
    # This is the default for most conversational sentences
    person_indicators = [
        r'\b(you|your|yours|yourself)\b',          # addressing someone
        r'\b(they|them|their|theirs)\b',            # referring to someone
        r'\b(friend|colleague|teacher|doctor|student|person|someone|anybody|everyone)\b',
        r'\b(i am|i feel|i was|i\'m|i\'ve)\b',     # self-reference (differs by speaker gender)
        r'\b(please|thank you|sorry|excuse me|hello|hi|good morning|good evening)\b',  # greetings/politeness
        r'\b(can you|could you|would you|will you|may i|shall we)\b',  # requests
        r'\b(how are you|are you|do you|did you|have you|were you)\b',  # questions to someone
        r'\b(meet|see|talk|speak|visit|help|ask|tell|show|send|give|bring)\b',  # interpersonal verbs
    ]

    for pattern in person_indicators:
        if re.search(pattern, text_lower):
            return True

    # If sentence is short (1-5 words) and contains common words, likely conversational
    word_list = text_lower.split()
    if len(word_list) <= 6:
        common_words = {'hello', 'hi', 'bye', 'thanks', 'sorry', 'please',
                        'yes', 'no', 'ok', 'okay', 'sure', 'fine', 'good', 'great'}
        if words & common_words:
            return True

    return False


# ── Session state ──────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "session_history": {
            "preferred_formality": None,
            "csi_categories_seen": [],
            "active_warnings": [],
            "turn_count": 0,
        },
        "translation_history": [],
        "last_translation":    None,
        "last_learning_input": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div style='padding:1rem 0 0.5rem;'>"
            "<div style='font-size:1.1rem;font-weight:600;color:white;'>🌏 CulturaTrans</div>"
            "<div style='font-size:11px;color:rgba(255,255,255,0.3);margin-top:2px;'>Culturally-Aware Translation</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("<div style='font-size:11px;font-weight:700;color:rgba(255,255,255,0.4);text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;'>Languages</div>", unsafe_allow_html=True)
        source_lang_name = st.selectbox("From", list(LANGUAGE_OPTIONS.keys()), index=0,  label_visibility="collapsed")
        st.markdown("<div style='text-align:center;color:rgba(255,255,255,0.25);font-size:20px;margin:2px 0;'>↓</div>", unsafe_allow_html=True)
        target_lang_name = st.selectbox("To",   list(LANGUAGE_OPTIONS.keys()), index=3, label_visibility="collapsed")
        st.divider()

        st.markdown("<div style='font-size:11px;font-weight:700;color:rgba(255,255,255,0.4);text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;'>Learning Mode</div>", unsafe_allow_html=True)
        learning_mode = st.toggle("Enable Learning Mode", value=False)
        cefr_level    = st.select_slider("Level", options=CEFR_LEVELS, value="B2",
                                         disabled=not learning_mode, label_visibility="collapsed")
        if learning_mode:
            st.caption(f"Declared level: **{cefr_level}** — output adapts to your proficiency")
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Turns", st.session_state.session_history.get("turn_count", 0))
        with c2:
            current_reg = st.session_state.session_history.get("preferred_formality") or "—"
            reg_colors  = {"formal": "#63b3ed", "casual": "#f6ad55", "neutral": "#a0aec0"}
            reg_color   = reg_colors.get(current_reg, "#a0aec0")
            st.markdown(
                f"<div style='font-size:11px;color:rgba(255,255,255,0.5);'>Register</div>"
                f"<div style='font-size:1.3rem;font-weight:600;color:{reg_color};'>{current_reg}</div>",
                unsafe_allow_html=True,
            )

        if st.button("↺ Reset session", use_container_width=True):
            st.session_state.session_history = {
                "preferred_formality": None,
                "csi_categories_seen": [],
                "active_warnings": [],
                "turn_count": 0,
            }
            st.session_state.translation_history = []
            st.session_state.last_translation    = None
            st.rerun()

        st.divider()
        st.markdown("<div style='font-size:11px;font-weight:700;color:rgba(255,255,255,0.4);text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;'>History</div>", unsafe_allow_html=True)
        if st.session_state.translation_history:
            # Show ALL history, newest first
            for item in reversed(st.session_state.translation_history):
                with st.expander(f"↳ {item['input'][:30]}…" if len(item['input']) > 30 else f"↳ {item['input']}", expanded=False):
                    st.caption(f"→ {item['target_lang']}")
                    st.write(item.get("recommended", "")[:120])
        else:
            st.markdown("<div style='color:rgba(255,255,255,0.2);font-size:12px;padding:4px 0;'>No translations yet</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("<div style='font-size:11px;color:rgba(255,255,255,0.2);'>PES University · GenAI · Team 15</div>", unsafe_allow_html=True)

    src      = LANGUAGE_OPTIONS[source_lang_name]
    tgt      = LANGUAGE_OPTIONS[target_lang_name]
    src_name = source_lang_name.split(" ", 1)[1] if " " in source_lang_name else source_lang_name
    tgt_name = target_lang_name.split(" ", 1)[1] if " " in target_lang_name else target_lang_name
    return src, tgt, src_name, tgt_name, learning_mode, cefr_level


# ── Gender tab renderer ───────────────────────────────────────────────────────
def _render_gender_tab(text_val: str, pron_val: str, hint: str):
    """
    Renders a single gender variant tab with:
      - Target language script (kanji / Arabic / Devanagari etc.) in large text
      - Romanized pronunciation below in italic
      - Hint label at bottom
    Built as separate string variables to avoid nested f-string rendering issues.
    """
    # Always render the text div — even if empty, the structure must be present
    text_content = text_val if text_val and text_val.strip() else "(translation not available)"

    kanji_div = (
        '<div class="hero-text" style="font-size:1.25rem;color:white;'
        'line-height:1.9;word-wrap:break-word;margin-bottom:6px;">'
        + text_content
        + '</div>'
    )

    pron_div = ""
    if pron_val and pron_val.strip():
        pron_div = (
            '<div style="font-size:0.9rem;color:rgba(255,255,255,0.45);'
            'margin-top:8px;font-style:italic;letter-spacing:0.02em;">'
            '🔊 ' + pron_val.strip()
            + '</div>'
        )

    hint_div = (
        '<div style="font-size:11px;color:rgba(255,255,255,0.2);margin-top:10px;">'
        + hint
        + '</div>'
    )

    full_html = (
        '<div class="hero-translation">'
        + kanji_div
        + pron_div
        + hint_div
        + '</div>'
    )

    st.markdown(full_html, unsafe_allow_html=True)


# ── Output renderer ────────────────────────────────────────────────────────────
def render_output(translation, learning_input, learning_mode, cefr_level, target_lang):
    detected    = translation.get("detected_formality", "neutral")
    reason      = translation.get("tone_recommendation_reason", "")
    confidence  = translation.get("confidence", 0.85)
    model_used  = translation.get("model_used", "")
    cefr_out    = translation.get("cefr_level", "B1")
    gender_note = translation.get("gender_note", "")
    detected_src= translation.get("_detected_source_formality", detected)

    # Use app-side gender detection — more reliable than LLM detection
    is_gendered = bool(translation.get("is_gendered", False))

    tc         = TONE_CONFIG.get(detected_src, TONE_CONFIG["neutral"])
    cefr_color = CEFR_COLORS.get(cefr_out, "#a0aec0")
    src_icons  = {"formal": "🎩", "casual": "😊", "neutral": "💬"}
    src_colors = {"formal": "#63b3ed", "casual": "#f6ad55", "neutral": "#a0aec0"}

    src_html = (
        "<span style='font-size:11px;color:rgba(255,255,255,0.4);'>Source detected as: </span>"
        "<span style='font-size:12px;font-weight:600;color:"
        + src_colors.get(detected_src, "#a0aec0")
        + ";'>"
        + src_icons.get(detected_src, "")
        + " "
        + detected_src.upper()
        + "</span>"
    ) if detected_src else ""

    model_span = (
        "<span class='model-badge'>⚡ " + model_used + "</span>"
    ) if model_used else ""

    # ── Register banner ──────────────────────────────────────────────────────
    banner_html = (
        '<div class="tone-card" style="background:' + tc["bg"] + ';border-color:' + tc["border"] + ';">'
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px;">'
        '<div style="flex:1;">'
        '<div style="margin-bottom:6px;">' + src_html + '</div>'
        '<div class="tone-badge" style="background:' + tc["badge_bg"] + ';color:' + tc["color"] + ';">'
        + tc["icon"] + ' Register: ' + detected_src.upper()
        + '</div>'
        '<div style="color:rgba(255,255,255,0.65);font-size:0.88rem;margin-top:4px;line-height:1.6;">' + reason + '</div>'
        '</div>'
        '<div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px;">'
        '<div style="text-align:right;">'
        '<div style="font-size:10px;color:rgba(255,255,255,0.3);margin-bottom:2px;">CONFIDENCE</div>'
        '<div style="font-size:1.4rem;font-weight:600;color:' + tc["color"] + ';">' + f'{confidence:.0%}' + '</div>'
        '<div class="conf-bar-bg"><div class="conf-bar-fill" style="width:' + f'{confidence*100:.0f}' + '%;"></div></div>'
        '</div>'
        '<div style="display:flex;gap:6px;align-items:center;">'
        '<span class="cefr-badge" style="background:' + cefr_color + '22;color:' + cefr_color + ';">CEFR ' + cefr_out + '</span>'
        + model_span
        + '</div></div></div></div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)

    # ── Profanity / explicit content warning ─────────────────────────────────
    profanity_flag = translation.get("profanity_flag", {})
    if profanity_flag and profanity_flag.get("found"):
        level = profanity_flag.get("level", "profanity")
        icon  = "⚠️" if level == "profanity" else "🔞" if level == "explicit" else "⚠️🔞"
        label = (
            "Offensive / profane language"  if level == "profanity" else
            "Sexually explicit content"      if level == "explicit"  else
            "Offensive and explicit content"
        )
        terms = profanity_flag.get("terms_found", [])
        terms_display = ", ".join(f'"{t}"' for t in terms[:4])
        if len(terms) > 4:
            terms_display += f" +{len(terms)-4} more"
        st.markdown(
            '<div style="background:rgba(246,173,85,0.08);border:1px solid rgba(246,173,85,0.35);'
            'border-radius:10px;padding:12px 16px;margin-bottom:10px;">'
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            '<span style="font-size:15px;">' + icon + '</span>'
            '<span style="color:#f6ad55;font-weight:600;font-size:13px;">' + label + '</span>'
            '</div>'
            '<div style="color:rgba(255,255,255,0.6);font-size:12px;margin-bottom:4px;">'
            + profanity_flag.get("message", "") + '</div>'
            '<div style="color:rgba(255,255,255,0.35);font-size:11px;">Detected: ' + terms_display + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Sensitivity warnings ──────────────────────────────────────────────────
    for flag in translation.get("sensitivity_flags", []):
        sc = {"high": "#fc8181", "medium": "#f6ad55", "low": "#68d391"}.get(flag["severity"], "#a0aec0")
        st.markdown(
            '<div style="background:' + sc + '11;border:1px solid ' + sc + '44;border-radius:10px;'
            'padding:12px 16px;margin-bottom:8px;">'
            '<span style="color:' + sc + ';font-weight:600;">⚠ "' + flag["span"] + '"</span>'
            '<span style="color:rgba(255,255,255,0.5);font-size:13px;"> — ' + flag["warning_type"] + '</span>'
            '<div style="color:rgba(255,255,255,0.6);font-size:13px;margin-top:4px;">💡 ' + flag["suggestion"] + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Gender note ───────────────────────────────────────────────────────────
    if is_gendered and gender_note:
        st.markdown(
            '<div style="background:rgba(159,122,234,0.08);border:1px solid rgba(159,122,234,0.25);'
            'border-radius:10px;padding:12px 16px;margin-bottom:12px;">'
            '<span style="color:#b794f4;font-weight:600;">⚥ Gender-sensitive translation</span>'
            '<div style="color:rgba(255,255,255,0.65);font-size:0.88rem;margin-top:4px;">' + gender_note + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Translations ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Translations</div>", unsafe_allow_html=True)

    if is_gendered:
        # Get all four gender variants — fall back to non-gendered if LLM didn't provide
        fm  = translation.get("formal_male")   or translation.get("formal", "")
        ff  = translation.get("formal_female") or translation.get("formal", "")
        cm  = translation.get("casual_male")   or translation.get("casual", "")
        cf  = translation.get("casual_female") or translation.get("casual", "")
        fmp = translation.get("formal_male_pronunciation")   or translation.get("formal_pronunciation", "")
        ffp = translation.get("formal_female_pronunciation") or translation.get("formal_pronunciation", "")
        cmp = translation.get("casual_male_pronunciation")   or translation.get("casual_pronunciation", "")
        cfp = translation.get("casual_female_pronunciation") or translation.get("casual_pronunciation", "")

        tab_fm, tab_ff, tab_cm, tab_cf = st.tabs([
            "🎩♂ Formal (Male)", "🎩♀ Formal (Female)",
            "😊♂ Casual (Male)", "😊♀ Casual (Female)",
        ])

        with tab_fm:
            _render_gender_tab(fm, fmp, "Formal register — addressing/referring to a male")
        with tab_ff:
            _render_gender_tab(ff, ffp, "Formal register — addressing/referring to a female")
        with tab_cm:
            _render_gender_tab(cm, cmp, "Casual register — addressing/referring to a male")
        with tab_cf:
            _render_gender_tab(cf, cfp, "Casual register — addressing/referring to a female")

    else:
        # Standard: formal hero + casual card
        formal_text = translation.get("formal", "")
        casual_text = translation.get("casual", "")
        fp          = translation.get("formal_pronunciation", "")
        cp          = translation.get("casual_pronunciation", "")

        formal_pron_div = (
            '<div class="pronunciation">🔊 ' + fp + '</div>'
        ) if fp else ""

        st.markdown(
            '<div class="hero-translation">'
            '<div class="hero-label">' + tc["icon"] + ' ' + detected_src.upper() + ' — Formal variant</div>'
            '<div class="hero-text">' + formal_text + '</div>'
            + formal_pron_div
            + '<div style="font-size:11px;color:rgba(255,255,255,0.2);margin-top:6px;">'
            + TONE_CONFIG["formal"]["hint"]
            + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

        casual_pron_div = (
            '<div class="variant-pron">🔊 ' + cp + '</div>'
        ) if cp else ""

        st.markdown(
            '<div class="variant-card">'
            '<div class="variant-label">😊 CASUAL</div>'
            '<div class="variant-text">' + casual_text + '</div>'
            + casual_pron_div
            + '<div class="variant-hint">' + TONE_CONFIG["casual"]["hint"] + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Literal breakdown ─────────────────────────────────────────────────────
    literal_raw  = translation.get("literal", "")
    literal_pron = translation.get("literal_pronunciation", "")

    literal_words = re.sub(r'\[pronunciation:.*?\]', '', literal_raw, flags=re.IGNORECASE).strip()
    literal_words = literal_words.replace("\u00b7", " \u00b7 ")

    literal_pron_div = (
        '<div style="font-size:12px;color:rgba(255,255,255,0.4);'
        'font-style:italic;margin-top:8px;">🔊 ' + literal_pron + '</div>'
    ) if literal_pron else ""

    literal_hint_div = (
        '<div style="font-size:12px;color:rgba(255,255,255,0.25);margin-top:6px;">'
        'Each word shown with English meaning in brackets. Reveals grammatical structure.'
        '</div>'
    )

    st.markdown(
        '<div class="literal-card" style="margin-top:8px;">'
        '<div class="variant-label">📖 LITERAL — word-by-word breakdown</div>'
        '<div class="literal-text">' + literal_words + '</div>'
        + literal_pron_div
        + literal_hint_div
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── Cultural notes ────────────────────────────────────────────────────────
    if translation.get("cultural_notes"):
        st.markdown("<div class='section-header'>Cultural Intelligence</div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.18);'
            'border-radius:10px;padding:14px 16px;color:rgba(255,255,255,0.7);'
            'font-size:0.88rem;line-height:1.75;">🌐 '
            + translation["cultural_notes"]
            + '</div>',
            unsafe_allow_html=True,
        )

    # ── CSI spans ─────────────────────────────────────────────────────────────
    if translation.get("csi_spans"):
        with st.expander(f"🔍 Culture-Specific Items ({len(translation['csi_spans'])} detected)"):
            for span in translation["csi_spans"]:
                st.markdown(
                    f"**\"{span['span']}\"** — *{span['category']}*\n\n{span['explanation']}"
                )
                st.divider()

    # ── CoT reasoning ─────────────────────────────────────────────────────────
    with st.expander("🧠 Chain-of-Thought Reasoning"):
        st.markdown(
            "<div style='color:rgba(255,255,255,0.4);font-size:12px;margin-bottom:10px;'>"
            "Step-by-step reasoning about cultural adaptation. "
            "This transparency is a core paper contribution.</div>",
            unsafe_allow_html=True,
        )
        reasoning = translation.get("cot_reasoning", "No reasoning available.")
        steps = reasoning.split("STEP ")
        if len(steps) > 1:
            for step in steps[1:]:
                num = step[0] if step else "?"
                txt = step[2:].strip() if len(step) > 2 else step
                st.markdown(f"**Step {num}:** {txt}")
        else:
            st.markdown(
                '<div style="color:rgba(255,255,255,0.65);">' + reasoning + '</div>',
                unsafe_allow_html=True,
            )

    with st.expander("🔬 Word Alignment Heatmap (MarianMT)"):
        st.markdown(
            "<div style='color:rgba(255,255,255,0.4);font-size:12px;margin-bottom:8px;'>"
            "Shows which source words align to which target words. "
            "Uses Helsinki-NLP MarianMT for interpretability — "
            "Claude's actual translation may differ slightly. "
            "Brighter = stronger attention weight.</div>",
            unsafe_allow_html=True,
        )
        try:
            from eval.attention_heatmap import render_attention_heatmap
            import streamlit.components.v1 as components
            src_text = translation.get("source_text", "")
            if src_text:
                with st.spinner("Loading alignment model (first load takes ~30s)..."):
                    heatmap_html = render_attention_heatmap(src_text, target_lang)
                components.html(heatmap_html, height=460, scrolling=False)
            else:
                st.info("Source text not available.")
        except ImportError:
            st.info("Heatmap needs: pip install transformers sentencepiece")
        except Exception as e:
            st.info(f"Heatmap unavailable for this language pair: {e}")

    # ── Learning mode ─────────────────────────────────────────────────────────
    if learning_mode:
        st.markdown("<div class='section-header'>Learning Mode</div>", unsafe_allow_html=True)
        _render_learning_mode(translation, cefr_level, cefr_out)


# ── Learning mode renderer ────────────────────────────────────────────────────
def _render_learning_mode(translation, user_cefr: str, output_cefr: str):
    CEFR_ORDER  = ["A1", "A2", "B1", "B2", "C1", "C2"]
    user_idx    = CEFR_ORDER.index(user_cefr)   if user_cefr   in CEFR_ORDER else 3
    output_idx  = CEFR_ORDER.index(output_cefr) if output_cefr in CEFR_ORDER else 3
    cefr_color  = CEFR_COLORS.get(output_cefr, "#a0aec0")
    user_color  = CEFR_COLORS.get(user_cefr,   "#a0aec0")

    if output_idx > user_idx:
        diff_msg   = f"⚠️ This translation ({output_cefr}) is above your declared level ({user_cefr}). The casual variant has been simplified."
        diff_color = "#f6ad55"
    elif output_idx == user_idx:
        diff_msg   = f"✓ This translation ({output_cefr}) matches your declared level ({user_cefr}). Good fit for your proficiency."
        diff_color = "#68d391"
    else:
        diff_msg   = f"✓ This translation ({output_cefr}) is below your declared level ({user_cefr}). Should be comfortable to read."
        diff_color = "#63b3ed"

    st.markdown(
        '<div style="background:' + diff_color + '11;border:1px solid ' + diff_color + '44;'
        'border-radius:10px;padding:14px 16px;margin-bottom:1rem;">'
        '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
        '<div style="color:rgba(255,255,255,0.75);font-size:0.88rem;">' + diff_msg + '</div>'
        '<div style="display:flex;gap:8px;align-items:center;">'
        '<span style="font-size:11px;color:rgba(255,255,255,0.35);">Output:</span>'
        '<span class="cefr-badge" style="background:' + cefr_color + '22;color:' + cefr_color + ';">' + output_cefr + '</span>'
        '<span style="font-size:11px;color:rgba(255,255,255,0.35);">Your level:</span>'
        '<span class="cefr-badge" style="background:' + user_color + '22;color:' + user_color + ';">' + user_cefr + '</span>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📖 Word Breakdown", "🗂 Flashcards", "📤 Export"])

    with tab1:
        word_breakdown = translation.get("word_breakdown", [])
        if word_breakdown:
            cols = st.columns(min(len(word_breakdown), 4))
            for i, word in enumerate(word_breakdown[:4]):
                with cols[i % 4]:
                    st.markdown(
                        '<div class="word-card">'
                        '<div class="word-main">' + word.get("word","") + '</div>'
                        '<div class="word-rom">' + word.get("romanization","") + '</div>'
                        '<div><span class="word-pos">' + word.get("pos","") + '</span></div>'
                        '<div class="word-mean">' + word.get("meaning","") + '</div>'
                        '<div class="word-note">' + word.get("note","") + '</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Word breakdown not available for this translation.")

    with tab2:
        flashcards = translation.get("flashcards", [])
        if flashcards:
            st.markdown(
                "<div style='color:rgba(255,255,255,0.5);font-size:12px;margin-bottom:12px;'>"
                "Tap any card to see the example sentence. Export all cards in the Export tab.</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(min(len(flashcards), 3))
            for i, card in enumerate(flashcards[:3]):
                with cols[i % 3]:
                    st.markdown(
                        '<div class="flashcard">'
                        '<div class="fc-front">' + card.get("front","") + '</div>'
                        '<div class="fc-rom">' + card.get("pronunciation","") + '</div>'
                        '<div class="fc-back">→ ' + card.get("back","") + '</div>'
                        '<div class="fc-example">'
                        + card.get("example","")
                        + '<br><span style="color:rgba(255,255,255,0.3);">'
                        + card.get("example_translation","")
                        + '</span></div></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Flashcards not available for this translation.")

    with tab3:
        flashcards = translation.get("flashcards", [])
        if flashcards:
            st.markdown(
                "<div style='color:rgba(255,255,255,0.6);font-size:13px;margin-bottom:12px;'>"
                "Download as Anki-compatible JSON. Import using the CrowdAnki add-on.</div>",
                unsafe_allow_html=True,
            )
            anki_export = []
            for card in flashcards:
                anki_export.append({
                    "Front": card.get("front","") + " <br><i>" + card.get("pronunciation","") + "</i>",
                    "Back":  (
                        card.get("back","") + "<br><br>"
                        "<i>" + card.get("example","") + "</i><br>"
                        + card.get("example_translation","")
                    ),
                    "Tags": ["culturatrans", translation.get("target_lang",""), translation.get("cefr_level","")],
                })
            json_str = json.dumps(anki_export, ensure_ascii=False, indent=2)
            st.download_button(
                label="⬇️ Download Anki JSON",
                data=json_str,
                file_name="culturatrans_" + translation.get("target_lang","xx") + "_flashcards.json",
                mime="application/json",
                use_container_width=True,
            )
            st.code(json_str[:400] + ("..." if len(json_str) > 400 else ""), language="json")
        else:
            st.info("No flashcards to export.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_session()
    source_lang, target_lang, source_name, target_name, learning_mode, cefr_level = render_sidebar()

    st.markdown(
        '<div class="main-header">'
        '<h1>🌏 CulturaTrans</h1>'
        '<p>Culturally-aware translation with register intelligence &mdash; PES University GenAI Project, Team 15</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if source_lang == target_lang:
        st.warning("⚠️ Source and target languages are the same. Please select different languages.")
        return

    col_input, col_meta = st.columns([3, 1])
    with col_input:
        st.markdown(
            "<div style='font-size:12px;font-weight:700;color:rgba(255,255,255,0.4);"
            "text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;'>"
            "Text in " + source_name + "</div>",
            unsafe_allow_html=True,
        )
        user_input = st.text_area(
            "input",
            placeholder="Type or paste text in " + source_name + "…",
            height=140,
            label_visibility="collapsed",
        )

    with col_meta:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        learning_line = "<br>📚 Learning mode ON" if learning_mode else ""
        st.markdown(
            '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
            'border-radius:10px;padding:14px;font-size:12px;color:rgba(255,255,255,0.4);line-height:2;">'
            '📤 <strong style="color:rgba(255,255,255,0.65)">' + source_name + '</strong><br>'
            '📥 <strong style="color:rgba(255,255,255,0.65)">' + target_name + '</strong><br>'
            '🔄 3 register variants<br>'
            '🔊 Pronunciation guide<br>'
            '🧠 Cultural tone advice'
            + learning_line
            + '</div>',
            unsafe_allow_html=True,
        )

    col_btn, col_clr, _ = st.columns([2, 1, 4])
    with col_btn:
        go = st.button("🌐  Translate", type="primary", use_container_width=True)
    with col_clr:
        if st.button("✕ Clear", use_container_width=True):
            st.session_state.last_translation = None
            st.rerun()

    if go:
        # ── Input validation ──────────────────────────────────────────────────
        is_valid, error_msg = validate_input(user_input)
        if not is_valid:
            st.error("⚠️ " + error_msg)
            return

        # ── App-side gender detection ─────────────────────────────────────────
        # Do this before calling the API so we can override LLM's decision
        gender_needed = check_gender_needed(user_input, target_lang)

        with st.spinner("Translating to " + target_name + "…"):
            try:
                from pipeline.variant_formatter import run
                translation, learning_input = run(
                    text=user_input,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    user_cefr_level=cefr_level,
                    session_history=st.session_state.session_history,
                    modality="text",
                )

                # Override is_gendered with our more reliable app-side check
                translation["is_gendered"] = gender_needed

                # Store detected source formality for display
                from pipeline.formality_classifier import classify_formality
                detected_formality, _ = classify_formality(user_input, source_lang)
                translation["_detected_source_formality"] = detected_formality

                st.session_state.last_translation    = translation
                st.session_state.last_learning_input = learning_input

                rec_tone = translation.get("detected_formality", "neutral")
                st.session_state.translation_history.append({
                    "input":       user_input,
                    "target_lang": target_name,
                    "recommended": translation.get(rec_tone, translation.get("formal", "")),
                })

            except EnvironmentError as e:
                st.error("❌ **API Key Error**\n\n" + str(e))
                return
            except Exception as e:
                st.error("❌ Translation failed: " + str(e))
                logging.exception("Translation error")
                return

    if st.session_state.last_translation is not None:
        st.markdown(
            "<hr style='border-color:rgba(255,255,255,0.06);margin:1.5rem 0;'>",
            unsafe_allow_html=True,
        )
        render_output(
            st.session_state.last_translation,
            st.session_state.last_learning_input,
            learning_mode,
            cefr_level,
            target_lang,
        )


if __name__ == "__main__":
    main()
