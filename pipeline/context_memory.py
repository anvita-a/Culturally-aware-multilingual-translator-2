"""
context_memory.py
-----------------
Asmi's Module 3 — Context Memory

Manages session state across dialogue turns so the system behaves as a
coherent cultural interpreter rather than treating each translation in isolation.

What it tracks:
  preferred_formality — once two consecutive turns use the same formality,
                         it is locked in and passed to the LLM as a preference
  csi_categories_seen — accumulates across turns for learning analytics
  active_warnings     — high-severity sensitivity flags carry forward
  turn_count          — simple turn counter

Why it matters:
  Without this, every translation starts fresh. With it, if the user writes
  formally for two turns in a row, turn 3 auto-uses formal register even for
  a vague input. CSI categories seen earlier surface in later turns.
"""

import logging
from typing import List, Optional

from pipeline.interfaces import SessionHistory, SensitivityFlag, CSISpan

logger = logging.getLogger(__name__)

# How many consecutive turns with the same formality before it becomes "preferred"
FORMALITY_CONSISTENCY_THRESHOLD = 2


def create_session() -> SessionHistory:
    """Create a fresh SessionHistory for a new conversation."""
    return {
        "preferred_formality":  None,
        "csi_categories_seen":  [],
        "active_warnings":      [],
        "turn_count":           0,
        # Internal streak counter — not in the TypedDict but stored as extra key
        "_formality_streak":    0,
        "_last_formality":      None,
    }


def update_session(
    session: SessionHistory,
    detected_formality: str,
    csi_spans: List[CSISpan],
    sensitivity_flags: List[SensitivityFlag],
) -> SessionHistory:
    """
    Update session history after each turn.

    Args:
        session:             Current session state (mutated and returned)
        detected_formality:  Formality detected this turn ("formal"|"neutral"|"casual")
        csi_spans:           CSI spans detected this turn
        sensitivity_flags:   Sensitivity flags detected this turn

    Returns:
        Updated SessionHistory
    """
    # Increment turn count
    session["turn_count"] = session.get("turn_count", 0) + 1

    # ── Formality preference tracking ─────────────────────────────────────────
    last = session.get("_last_formality")
    streak = session.get("_formality_streak", 0)

    if detected_formality == last:
        streak += 1
    else:
        streak = 1

    session["_last_formality"]   = detected_formality
    session["_formality_streak"] = streak

    # Lock in preference once threshold is reached
    if streak >= FORMALITY_CONSISTENCY_THRESHOLD:
        session["preferred_formality"] = detected_formality
    else:
        # On first turn always set it; it may be overridden later
        if session.get("preferred_formality") is None:
            session["preferred_formality"] = detected_formality

    # ── CSI category accumulation ─────────────────────────────────────────────
    seen_set = set(session.get("csi_categories_seen", []))
    for span in csi_spans:
        seen_set.add(span["category"])
    session["csi_categories_seen"] = sorted(seen_set)

    # ── Active warnings — keep high-severity from previous turns ──────────────
    prev_high = [
        w for w in session.get("active_warnings", [])
        if w["severity"] == "high"
    ]
    new_warnings = list(sensitivity_flags)

    seen_spans: set = set()
    merged: List[SensitivityFlag] = []
    for w in new_warnings + prev_high:
        key = w["span"].lower()
        if key not in seen_spans:
            seen_spans.add(key)
            merged.append(w)

    session["active_warnings"] = merged

    logger.debug(
        f"Session turn={session['turn_count']} "
        f"formality={session['preferred_formality']} "
        f"streak={streak} "
        f"csi_categories={session['csi_categories_seen']}"
    )

    return session


def get_session_summary(session: SessionHistory) -> dict:
    """
    Return a plain dict suitable for displaying in the Streamlit sidebar.
    """
    return {
        "turn_count":              session.get("turn_count", 0),
        "preferred_formality":     session.get("preferred_formality") or "—",
        "formality_established":   session.get("_formality_streak", 0) >= FORMALITY_CONSISTENCY_THRESHOLD,
        "csi_categories_seen":     session.get("csi_categories_seen", []),
        "active_warnings_count":   len(session.get("active_warnings", [])),
        "high_severity_warnings":  [
            w for w in session.get("active_warnings", [])
            if w["severity"] == "high"
        ],
    }