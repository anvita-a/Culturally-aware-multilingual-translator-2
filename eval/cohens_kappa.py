"""
eval/cohens_kappa.py
--------------------
Asmi's Component — Inter-Annotator Agreement (Cohen's κ)

Measures agreement between:
  - Claude's automatic CSI annotations (first-pass annotator)
  - Human spot-check annotations (or GPT-4 as second annotator)

Cohen's κ > 0.6 = substantial agreement (sufficient for the paper).
κ > 0.8 = almost perfect agreement (excellent).

This number goes in the paper as Table 1. It validates that the automated
CSI annotation pipeline produces reliable labels.

Usage:
  # Step 1: Generate Claude annotations (if not done)
  python3 eval/cohens_kappa.py --annotate --n 200

  # Step 2: Human verifier reviews the file produced
  # Edit eval/results/human_verification.json to mark each as correct/wrong

  # Step 3: Compute kappa
  python3 eval/cohens_kappa.py --compute

  # Or: Use GPT-4 as second annotator (no human needed)
  python3 eval/cohens_kappa.py --gpt4-second-pass
"""

import os, sys, json, re, logging, argparse, random
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

RESULTS_DIR = "eval/results"
ANNOTATION_FILE = os.path.join(RESULTS_DIR, "csi_annotations_sample.json")
VERIFICATION_FILE = os.path.join(RESULTS_DIR, "human_verification.json")

CSI_CATEGORIES = ["proper_name", "culturally_embedded", "institutional", "pragmatic", "none"]


def _get_flores_sentences(n: int = 200) -> List[str]:
    """Get n English sentences from FLORES-200."""
    try:
        from datasets import load_dataset
        flores = load_dataset("Muennighoff/flores200", "eng_Latn", split="devtest")
        sentences = [row["sentence"] for row in flores]
        random.seed(42)
        return random.sample(sentences, min(n, len(sentences)))
    except Exception as e:
        logger.warning(f"FLORES not available: {e}. Using built-in samples.")
        return [
            "Let's break the ice before the meeting.",
            "She brought mochi and wagashi to the party.",
            "The karma of the situation was undeniable.",
            "We celebrated Thanksgiving dinner together.",
            "The CEO announced the new GDPR policy.",
            "Please send kind regards to the team.",
            "The concept of wabi-sabi is beautiful.",
            "He knocked on wood for good luck.",
            "The biryani at the restaurant was excellent.",
            "That's what she said.",
        ][:n]


def annotate_with_claude(sentences: List[str]) -> List[Dict]:
    """
    First-pass annotation using Claude/Gemini.
    Returns list of {sentence, spans: [{span, category}], annotator: "claude"}
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No GEMINI_API_KEY — using rule-based annotation only")
        return _annotate_rule_based(sentences)

    import google.generativeai as genai
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        )
    )

    results = []
    batch_size = 5

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        numbered = "\n".join(f"{j+1}. {s}" for j, s in enumerate(batch))

        prompt = (
            "You are a cultural linguistics expert following Hershcovich et al. (2022).\n"
            "For each sentence below, identify ALL culture-specific items (CSIs).\n\n"
            f"{numbered}\n\n"
            "CSI categories:\n"
            "  proper_name — holidays, cultural foods, places, events\n"
            "  culturally_embedded — idioms, proverbs, untranslatable concepts\n"
            "  institutional — country-specific abbreviations (CEO, NHS, GPA)\n"
            "  pragmatic — politeness conventions, closings\n\n"
            "Return a JSON array, one object per sentence:\n"
            '[{"sentence_num": 1, "spans": [{"span": "...", "category": "..."}]}]\n'
            "If no CSIs in a sentence, spans = [].\n"
            "Return ONLY the JSON array."
        )

        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()
            items = json.loads(raw if raw.startswith("[") else
                               re.search(r"\[[\s\S]*\]", raw).group(0))

            for j, item in enumerate(items):
                if j < len(batch):
                    results.append({
                        "sentence":  batch[j],
                        "spans":     item.get("spans", []),
                        "annotator": "gemini",
                    })

            logger.info(f"Annotated batch {i//batch_size + 1}/{(len(sentences)+batch_size-1)//batch_size}")

        except Exception as e:
            logger.warning(f"Batch {i//batch_size} annotation failed: {e}")
            for s in batch:
                results.append({"sentence": s, "spans": [], "annotator": "fallback"})

    return results


def _annotate_rule_based(sentences: List[str]) -> List[Dict]:
    """Fallback: use the rule-based CSI detector."""
    from pipeline.text_preprocessor import preprocess_text
    from pipeline.csi_detector import detect_csi_spans

    results = []
    for sentence in sentences:
        try:
            inp   = preprocess_text(sentence, "ja", source_lang="en")
            spans = detect_csi_spans(inp)
            results.append({
                "sentence":  sentence,
                "spans":     [{"span": s["span"], "category": s["category"]} for s in spans],
                "annotator": "rule_based",
            })
        except Exception as e:
            results.append({"sentence": sentence, "spans": [], "annotator": "fallback"})
    return results


def annotate_with_gpt4(annotations: List[Dict]) -> List[Dict]:
    """
    Second-pass annotation using GPT-4 as independent annotator.
    Used when no human annotators are available.
    """
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No OPENAI_API_KEY — GPT-4 second pass not available")
        return []

    client = openai.OpenAI(api_key=api_key)
    results = []

    for ann in annotations:
        sentence = ann["sentence"]
        prompt = (
            f"Identify all culture-specific items (CSIs) in this sentence:\n"
            f'"{sentence}"\n\n'
            "Use categories: proper_name, culturally_embedded, institutional, pragmatic\n"
            "Return JSON: {\"spans\": [{\"span\": \"...\", \"category\": \"...\"}]}\n"
            "If none found: {\"spans\": []}"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=256,
            )
            data = json.loads(response.choices[0].message.content)
            results.append({
                "sentence":  sentence,
                "spans":     data.get("spans", []),
                "annotator": "gpt4",
            })
        except Exception as e:
            logger.warning(f"GPT-4 annotation failed: {e}")
            results.append({"sentence": sentence, "spans": [], "annotator": "fallback"})

    return results


def compute_kappa(
    annotations_a: List[Dict],
    annotations_b: List[Dict],
) -> Dict:
    """
    Compute Cohen's κ between two sets of annotations.

    Converts span-level annotations to sentence-level binary:
    for each sentence, each category is 1 (present) or 0 (absent).
    κ is computed per category and as a macro average.
    """
    if len(annotations_a) != len(annotations_b):
        min_len = min(len(annotations_a), len(annotations_b))
        annotations_a = annotations_a[:min_len]
        annotations_b = annotations_b[:min_len]

    categories = ["proper_name", "culturally_embedded", "institutional", "pragmatic"]
    kappas = {}

    for cat in categories:
        labels_a = []
        labels_b = []
        for ann_a, ann_b in zip(annotations_a, annotations_b):
            has_a = int(any(s["category"] == cat for s in ann_a.get("spans", [])))
            has_b = int(any(s["category"] == cat for s in ann_b.get("spans", [])))
            labels_a.append(has_a)
            labels_b.append(has_b)

        # Cohen's kappa formula
        n = len(labels_a)
        if n == 0:
            kappas[cat] = 0.0
            continue

        # Observed agreement
        p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

        # Expected agreement
        p_a1 = sum(labels_a) / n
        p_b1 = sum(labels_b) / n
        p_e  = (p_a1 * p_b1) + ((1 - p_a1) * (1 - p_b1))

        kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0
        kappas[cat] = round(kappa, 4)

    macro_kappa = round(sum(kappas.values()) / len(kappas), 4)

    # Interpret
    def interpret(k):
        if k < 0:       return "poor (worse than chance)"
        elif k < 0.2:   return "slight"
        elif k < 0.4:   return "fair"
        elif k < 0.6:   return "moderate"
        elif k < 0.8:   return "substantial ✓ (publishable)"
        else:           return "almost perfect ✓✓"

    return {
        "n_sentences":    len(annotations_a),
        "per_category":   kappas,
        "macro_kappa":    macro_kappa,
        "interpretation": interpret(macro_kappa),
        "paper_ready":    macro_kappa >= 0.6,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate",        action="store_true",
                        help="Run first-pass annotation on 200 FLORES sentences")
    parser.add_argument("--gpt4-second-pass",action="store_true",
                        help="Run GPT-4 as second annotator (no human needed)")
    parser.add_argument("--compute",         action="store_true",
                        help="Compute kappa from existing annotation files")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of sentences to annotate (default: 200)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.annotate or not os.path.exists(ANNOTATION_FILE):
        logger.info(f"Annotating {args.n} FLORES sentences with Gemini/Claude...")
        sentences = _get_flores_sentences(args.n)
        annotations = annotate_with_claude(sentences)

        with open(ANNOTATION_FILE, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(annotations)} annotations to {ANNOTATION_FILE}")
        logger.info(f"\nNext step:")
        logger.info(f"  Option A: Open {ANNOTATION_FILE} and have a human")
        logger.info(f"            verify each annotation (correct/wrong)")
        logger.info(f"  Option B: Run --gpt4-second-pass for automated verification")

    if args.gpt4_second_pass:
        logger.info("Running GPT-4 as second annotator...")
        with open(ANNOTATION_FILE, encoding="utf-8") as f:
            first_pass = json.load(f)

        second_pass = annotate_with_gpt4(first_pass)
        if second_pass:
            with open(VERIFICATION_FILE, "w", encoding="utf-8") as f:
                json.dump(second_pass, f, ensure_ascii=False, indent=2)
            logger.info(f"GPT-4 annotations saved to {VERIFICATION_FILE}")
            args.compute = True

    if args.compute:
        if not os.path.exists(ANNOTATION_FILE):
            logger.error(f"First run: python3 eval/cohens_kappa.py --annotate")
            return
        if not os.path.exists(VERIFICATION_FILE):
            logger.error(f"Need second-pass annotations. Run --gpt4-second-pass")
            return

        with open(ANNOTATION_FILE, encoding="utf-8") as f:
            ann_a = json.load(f)
        with open(VERIFICATION_FILE, encoding="utf-8") as f:
            ann_b = json.load(f)

        result = compute_kappa(ann_a, ann_b)

        out_path = os.path.join(RESULTS_DIR, "kappa_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info("\n" + "=" * 55)
        logger.info("COHEN'S κ RESULTS  (Table 1 in paper)")
        logger.info("=" * 55)
        for cat, k in result["per_category"].items():
            logger.info(f"  {cat:<25} κ = {k:.4f}")
        logger.info(f"  {'Macro average':<25} κ = {result['macro_kappa']:.4f}")
        logger.info(f"\n  Interpretation: {result['interpretation']}")
        logger.info(f"  Paper-ready (κ ≥ 0.6): {result['paper_ready']}")
        logger.info(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()