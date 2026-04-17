"""
build_formality_data.py
-----------------------
Downloads the Pavlick formality scores dataset and prepares training data
for the XLM-RoBERTa formality classifier.

Dataset: Pavlick & Tetreault (2016) — Empirical Patterns of Formality in Online Communication
  - 11,263 sentences rated on a continuous formality scale
  - Available free on HuggingFace: osyvokon/pavlick-formality-scores
  - No account, no request needed

Also downloads X-FORMAL (Briakou et al., 2021) for French + Portuguese.

Output:
  data/formality_train.csv   — training split (80%)
  data/formality_test.csv    — test split (20%)
  data/formality_stats.json  — paper statistics

Usage:
  python3 data/build_formality_data.py
"""

import os, sys, json, logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_pavlick() -> list:
    """Download Pavlick formality scores from HuggingFace."""
    from datasets import load_dataset
    logger.info("Downloading Pavlick formality scores...")
    data = load_dataset("osyvokon/pavlick-formality-scores", split="train")
    logger.info(f"Pavlick: {len(data)} sentences, columns: {data.column_names}")

    records = []
    for row in data:
        score = row.get("avg_score", 0)
        # Continuous score → 3-class label
        # Pavlick scores: positive = formal, negative = informal
        # Use ±0.5 threshold to create a neutral class
        if score > 0.5:
            label = "formal"
        elif score < -0.5:
            label = "casual"
        else:
            label = "neutral"
        records.append({
            "text":  row.get("sentence", row.get("text", "")),
            "label": label,
            "lang":  "en",
            "score": round(score, 4),
        })
    return records


def load_xformal() -> list:
    """Download X-FORMAL (French + Portuguese formal/informal pairs)."""
    try:
        from datasets import load_dataset
        records = []
        configs = [("fr", "fr"), ("pt-br", "pt")]
        for config, lang in configs:
            try:
                data = load_dataset("Elbria/xformal-FoST", config, split="train")
                logger.info(f"X-FORMAL {config}: {len(data)} rows, columns: {data.column_names}")
                for row in data:
                    # X-FORMAL has parallel formal/informal columns
                    formal_text   = row.get("formal",   row.get("formal_ref", ""))
                    informal_text = row.get("informal", row.get("informal_ref", ""))
                    if formal_text:
                        records.append({"text": formal_text,   "label": "formal", "lang": lang, "score": 1.0})
                    if informal_text:
                        records.append({"text": informal_text, "label": "casual", "lang": lang, "score": -1.0})
            except Exception as e:
                logger.warning(f"X-FORMAL {config} failed: {e}")
        return records
    except Exception as e:
        logger.warning(f"X-FORMAL download failed: {e}")
        return []


def main():
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
    except ImportError:
        logger.error("Run: pip install datasets pandas scikit-learn")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Building formality training data")
    logger.info("=" * 60)

    all_records = []

    # Source 1: Pavlick English scores
    try:
        pavlick = load_pavlick()
        all_records.extend(pavlick)
        logger.info(f"Pavlick: {len(pavlick)} records added")
    except Exception as e:
        logger.error(f"Pavlick failed: {e}")

    # Source 2: X-FORMAL French + Portuguese
    try:
        xformal = load_xformal()
        all_records.extend(xformal)
        logger.info(f"X-FORMAL: {len(xformal)} records added")
    except Exception as e:
        logger.warning(f"X-FORMAL failed (non-fatal): {e}")

    if not all_records:
        logger.error("No data downloaded. Check internet connection.")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    df = df[df["text"].str.strip().str.len() > 5]  # remove empties

    # Split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_path = os.path.join(OUTPUT_DIR, "formality_train.csv")
    test_path  = os.path.join(OUTPUT_DIR, "formality_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    stats = {
        "datasets":        ["Pavlick & Tetreault (2016)", "X-FORMAL (Briakou et al., 2021)"],
        "total":           len(df),
        "train":           len(train_df),
        "test":            len(test_df),
        "label_dist":      df["label"].value_counts().to_dict(),
        "lang_dist":       df["lang"].value_counts().to_dict(),
    }
    with open(os.path.join(OUTPUT_DIR, "formality_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total sentences: {stats['total']}")
    logger.info(f"Train: {stats['train']}, Test: {stats['test']}")
    logger.info(f"Label distribution: {stats['label_dist']}")
    logger.info(f"Language distribution: {stats['lang_dist']}")
    logger.info(f"\nSaved to {train_path}")
    logger.info("Now run: python3 train_formality.py")


if __name__ == "__main__":
    main()