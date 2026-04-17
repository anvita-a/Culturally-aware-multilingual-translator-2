"""
train_formality.py
------------------
Anvita's Training Script — XLM-RoBERTa Formality Classifier

Run this script once in Week 2 to train the formality classifier.
Saves the model to models/formality_classifier/

Usage:
    python train_formality.py

What this does:
  1. Downloads GYAFC dataset (English formal/informal pairs)
  2. Optionally loads X-FORMAL (French, Portuguese)
  3. Loads LLM-generated labels for other languages (if file exists)
  4. Fine-tunes XLM-RoBERTa-base as a 3-class classifier: formal/neutral/casual
  5. Evaluates on held-out test set
  6. Saves model + tokenizer to models/formality_classifier/
  7. Prints classification report for the paper

Why XLM-RoBERTa-base (not large):
  - Base is sufficient for a 3-class classification task
  - Large would take 3-4x longer to train with marginal accuracy gain
  - Base fits in 8GB RAM during training

Expected training time:
  - ~20 minutes on a MacBook M1/M2 with MPS acceleration
  - ~45 minutes on CPU-only
  - ~5 minutes on Google Colab T4 GPU
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "xlm-roberta-base"
OUTPUT_DIR = Path("models/formality_classifier")
DATA_DIR = Path("data")
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
LABEL2ID = {"formal": 0, "neutral": 1, "casual": 2}
ID2LABEL = {0: "formal", 1: "neutral", 2: "casual"}


def load_gyafc_data() -> pd.DataFrame:
    """
    Load GYAFC dataset.
    GYAFC must be downloaded manually (requires a request to the authors):
    https://github.com/raosudha89/GYAFC-corpus

    Expected structure:
      data/gyafc/
        Entertainment_Music/
          train/formal, train/informal
          test/formal, test/informal
        Family_Relationships/
          train/formal, train/informal
          test/formal, test/informal

    Returns DataFrame with columns: text, label, lang, split
    """
    gyafc_path = DATA_DIR / "gyafc"
    records = []

    if not gyafc_path.exists():
        logger.warning(
            "GYAFC data not found at data/gyafc/. "
            "Download from: https://github.com/raosudha89/GYAFC-corpus\n"
            "Skipping GYAFC — using synthetic data for testing."
        )
        # Return small synthetic dataset for testing the training pipeline
        synthetic = [
            ("Dear Professor, I am writing to request an extension.", "formal", "en", "train"),
            ("I would like to formally inquire about the position.", "formal", "en", "train"),
            ("Please find attached the report as requested.", "formal", "en", "train"),
            ("Pursuant to our earlier discussion, I hereby submit.", "formal", "en", "train"),
            ("hey can u send me the file?", "casual", "en", "train"),
            ("omg that was so funny lol", "casual", "en", "train"),
            ("wanna grab coffee later?", "casual", "en", "train"),
            ("ngl i think we should just do it", "casual", "en", "train"),
            ("The meeting is scheduled for 3pm.", "neutral", "en", "train"),
            ("I need to submit the report by Friday.", "neutral", "en", "train"),
            ("Can you review this document?", "neutral", "en", "train"),
            ("The project deadline has been extended.", "neutral", "en", "train"),
            # Test split
            ("I humbly request your consideration of this matter.", "formal", "en", "test"),
            ("hey wats up?", "casual", "en", "test"),
            ("The document is ready for review.", "neutral", "en", "test"),
        ]
        return pd.DataFrame(synthetic, columns=["text", "label", "lang", "split"])

    for domain in ["Entertainment_Music", "Family_Relationships"]:
        for split in ["train", "test"]:
            for formality in ["formal", "informal"]:
                filepath = gyafc_path / domain / split / formality
                if not filepath.exists():
                    continue
                label = "formal" if formality == "formal" else "casual"
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append({
                                "text": line,
                                "label": label,
                                "lang": "en",
                                "split": split,
                            })

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} GYAFC examples")
    return df


def load_xformal_data() -> pd.DataFrame:
    """
    Load X-FORMAL dataset (French, Portuguese).
    Download from: https://github.com/Elbria/xformal-FoST

    Expected structure:
      data/xformal/
        fr/train.formal, fr/train.informal
        pt-br/train.formal, pt-br/train.informal

    Returns DataFrame with columns: text, label, lang, split
    """
    xformal_path = DATA_DIR / "xformal"
    records = []

    if not xformal_path.exists():
        logger.warning("X-FORMAL data not found. Skipping.")
        return pd.DataFrame(columns=["text", "label", "lang", "split"])

    lang_map = {"fr": "fr", "pt-br": "pt"}

    for folder, lang_code in lang_map.items():
        folder_path = xformal_path / folder
        if not folder_path.exists():
            continue
        for split in ["train", "test"]:
            for formality in ["formal", "informal"]:
                filepath = folder_path / f"{split}.{formality}"
                if not filepath.exists():
                    continue
                label = "formal" if formality == "formal" else "casual"
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append({
                                "text": line,
                                "label": label,
                                "lang": lang_code,
                                "split": split,
                            })

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} X-FORMAL examples")
    return df


def load_llm_generated_labels() -> pd.DataFrame:
    """
    Load LLM-generated formality labels for Japanese, Korean, Hindi, Arabic, etc.
    These are generated by running generate_formality_labels.py (see below).

    Expected file: data/llm_formality_labels.jsonl
    Each line: {"text": str, "label": str, "lang": str}
    """
    filepath = DATA_DIR / "llm_formality_labels.jsonl"
    if not filepath.exists():
        logger.warning("LLM-generated labels not found. Skipping.")
        return pd.DataFrame(columns=["text", "label", "lang", "split"])

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            rec["split"] = "train"  # all LLM-generated go to train
            records.append(rec)

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} LLM-generated formality labels")
    return df


def prepare_dataset(df: pd.DataFrame, tokenizer):
    """Convert DataFrame to HuggingFace Dataset."""
    from datasets import Dataset

    df = df[["text", "label"]].copy()
    df["labels"] = df["label"].map(LABEL2ID)
    df = df.dropna(subset=["labels"])

    dataset = Dataset.from_pandas(df[["text", "labels"]])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def train():
    """Main training function."""
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    import torch

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading training data...")

    # Priority 1: Pavlick formality CSV (downloaded by build_formality_data.py)
    pavlick_train = DATA_DIR / "formality_train.csv"
    pavlick_test  = DATA_DIR / "formality_test.csv"

    if pavlick_train.exists():
        logger.info(f"Loading Pavlick formality scores from {pavlick_train}")
        train_df = pd.read_csv(pavlick_train)[["text","label"]].copy()
        test_df  = pd.read_csv(pavlick_test)[["text","label"]].copy()
        train_df["split"] = "train"
        test_df["split"]  = "test"
        logger.info(f"Pavlick: {len(train_df)} train, {len(test_df)} test")
    else:
        logger.warning(
            "Pavlick CSV not found. Run: python3 data/build_formality_data.py\n"
            "Falling back to GYAFC/X-FORMAL/synthetic data."
        )
        dfs = [load_gyafc_data(), load_xformal_data(), load_llm_generated_labels()]
        df  = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
        train_df = df[df["split"] == "train"].copy()
        test_df  = df[df["split"] == "test"].copy()

    # Supplement with GYAFC if available (adds English formal/informal pairs)
    gyafc_df = load_gyafc_data()
    if len(gyafc_df) > 10:
        logger.info(f"Adding {len(gyafc_df)} GYAFC examples")
        gyafc_train = gyafc_df[gyafc_df["split"] == "train"][["text","label","split"]]
        gyafc_test  = gyafc_df[gyafc_df["split"] == "test"][["text","label","split"]]
        train_df = pd.concat([train_df, gyafc_train], ignore_index=True)
        test_df  = pd.concat([test_df,  gyafc_test],  ignore_index=True)

    # Add LLM-generated multilingual labels if available
    llm_df = load_llm_generated_labels()
    if len(llm_df) > 0:
        logger.info(f"Adding {len(llm_df)} LLM-generated labels")
        train_df = pd.concat([train_df, llm_df[["text","label"]].assign(split="train")], ignore_index=True)

    # If no test split, carve 15% from training
    if len(test_df) == 0:
        train_df, test_df = train_test_split(train_df, test_size=0.15, stratify=train_df["label"])

    logger.info(f"Training: {len(train_df)} examples | Test: {len(test_df)} examples")
    logger.info(f"Label distribution:\n{train_df['label'].value_counts()}")

    # ── Load tokenizer and model ──────────────────────────────────────────────
    logger.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── Tokenize ──────────────────────────────────────────────────────────────
    train_dataset = prepare_dataset(train_df, tokenizer)
    test_dataset = prepare_dataset(test_df, tokenizer)

    # ── Training arguments ────────────────────────────────────────────────────
    # Use MPS on Apple Silicon, CUDA if available, else CPU
    if torch.backends.mps.is_available():
        use_mps = True
        logger.info("Using Apple MPS acceleration")
    else:
        use_mps = False

    # eval_strategy replaces evaluation_strategy in transformers >= 4.45
    # use_mps_device was removed — MPS is auto-detected now
    import transformers as _tf
    _tf_version = tuple(int(x) for x in _tf.__version__.split(".")[:2])
    _eval_strategy_key = "eval_strategy" if _tf_version >= (4, 45) else "evaluation_strategy"

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=min(BATCH_SIZE, max(1, len(train_df) // 2)),
        per_device_eval_batch_size=min(BATCH_SIZE, max(1, len(test_df))),
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        **{_eval_strategy_key: "epoch"},
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
        report_to="none",  # disable wandb/tensorboard
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = (predictions == labels).mean()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    trainer.train()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = test_df["label"].map(LABEL2ID).values[:len(pred_labels)]

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=["formal", "neutral", "casual"],
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Save report for the paper
    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write(report)
    logger.info(f"Saved classification report to {OUTPUT_DIR / 'classification_report.txt'}")

    # ── Save model ────────────────────────────────────────────────────────────
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info(f"Model saved to {OUTPUT_DIR}")
    logger.info("Training complete. You can now run the full pipeline.")


if __name__ == "__main__":
    train()
