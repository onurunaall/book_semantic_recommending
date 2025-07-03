#!/usr/bin/env python3

import logging
from typing import Any, Dict, List

import pandas as pd
import torch
import nltk
from transformers import Pipeline, pipeline

# ensure NLTK tokenizer is present
nltk.download("punkt", quiet=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_classifier() -> Pipeline:
    """Initialize emotion classifier on available device."""
    if torch.cuda.is_available():
        device = "cuda"
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = "mps"
    else:
        device = "cpu"

    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=device,
    )


def calculate_max_emotion_scores(
    sentence_predictions: List[List[Dict[str, Any]]]
) -> Dict[str, float]:
    """Return max score per label across all sentence predictions."""
    max_scores: Dict[str, float] = {}
    for preds in sentence_predictions:
        for entry in preds:
            label = entry["label"]
            score = entry["score"]
            if label not in max_scores or score > max_scores[label]:
                max_scores[label] = score
    return max_scores


def process_books(
    classifier: Pipeline,
    books_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Batch-process all descriptions, compute max emotion scores,
    and return a frame with isbn13 + scores.
    """
    sentence_lists = books_df["description"].apply(
        nltk.sent_tokenize
    ).tolist()
    all_sents: List[str] = [
        s for group in sentence_lists for s in group
    ]

    logger.info("Classifying %d sentences", len(all_sents))
    try:
        all_preds = classifier(all_sents)
    except Exception as err:
        logger.error("Batch classification failed: %s", err)
        all_preds = [[] for _ in all_sents]

    records = []
    idx = 0
    for isbn, group in zip(books_df["isbn13"], sentence_lists):
        preds = all_preds[idx:(idx + len(group))]
        idx += len(group)
        max_scores = calculate_max_emotion_scores(preds)
        max_scores["isbn13"] = isbn
        records.append(max_scores)

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    df = pd.read_csv("books_with_categories.csv")
    clf = initialize_classifier()
    result = process_books(clf, df)
    merged = df.merge(result, on="isbn13")
    merged.to_csv("books_with_emotions.csv", index=False)
