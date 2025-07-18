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
    batch_size: int = 128,
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

    logger.info("Classifying %d sentences in batches of %d", len(all_sents), batch_size)
    
    # Process sentences in batches
    all_preds = []
    for i in range(0, len(all_sents), batch_size):
        batch = all_sents[i:i + batch_size]
        try:
            batch_preds = classifier(batch)
            all_preds.extend(batch_preds)
        except Exception as err:
            logger.warning(
                "Batch classification failed for batch %d-%d: %s",
                i, i + len(batch), err
            )
            # Add empty predictions for failed batch
            all_preds.extend([[] for _ in batch])

    records = []
    idx = 0
    for isbn, group in zip(books_df["isbn13"], sentence_lists):
        preds = all_preds[idx:(idx + len(group))]
        idx += len(group)
        max_scores = calculate_max_emotion_scores(preds)
        max_scores["isbn13"] = isbn
        records.append(max_scores)

    return pd.DataFrame.from_records(records)
