#!/usr/bin/env python3

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import Pipeline, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_books() -> pd.DataFrame:
    """Load cleaned books CSV."""
    return pd.read_csv("books_cleaned.csv")


def map_categories(books_df: pd.DataFrame) -> pd.DataFrame:
    """Map detailed categories to simplified ones."""
    mapping = {
        "Fiction": "Fiction",
        "Juvenile Fiction": "Children's Fiction",
        "Biography & Autobiography": "Nonfiction",
        "History": "Nonfiction",
        "Literary Criticism": "Nonfiction",
        "Philosophy": "Nonfiction",
        "Religion": "Nonfiction",
        "Comics & Graphic Novels": "Fiction",
        "Drama": "Fiction",
        "Juvenile Nonfiction": "Children's Nonfiction",
        "Science": "Nonfiction",
        "Poetry": "Fiction",
    }
    books_df["simple_categories"] = books_df["categories"].map(mapping)
    return books_df


def initialize_zero_shot_classifier() -> Pipeline:
    """Initialize zero-shot classifier on best device."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )


def predict_missing_categories(
    books_df: pd.DataFrame,
    classifier: Pipeline,
) -> pd.DataFrame:
    """
    Predict missing simple categories via zero-shot.
    Returns DataFrame with isbn13 + predicted_categories.
    """
    missing = books_df[books_df["simple_categories"].isna()]
    if missing.empty:
        return pd.DataFrame(columns=["isbn13", "predicted_categories"])

    missing = missing[["isbn13", "description"]].reset_index(drop=True)
    descs = missing["description"].tolist()

    try:
        preds = classifier(descs, candidate_labels=["Fiction", "Nonfiction"])
    except Exception as err:
        logger.error("Zero-shot failed: %s", err)
        raise

    labels: list[Any] = []
    for i, p in enumerate(preds):
        lbls = p.get("labels", [])
        scores = p.get("scores", [])
        if lbls and scores:
            try:
                idx = int(np.nanargmax(scores))
                labels.append(lbls[idx])
            except ValueError:
                logger.error("Could not find max score in prediction %d", i)
                labels.append(None)

        else:
            labels.append(None)

    return pd.DataFrame({
        "isbn13": missing["isbn13"],
        "predicted_categories": labels,
    })


def classify_categories(books_df: pd.DataFrame) -> pd.DataFrame:
    """Map and then predict+fill missing simple_categories."""
    df = map_categories(books_df)
    clf = initialize_zero_shot_classifier()
    preds = predict_missing_categories(df, clf)
    df = df.merge(preds, on="isbn13", how="left")
    df["simple_categories"] = df["simple_categories"].fillna(
        df["predicted_categories"]
    )
    return df.drop(columns=["predicted_categories"])


def main() -> None:
    """Run classification and save to disk."""
    df = load_books()
    df = classify_categories(df)
    df.to_csv("books_with_categories.csv", index=False)
    logger.info("Saved 'books_with_categories.csv'.")


if __name__ == "__main__":
    main()
