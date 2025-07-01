#!/usr/bin/env python3

import logging
import torch
import numpy as np
import pandas as pd
from transformers import pipeline, Pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_books() -> pd.DataFrame:
    """Load the cleaned books CSV into a DataFrame."""
    return pd.read_csv("books_cleaned.csv")

def map_categories(books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the detailed book categories to simplified categories.
    Adds a 'simple_categories' column.
    """
    category_mapping = {
        'Fiction': "Fiction",
        'Juvenile Fiction': "Children's Fiction",
        'Biography & Autobiography': "Nonfiction",
        'History': "Nonfiction",
        'Literary Criticism': "Nonfiction",
        'Philosophy': "Nonfiction",
        'Religion': "Nonfiction",
        'Comics & Graphic Novels': "Fiction",
        'Drama': "Fiction",
        'Juvenile Nonfiction': "Children's Nonfiction",
        'Science': "Nonfiction",
        'Poetry': "Fiction"
    }
    books_df["simple_categories"] = books_df["categories"].map(category_mapping)
    return books_df

def initialize_zero_shot_classifier() -> Pipeline:
    """
    Initialize a zero-shot classification pipeline with the best available device.
    Returns the classifier pipeline.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )

def predict_missing_categories(books_df: pd.DataFrame, classifier: Pipeline) -> pd.DataFrame:
    """
    Predicts 'Fiction' or 'Nonfiction' for books with missing simple categories.
    Returns a DataFrame with isbn13 and predicted_categories columns.
    """
    # Select books with missing categories
    missing_mask = books_df["simple_categories"].isna()
    missing_books = books_df.loc[missing_mask, ["isbn13", "description"]].reset_index(drop=True)
    descriptions = missing_books["description"].tolist()

    # Try zero-shot classification
    try:
        predictions = classifier(
            descriptions,
            candidate_labels=["Fiction", "Nonfiction"]
        )
    except Exception as e:
        logger.error(f"Zero-shot classification failed: {e}")
        predictions = [{} for _ in descriptions]

    # Parse predictions and select highest scoring label
    predicted_labels = []
    for i, pred in enumerate(predictions):
        try:
            labels = pred.get("labels", [])
            scores = pred.get("scores", [])
            if labels and scores:
                best_idx = int(np.nanargmax(scores))
                predicted_labels.append(labels[best_idx])
            else:
                predicted_labels.append(None)
        except Exception as e:
            logger.error(f"Prediction parsing failed for row {i}: {e}")
            predicted_labels.append(None)

    return pd.DataFrame({
        "isbn13": missing_books["isbn13"],
        "predicted_categories": predicted_labels
    })

def main() -> None:
    logger.info("Loading books data...")
    books_df = load_books()

    logger.info("Mapping detailed categories to simplified categories...")
    books_df = map_categories(books_df)

    logger.info("Initializing zero-shot classifier...")
    classifier = initialize_zero_shot_classifier()

    logger.info("Predicting missing categories...")
    predictions_df = predict_missing_categories(books_df, classifier)

    logger.info("Updating missing simple categories with predictions...")
    books_with_preds = books_df.merge(predictions_df, on="isbn13", how="left")
    books_with_preds["simple_categories"] = books_with_preds["simple_categories"].fillna(books_with_preds["predicted_categories"])
    books_with_preds = books_with_preds.drop(columns=["predicted_categories"])

    books_with_preds.to_csv("books_with_categories.csv", index=False)
    logger.info("Saved 'books_with_categories.csv'.")

if __name__ == "__main__":
    main()
