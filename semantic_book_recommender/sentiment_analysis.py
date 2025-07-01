#!/usr/bin/env python3

import logging
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
import nltk
from transformers import pipeline, Pipeline

# Ensure the NLTK sentence tokenizer is available
nltk.download("punkt", quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_classifier() -> Pipeline:
    """
    Initialize the emotion classification pipeline on the best available device.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=device
    )
    return classifier

def calculate_max_emotion_scores(sentence_predictions: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    Given a list of predictions (one list per sentence), 
    return a dictionary of the max score for each label across all sentences.
    """
    max_scores: Dict[str, float] = {}
    for preds in sentence_predictions:
        for entry in preds:
            label = entry["label"]
            score = entry["score"]
            if label not in max_scores or score > max_scores[label]:
                max_scores[label] = score
    return max_scores

def process_books(classifier: Pipeline, books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run emotion analysis on all book descriptions in batch.
    Return a DataFrame with isbn13 and max emotion scores per book.
    """
    # Tokenize descriptions into lists of sentences
    sentence_lists: List[List[str]] = books_df["description"].apply(nltk.sent_tokenize).tolist()
    
    # Flatten the sentences for batch classification
    all_sentences: List[str] = []
    for sentence_group in sentence_lists:
        all_sentences.extend(sentence_group)
    
    logger.info(f"Running batch emotion classification on {len(all_sentences)} sentences.")
    
    # Run batch classification
    try:
        all_predictions = classifier(all_sentences)
    except Exception as e:
        logger.error(f"Batch emotion classification failed: {e}")
        all_predictions = [[] for _ in all_sentences]
    
    # Group predictions back to their respective books
    grouped_predictions: List[List[List[Dict[str, Any]]]] = []
    idx = 0
    for sentence_group in sentence_lists:
        n_sentences = len(sentence_group)
        grouped_predictions.append(all_predictions[idx : idx + n_sentences])
        idx += n_sentences
    
    # For each book, compute max emotion scores
    records = []
    for isbn, sentence_preds in zip(books_df["isbn13"], grouped_predictions):
        max_scores = calculate_max_emotion_scores(sentence_preds)
        max_scores["isbn13"] = isbn
        records.append(max_scores)
    
    # Return DataFrame with max scores and isbn13
    return pd.DataFrame.from_records(records)

def main(books_df: pd.DataFrame) -> None:
    logger.info("Initializing emotion classifier...")
    classifier = initialize_classifier()

    logger.info(f"Processing {len(books_df)} books for emotion scores...")
    emotions_df = process_books(classifier, books_df)

    # Merge the new emotion scores with the original DataFrame and save
    merged_df = books_df.merge(emotions_df, on="isbn13")
    merged_df.to_csv("books_with_emotions.csv", index=False)
    logger.info("Saved 'books_with_emotions.csv' with emotion scores.")

if __name__ == "__main__":
    # For standalone execution, load the already-categorized data
    books_df = pd.read_csv("books_with_categories.csv")
    main(books_df)
