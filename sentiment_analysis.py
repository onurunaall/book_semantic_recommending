#!/usr/bin/env python3
"""
Sentiment Analysis for Books

This module loads the books data with categories from "books_with_categories.csv",
computes the maximum emotion scores (for emotions such as anger, joy, etc.) for each book's description
using a transformer-based classifier, and then saves the enriched data as "books_with_emotions.csv".
"""

import numpy as np
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

def load_books() -> pd.DataFrame:
    """Load the books dataset with categories from 'books_with_categories.csv'."""
    return pd.read_csv("books_with_categories.csv")

def initialize_classifier():
    """Initialize the emotion classification pipeline using j-hartmann/emotion-english-distilroberta-base."""
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device="mps"
    )
    return classifier

def calculate_max_emotion_scores(predictions: list) -> dict:
    """
    Given a list of predictions (one per sentence), return the maximum score for each emotion.

    Args:
        predictions (list): A list of prediction results.

    Returns:
        dict: A dictionary mapping each emotion (e.g., "joy", "fear") to its maximum score.
    """
    emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    per_emotion_scores = {label: [] for label in emotion_labels}
    for prediction in predictions:
        # Sort predictions by label so that we can reliably index them.
        sorted_predictions = sorted(prediction, key=lambda x: x["label"])
        for idx, label in enumerate(emotion_labels):
            per_emotion_scores[label].append(sorted_predictions[idx]["score"])
    return {label: np.max(scores) for label, scores in per_emotion_scores.items()}

def process_books(classifier, books: pd.DataFrame) -> pd.DataFrame:
    """
    Process each book to compute the maximum emotion scores from its description.

    Args:
        classifier: The emotion classifier pipeline.
        books (pd.DataFrame): The books DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the maximum emotion scores and corresponding ISBN numbers.
    """
    emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    isbns = []
    emotion_scores = {label: [] for label in emotion_labels}
    for i in tqdm(range(len(books)), desc="Computing emotion scores"):
        isbns.append(books["isbn13"].iloc[i])
        # Split the description into sentences.
        sentences = books["description"].iloc[i].split(".")
        predictions = classifier(sentences)
        max_scores = calculate_max_emotion_scores(predictions)
        for label in emotion_labels:
            emotion_scores[label].append(max_scores[label])
    emotions_df = pd.DataFrame(emotion_scores)
    emotions_df["isbn13"] = isbns
    return emotions_df

def main() -> None:
    """Run the sentiment analysis process and save the output as 'books_with_emotions.csv'."""
    print("Loading books data with categories...")
    books = load_books()
    print("Initializing emotion classifier...")
    classifier = initialize_classifier()
    emotions_df = process_books(classifier, books)
    books_with_emotions = pd.merge(books, emotions_df, on="isbn13")
    books_with_emotions.to_csv("books_with_emotions.csv", index=False)
    print("Books with emotion scores saved as 'books_with_emotions.csv'.")

if __name__ == "__main__":
    main()