#!/usr/bin/env python3
"""
Text Classification for Books

This module takes the cleaned books data (books_cleaned.csv),
maps the original categories to simplified ones, and for any book where this mapping is missing,
uses a zero-shot classification model (BART-MNLI) to predict the simplified category.
The output is saved as "books_with_categories.csv".
"""

from typing import List
import numpy as np
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

def load_books() -> pd.DataFrame:
    """Load the cleaned books data from 'books_cleaned.csv'."""
    return pd.read_csv("books_cleaned.csv")

def map_categories(books: pd.DataFrame) -> pd.DataFrame:
    """
    Map the original book categories to a simpler set.

    Mapping details:
      - 'Fiction' becomes "Fiction"
      - 'Juvenile Fiction' becomes "Children's Fiction"
      - 'Biography & Autobiography', 'History', 'Literary Criticism', 'Philosophy', 'Religion', 'Science' become "Nonfiction"
      - 'Comics & Graphic Novels', 'Drama', 'Poetry' become "Fiction"
      - 'Juvenile Nonfiction' becomes "Children's Nonfiction"
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
    books["simple_categories"] = books["categories"].map(category_mapping)
    return books

def initialize_zero_shot_classifier():
    """Initialize the zero-shot classification pipeline using facebook/bart-large-mnli."""
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device="mps"
    )
    return classifier

def generate_prediction(sequence: str, candidate_labels: List[str], classifier) -> str:
    """
    Use the zero-shot classifier to predict a category for the given text.

    Args:
        sequence (str): The book description.
        candidate_labels (List[str]): The candidate labels (e.g., ["Fiction", "Nonfiction"]).
        classifier: The zero-shot classification pipeline.

    Returns:
        str: The predicted category.
    """
    predictions = classifier(sequence, candidate_labels)
    max_index = int(np.argmax(predictions["scores"]))
    return predictions["labels"][max_index]

def predict_missing_categories(books: pd.DataFrame, classifier) -> pd.DataFrame:
    """
    For books that are missing a simplified category, predict one using the zero-shot classifier.

    Args:
        books (pd.DataFrame): The DataFrame of books.
        classifier: The zero-shot classification pipeline.

    Returns:
        pd.DataFrame: A DataFrame with ISBNs and the predicted categories.
    """
    candidate_labels = ["Fiction", "Nonfiction"]
    missing = books.loc[books["simple_categories"].isna(), ["isbn13", "description"]].reset_index(drop=True)
    predicted_categories = []
    isbns = []
    for i in tqdm(range(len(missing)), desc="Predicting missing categories"):
        text = missing["description"].iloc[i]
        pred = generate_prediction(text, candidate_labels, classifier)
        predicted_categories.append(pred)
        isbns.append(missing["isbn13"].iloc[i])
    return pd.DataFrame({"isbn13": isbns, "predicted_categories": predicted_categories})

def main() -> None:
    """Run the category mapping and prediction, then save the results as 'books_with_categories.csv'."""
    print("Loading cleaned books data...")
    books = load_books()
    books = map_categories(books)
    print("Initializing zero-shot classifier...")
    classifier = initialize_zero_shot_classifier()
    missing_pred_df = predict_missing_categories(books, classifier)
    books = pd.merge(books, missing_pred_df, on="isbn13", how="left")
    books["simple_categories"] = books["simple_categories"].fillna(books["predicted_categories"])
    books.drop(columns=["predicted_categories"], inplace=True)
    books.to_csv("books_with_categories.csv", index=False)
    print("Books with simplified categories saved as 'books_with_categories.csv'.")

if __name__ == "__main__":
    main()