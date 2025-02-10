#!/usr/bin/env python3
"""
Main Pipeline for the Semantic Book Recommender

This script runs the entire data processing pipeline:
  1. Downloads and cleans the raw 7k books dataset.
  2. Maps and predicts missing category labels.
  3. Computes emotion scores for each book.
  4. Prepares the vector store.
  5. Launches the Gradio dashboard for interactive recommendations.

Simply run this file to go straight from raw data to the final interactive product.
"""

import pandas as pd

# Import functions from our modules
from data_exploration import download_dataset, load_books, clean_books_data
from text_classification import load_books as load_clean_books, map_categories, initialize_zero_shot_classifier, predict_missing_categories
from sentiment_analysis import load_books as load_books_with_categories, initialize_classifier, process_books
from vector_search import save_tagged_descriptions
from gradio_dashboard import launch_dashboard


def run_pipeline() -> None:
    # -------------------------------
    # Step 1: Data Exploration and Cleaning
    # -------------------------------
    print("Step 1: Downloading and cleaning the dataset...")
    dataset_path = download_dataset()
    raw_books = load_books(dataset_path)
    cleaned_books = clean_books_data(raw_books)
    cleaned_books.to_csv("books_cleaned.csv", index=False)
    print("  → Cleaned data saved as 'books_cleaned.csv'.")

    # -------------------------------
    # Step 2: Text Classification (Category Mapping)
    # -------------------------------
    print("Step 2: Mapping and classifying book categories...")
    books_clean = pd.read_csv("books_cleaned.csv")
    books_classified = map_categories(books_clean)
    classifier_zero_shot = initialize_zero_shot_classifier()
    missing_preds = predict_missing_categories(books_classified, classifier_zero_shot)
    # Fill missing simplified categories with predictions
    books_classified = pd.merge(books_classified, missing_preds, on="isbn13", how="left")
    books_classified["simple_categories"] = books_classified["simple_categories"].fillna(books_classified["predicted_categories"])
    books_classified.drop(columns=["predicted_categories"], inplace=True)
    books_classified.to_csv("books_with_categories.csv", index=False)
    print("  → Books with categories saved as 'books_with_categories.csv'.")

    # -------------------------------
    # Step 3: Sentiment Analysis (Emotion Scores)
    # -------------------------------
    print("Step 3: Computing emotion scores for each book...")
    books_with_categories = pd.read_csv("books_with_categories.csv")
    classifier_emotion = initialize_classifier()
    emotions_df = process_books(classifier_emotion, books_with_categories)
    books_with_emotions = pd.merge(books_with_categories, emotions_df, on="isbn13")
    books_with_emotions.to_csv("books_with_emotions.csv", index=False)
    print("  → Books with emotion scores saved as 'books_with_emotions.csv'.")

    # -------------------------------
    # Step 4: Prepare Tagged Descriptions for Vector Search
    # -------------------------------
    print("Step 4: Preparing tagged descriptions for vector search...")
    # The 'tagged_description' column is created during cleaning (in books_cleaned.csv)
    books_for_vector = pd.read_csv("books_cleaned.csv")
    save_tagged_descriptions(books_for_vector)
    print("  → Tagged descriptions saved as 'tagged_description.txt'.")


def main() -> None:
    """Run the complete pipeline and launch the Gradio dashboard."""
    run_pipeline()
    print("All processing done. Launching the Gradio Dashboard...")
    launch_dashboard()


if __name__ == "__main__":
    main()