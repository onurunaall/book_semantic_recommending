#!/usr/bin/env python3
"""
Data Exploration and Cleaning

This module downloads the 7k books dataset using kagglehub,
adds some helper columns (like book age and missing-description flags),
filters out books with very short descriptions, combines the title and subtitle,
and creates a 'tagged_description' (ISBN + description) that will later be used
for semantic vector search. The cleaned data is meant to be saved as "books_cleaned.csv".

You can run this file on its own if you wish, but in our pipeline it is called from main.py.
"""

import os
import warnings
import numpy as np
import pandas as pd
import kagglehub

def download_dataset() -> str:
    """Download the 7k books dataset and return the local path."""
    warnings.filterwarnings("ignore")
    dataset_path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
    return dataset_path

def load_books(dataset_path: str) -> pd.DataFrame:
    """Load the raw books CSV file from the downloaded dataset."""
    books_csv = os.path.join(dataset_path, "books.csv")
    return pd.read_csv(books_csv)

def clean_books_data(books: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process the raw books data.

    Steps include:
      - Flagging missing descriptions and computing the book's age.
      - Filtering out books missing key fields.
      - Counting the words in each description and keeping only those with 25 or more words.
      - Combining the title and subtitle (if available).
      - Creating a tagged description (ISBN followed by the description).
      - Dropping intermediate columns.
    """
    # Mark books with missing descriptions and compute age.
    books["missing_description"] = np.where(books["description"].isna(), 1, 0)
    books["age_of_book"] = 2024 - books["published_year"]

    # Filter out rows missing any essential field.
    book_missing = books[
        ~(books["description"].isna()) &
        ~(books["num_pages"].isna()) &
        ~(books["average_rating"].isna()) &
        ~(books["published_year"].isna())
    ].copy()

    # Count the number of words in each description.
    book_missing["words_in_description"] = book_missing["description"].str.split().str.len()

    # Keep only books whose descriptions have at least 25 words.
    book_missing_25_words = book_missing[book_missing["words_in_description"] >= 25].copy()

    # Combine title and subtitle; if subtitle exists, join with a colon.
    book_missing_25_words["title_and_subtitle"] = np.where(
        book_missing_25_words["subtitle"].isna(),
        book_missing_25_words["title"],
        book_missing_25_words[["title", "subtitle"]].astype(str).agg(": ".join, axis=1)
    )

    # Create a tagged description (ISBN + description).
    book_missing_25_words["tagged_description"] = book_missing_25_words[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)

    # Drop the columns that were only needed temporarily.
    cleaned_books = book_missing_25_words.drop(
        ["subtitle", "missing_description", "age_of_book", "words_in_description"],
        axis=1
    )
    return cleaned_books

def main() -> None:
    """Run the data download and cleaning, then save the cleaned data to a CSV file."""
    print("Downloading and cleaning the 7k books dataset...")
    dataset_path = download_dataset()
    books = load_books(dataset_path)
    cleaned_books = clean_books_data(books)
    cleaned_books.to_csv("books_cleaned.csv", index=False)
    print("Cleaned dataset saved as 'books_cleaned.csv'.")

if __name__ == "__main__":
    main()
