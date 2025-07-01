#!/usr/bin/env python3

import os
import warnings
import numpy as np
import pandas as pd
import kagglehub

def download_dataset() -> str:
    """Download the 7k-books-with-metadata dataset and return its directory path."""
    warnings.filterwarnings("ignore")
    dataset_path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
    return dataset_path

def load_books(dataset_dir: str) -> pd.DataFrame:
    """Load the books CSV file into a DataFrame."""
    csv_path = os.path.join(dataset_dir, "books.csv")
    return pd.read_csv(csv_path)

def combine_title_and_subtitle(row: pd.Series) -> str | float:
    """Combine title and subtitle columns, handling NaN cases gracefully."""
    title = row["title"]
    subtitle = row["subtitle"]

    if pd.isna(title) and pd.isna(subtitle):
        return np.nan
    if pd.isna(subtitle):
        return title
    if pd.isna(title):
        return subtitle
    return f"{title}: {subtitle}"

def clean_books_data(raw_books: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw books DataFrame and return a filtered DataFrame with required columns."""
    # Mark rows with missing descriptions
    raw_books["missing_description"] = raw_books["description"].isna().astype(int)

    # Calculate the age of each book
    current_year = pd.to_datetime("today").year
    raw_books["age_of_book"] = current_year - raw_books["published_year"]

    # Keep only rows with all essential fields present
    has_all_fields = (
        raw_books["description"].notna() &
        raw_books["num_pages"].notna() &
        raw_books["average_rating"].notna() &
        raw_books["published_year"].notna()
    )
    complete_books = raw_books[has_all_fields].copy()

    # Add a word count for descriptions
    complete_books["words_in_description"] = complete_books["description"].str.split().str.len()

    # Only keep books with at least 25 words in their description
    enough_words = complete_books["words_in_description"] >= 25
    valid_books = complete_books[enough_words].copy()

    # Combine title and subtitle safely
    valid_books["title_and_subtitle"] = valid_books.apply(combine_title_and_subtitle, axis=1)

    # Add a tagged description column (ISBN + description)
    valid_books["tagged_description"] = valid_books[["isbn13", "description"]].agg(" ".join, axis=1)

    # Remove helper columns
    columns_to_drop = ["subtitle", "missing_description", "age_of_book", "words_in_description"]
    cleaned_books = valid_books.drop(columns=columns_to_drop)

    return cleaned_books

def main() -> None:
    print("Downloading and cleaning the 7k books dataset...")

    # Download data
    dataset_dir = download_dataset()

    # Load data
    raw_books = load_books(dataset_dir)

    # Clean data
    cleaned_books = clean_books_data(raw_books)

    # Save cleaned data to CSV
    cleaned_books.to_csv("books_cleaned.csv", index=False)
    print("Cleaned dataset saved as 'books_cleaned.csv'.")

if __name__ == "__main__":
    main()
