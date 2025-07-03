#!/usr/bin/env python3

import os
import warnings

import numpy as np
import pandas as pd
import kagglehub


def download_dataset() -> str:
    """Download the 7k-books-with-metadata dataset and return its directory."""
    warnings.filterwarnings("ignore")
    return kagglehub.dataset_download(
        "dylanjcastillo/7k-books-with-metadata"
    )


def load_books(dataset_dir: str) -> pd.DataFrame:
    """Load books CSV into a DataFrame."""
    csv_path = os.path.join(dataset_dir, "books.csv")
    return pd.read_csv(csv_path)


def combine_title_and_subtitle(row: pd.Series) -> str | float:
    """Combine title and subtitle, handling NaN cases."""
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
    """Clean raw books and return a filtered DataFrame."""
    raw_books["missing_description"] = (
        raw_books["description"].isna().astype(int)
    )

    current_year = pd.to_datetime("today").year
    raw_books["age_of_book"] = (
        current_year - raw_books["published_year"]
    )

    mask = (
        raw_books["description"].notna()
        & raw_books["num_pages"].notna()
        & raw_books["average_rating"].notna()
        & raw_books["published_year"].notna()
    )
    complete = raw_books[mask]

    complete["words_in_description"] = (
        complete["description"].str.split().str.len()
    )

    valid = complete[complete["words_in_description"] >= 25].copy()

    valid["title_and_subtitle"] = valid.apply(combine_title_and_subtitle, axis=1)

    valid["tagged_description"] = valid[
        ["isbn13", "description"]
    ].agg(" ".join, axis=1)

    drop_cols = [
        "subtitle",
        "missing_description",
        "age_of_book",
        "words_in_description",
    ]
    return valid.drop(columns=drop_cols)


def main() -> None:
    """Download, clean, and save the books dataset."""
    print("Downloading and cleaning the 7k books dataset...")
    dataset_dir = download_dataset()
    raw = load_books(dataset_dir)
    cleaned = clean_books_data(raw)
    cleaned.to_csv("books_cleaned.csv", index=False)
    print("Cleaned dataset saved as 'books_cleaned.csv'.")


if __name__ == "__main__":
    main()
