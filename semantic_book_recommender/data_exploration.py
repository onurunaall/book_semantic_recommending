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
    # mark missing descriptions
    raw_books["missing_description"] = np.where(
        raw_books["description"].isna(),
        1,
        0,
    )
    # compute book age using current year
    raw_books["age_of_book"] = (
        pd.to_datetime("today").year
        - raw_books["published_year"]
    )

    # filter out rows missing essential fields
    mask = (
        raw_books["description"].notna()
        & raw_books["num_pages"].notna()
        & raw_books["average_rating"].notna()
        & raw_books["published_year"].notna()
    )
    complete_books = raw_books[mask].copy()

    # count words in each description
    complete_books["words_in_description"] = (
        complete_books["description"]
        .str.split()
        .str.len()
    )

    # keep only books with at least 25 words
    valid_books = complete_books[
        complete_books["words_in_description"] >= 25
    ].copy()

    # combine title and subtitle safely
    valid_books["title_and_subtitle"] = valid_books.apply(
        combine_title_and_subtitle,
        axis=1,
    )

    # tag description with ISBN
    valid_books["tagged_description"] = (
        valid_books[["isbn13", "description"]]
        .astype(str)
        .agg(" ".join, axis=1)
    )

    # drop helper columns
    drop_cols = [
        "subtitle",
        "missing_description",
        "age_of_book",
        "words_in_description",
    ]
    final_books = valid_books.drop(columns=drop_cols)

    return final_books


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
