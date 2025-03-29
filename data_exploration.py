#!/usr/bin/env python3

import os
import warnings
import numpy as np
import pandas as pd
import kagglehub

def download_dataset() -> str:
    # Turn off warnings and download the dataset from Kaggle
    warnings.filterwarnings("ignore")
    dataset_dir: str = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
    return dataset_dir

def load_books(dataset_dir: str) -> pd.DataFrame:
    # Build the path to the CSV file and load it into a DataFrame
    csv_path: str = os.path.join(dataset_dir, "books.csv")
    return pd.read_csv(csv_path)

def clean_books_data(raw_books: pd.DataFrame) -> pd.DataFrame:
    # Flag rows with missing descriptions and compute the age of each book
    raw_books["missing_description"] = np.where(raw_books["description"].isna(), 1, 0)
    raw_books["age_of_book"] = 2024 - raw_books["published_year"]

    # Remove rows missing any essential field
    complete_books: pd.DataFrame = raw_books[
        ~(raw_books["description"].isna()) &
        ~(raw_books["num_pages"].isna()) &
        ~(raw_books["average_rating"].isna()) &
        ~(raw_books["published_year"].isna())
    ].copy()

    # Count words in each description and retain only rows with at least 25 words
    complete_books["words_in_description"] = complete_books["description"].str.split().str.len()
    valid_books: pd.DataFrame = complete_books[complete_books["words_in_description"] >= 25].copy()

    # Combine title and subtitle if the subtitle exists; otherwise, use the title
    valid_books["title_and_subtitle"] = np.where(
        valid_books["subtitle"].isna(),
        valid_books["title"],
        valid_books[["title", "subtitle"]].astype(str).agg(": ".join, axis=1)
    )

    # Create a tagged description by concatenating the ISBN and the description
    valid_books["tagged_description"] = valid_books[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)

    # Remove helper columns no longer needed
    final_books: pd.DataFrame = valid_books.drop(
        ["subtitle", "missing_description", "age_of_book", "words_in_description"],
        axis=1
    )

    return final_books

def main() -> None:
    print("Downloading and cleaning the 7k books dataset...")
    dataset_dir: str = download_dataset()
    raw_books: pd.DataFrame = load_books(dataset_dir)
    cleaned_books: pd.DataFrame = clean_books_data(raw_books)
    cleaned_books.to_csv("books_cleaned.csv", index=False)
    print("Cleaned dataset saved as 'books_cleaned.csv'.")

if __name__ == "__main__":
    main()
