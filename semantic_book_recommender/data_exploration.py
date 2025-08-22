#!/usr/bin/env python3

import os
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import kagglehub


def download_dataset() -> str:
    """Download the 7k-books-with-metadata dataset and return its directory."""
    warnings.filterwarnings("ignore")
    return kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")


def load_books(dataset_dir: str) -> pd.DataFrame:
    """Load books CSV into a DataFrame."""
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    csv_path = os.path.join(dataset_dir, "books.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Books CSV not found: {csv_path}")
    
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to load books CSV: {e}")


def combine_title_and_subtitle(row: pd.Series) -> Optional[str]:
    """Combine title and subtitle, handling NaN cases."""
    title, subtitle = row["title"], row["subtitle"]
    
    if pd.isna(title) and pd.isna(subtitle):
        return None
    if pd.isna(subtitle):
        return str(title)
    if pd.isna(title):
        return str(subtitle)
    
    return f"{str(title)}: {str(subtitle)}"


def filter_complete_books(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out books with missing essential fields."""
    required = ["description", "num_pages", "average_rating", "published_year"]
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Build a mask: keep rows with all required fields present
    mask = df["description"].notna()
    for col in required[1:]:
        mask &= df[col].notna()
    
    result = df[mask].copy()
    if result.empty:
        raise ValueError("No books remain after filtering for complete data")
    
    return result


def filter_by_description_length(
    df: pd.DataFrame,
    min_words: int = 25
) -> pd.DataFrame:
    """Filter books by minimum description word count."""
    if "description" not in df.columns:
        raise ValueError("DataFrame must contain 'description' column")
    
    df = df.copy()
    df["words_in_description"] = (
        df["description"].astype(str).str.split().str.len()
    )
    
    result = df[df["words_in_description"] >= min_words].copy()
    if result.empty:
        raise ValueError(f"No books remain with descriptions >= {min_words} words")
    
    return result.drop(columns=["words_in_description"])


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined title/subtitle and tagged description."""
    if df.empty:
        raise ValueError("Cannot process empty DataFrame")
    
    result = df.copy()
    result["title_and_subtitle"] = result.apply(
        combine_title_and_subtitle,
        axis=1,
    )
    result["tagged_description"] = (
        result[["isbn13", "description"]]
        .astype(str)
        .agg(" ".join, axis=1)
    )
    
    return result


def clean_books_data(
    raw_books: pd.DataFrame,
    min_description_words: int = 25
) -> pd.DataFrame:
    """
    Clean raw books data and return a filtered DataFrame.
    
    Steps:
      1. Remove rows missing essential fields
      2. Filter by description word count
      3. Add derived columns
      4. Drop redundant subtitle column
    """
    if raw_books.empty:
        raise ValueError("Input DataFrame is empty")
    
    books = filter_complete_books(raw_books)
    books = filter_by_description_length(books, min_description_words)
    books = add_derived_columns(books)
    
    if "subtitle" in books.columns:
        books = books.drop(columns=["subtitle"])
    
    return books