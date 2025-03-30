#!/usr/bin/env python3

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from transformers import pipeline, Pipeline
from tqdm import tqdm

def load_books() -> pd.DataFrame:
    return pd.read_csv("books_cleaned.csv")

def map_categories(books_df: pd.DataFrame) -> pd.DataFrame:
    # Map original book categories to simplified categories.
    category_mapping: Dict[str, str] = {
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
    
    books_df["simple_categories"] = books_df["categories"].map(category_mapping)
    return books_df

def initialize_zero_shot_classifier() -> Pipeline:
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device="mps"
    )

def generate_prediction(text: str, candidate_labels: List[str], classifier: Pipeline) -> str:
    # (Single-call version; retained for reference but not used in batch processing)
    predictions: Dict[str, Any] = classifier(text, candidate_labels)
    
    max_index: int = int(np.argmax(predictions["scores"]))
    
    return predictions["labels"][max_index]

def predict_missing_categories(books_df: pd.DataFrame, classifier: Pipeline) -> pd.DataFrame:
    # Filter books where the simplified category is missing.
    missing_df: pd.DataFrame = books_df.loc[books_df["simple_categories"].isna(), ["isbn13", "description"]].reset_index(drop=True)
    
    # Batch process all descriptions for efficiency.
    descriptions: List[str] = missing_df["description"].tolist()
    predictions_list = classifier(descriptions, candidate_labels=["Fiction", "Nonfiction"])
    
    predicted_categories: List[str] = []
    for prediction in predictions_list:
        # For each prediction, choose the label with the highest score.
        max_index: int = int(np.argmax(prediction["scores"]))
        predicted_categories.append(prediction["labels"][max_index])
    
    return pd.DataFrame({
        "isbn13": missing_df["isbn13"],
        "predicted_categories": predicted_categories})

def main() -> None:
    print("Loading cleaned books data...")
    books_df: pd.DataFrame = load_books()
    books_df = map_categories(books_df)
    
    print("Initializing zero-shot classifier...")
    classifier: Pipeline = initialize_zero_shot_classifier()
    
    missing_preds_df: pd.DataFrame = predict_missing_categories(books_df, classifier)
    
    # Merge predictions with existing data and fill in missing categories.
    books_df = pd.merge(books_df, missing_preds_df, on="isbn13", how="left")
    books_df["simple_categories"] = books_df["simple_categories"].fillna(books_df["predicted_categories"])
    books_df.drop(columns=["predicted_categories"], inplace=True)
    
    books_df.to_csv("books_with_categories.csv", index=False)
    print("Books with simplified categories saved as 'books_with_categories.csv'.")

if __name__ == "__main__":
    main()
