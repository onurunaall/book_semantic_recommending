#!/usr/bin/env python3

from typing import List, Any, Dict
import numpy as np
import pandas as pd
from transformers import pipeline, Pipeline
from tqdm import tqdm

def load_books() -> pd.DataFrame:
    return pd.read_csv("books_with_categories.csv")

def initialize_classifier() -> Pipeline:
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device="mps"
    )

def calculate_max_emotion_scores(predictions: List[Any]) -> Dict[str, float]:
    emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    
    # Create a dictionary to store scores for each emotion
    emotion_scores: Dict[str, List[float]] = {label: [] for label in emotion_labels}
    
    # Process each set of predictions
    for sentence_predictions in predictions:
        sorted_preds = sorted(sentence_predictions, key=lambda x: x["label"])
        
        # Append the score for each emotion based on the sorted order
        for idx, label in enumerate(emotion_labels):
            emotion_scores[label].append(sorted_preds[idx]["score"])
    
    return {label: np.max(scores) for label, scores in emotion_scores.items()}

def process_books(classifier: Any, books_df: pd.DataFrame) -> pd.DataFrame:
    emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    isbn_list: List[Any] = []
    
    # Prepare a dictionary to hold lists of maximum scores for each emotion
    all_emotion_scores: Dict[str, List[float]] = {label: [] for label in emotion_labels}

    # Loop over each book
    for index in tqdm(range(len(books_df)), desc="Computing emotion scores"):
        isbn_list.append(books_df["isbn13"].iloc[index])
        
        # Split the description into sentences using a simple period separator
        sentences = books_df["description"].iloc[index].split(".")
        
        predictions = classifier(sentences)
        
        max_scores = calculate_max_emotion_scores(predictions)
        
        for label in emotion_labels:
            all_emotion_scores[label].append(max_scores[label])
    
    # Create a DataFrame from the emotion scores and attach the corresponding ISBNs
    emotions_df = pd.DataFrame(all_emotion_scores)
    emotions_df["isbn13"] = isbn_list
    return emotions_df

def main() -> None:
    print("Loading books data with categories...")
    books_df: pd.DataFrame = load_books()
    
    print("Initializing emotion classifier...")
    emotion_classifier = initialize_classifier()
    
    emotions_df: pd.DataFrame = process_books(emotion_classifier, books_df)
    
    # Merge the emotion scores with the original book data based on ISBN
    books_with_emotions_df: pd.DataFrame = pd.merge(books_df, emotions_df, on="isbn13")
    books_with_emotions_df.to_csv("books_with_emotions.csv", index=False)
    print("Books with emotion scores saved as 'books_with_emotions.csv'.")

if __name__ == "__main__":
    main()
